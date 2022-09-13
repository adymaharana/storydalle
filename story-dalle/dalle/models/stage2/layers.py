# ------------------------------------------------------------------------------------
# Minimal DALL-E
# Copyright (c) 2021 KakaoBrain. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------
# Modified from minGPT (https://github.com/karpathy/minGPT)
# Copyright (c) 2020 Andrej Karpathy. All Rights Reserved.
# ------------------------------------------------------------------------------------

import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class GELU(nn.Module):
    def __init__(self, use_approx=False):
        super().__init__()
        self.use_approx = use_approx

    def forward(self, x):
        if self.use_approx:
            return x * torch.sigmoid(1.702 * x)
        else:
            return F.gelu(x)


class MultiHeadSelfAttention(nn.Module):

    def __init__(self,
                 ctx_len: int,
                 embed_dim: int,
                 n_heads: int,
                 resid_pdrop: float,
                 attn_pdrop: float,
                 attn_bias: bool,
                 use_mask: bool = True):
        super().__init__()
        assert embed_dim % n_heads == 0

        # key, query, value projections for all heads
        self.key = nn.Linear(embed_dim, embed_dim, bias=attn_bias)
        self.query = nn.Linear(embed_dim, embed_dim, bias=attn_bias)
        self.value = nn.Linear(embed_dim, embed_dim, bias=attn_bias)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        # output projection
        self.proj = nn.Linear(embed_dim, embed_dim, attn_bias)

        self.n_heads = n_heads
        self.ctx_len = ctx_len
        self.use_mask = use_mask
        if self.use_mask:
            self.register_buffer("mask", torch.ones(ctx_len, ctx_len), persistent=False)
            self.mask = torch.tril(self.mask).view(1, ctx_len, ctx_len)

    def forward(self, x, use_cache=False, layer_past=None):
        B, T, C = x.shape
        x = x.transpose(0, 1).contiguous()  # (B, T, C) -> (T, B, C)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(T, B*self.n_heads, C//self.n_heads).transpose(0, 1)  # (B*nh, T, hs)
        q = self.query(x).view(T, B*self.n_heads, C//self.n_heads).transpose(0, 1)  # (B*nh, T, hs)
        v = self.value(x).view(T, B*self.n_heads, C//self.n_heads).transpose(0, 1)  # (B*nh, T, hs)

        if use_cache:
            present = torch.stack([k, v])

        if layer_past is not None:
            # print(layer_past.shape, k.shape, v.shape, q.shape)
            # print("LayerPast shape", layer_past.shape)
            past_key, past_value = layer_past

            if len(past_key.shape) == 4:
                _, _, seq_len, dim = past_key.shape
                k = torch.cat([past_key.reshape(-1, seq_len, dim), k], dim=-2)
                v = torch.cat([past_value.reshape(-1, seq_len, dim), v], dim=-2)
            elif len(past_key.shape) == 3:
                past_key, past_value = layer_past
                k = torch.cat([past_key, k], dim=-2)
                v = torch.cat([past_value, v], dim=-2)
            else:
                raise ValueError

        if use_cache and layer_past is not None:
            # Tensor shape below: (B * nh, 1, hs) X (B * nh, hs, K) -> (B * nh, 1, K)
            att = torch.bmm(q, (k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))))
            att = F.softmax(att, dim=-1)
            att = self.attn_drop(att)
            y = torch.bmm(att, v)  # (B*nh, 1, K) X (B*nh, K, hs) -> (B*nh, 1, hs)
        else:
            # Tensor shape below: (B * nh, T, hs) X (B * nh, hs, T) -> (B * nh, T, T)
            att = torch.bmm(q, (k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))))
            if self.use_mask:
                # TODO : Flip when not prompt tunign
                # mask = self.mask if T == self.ctx_len else self.mask[:, :T, :T]
                if T == self.ctx_len:
                    mask = self.mask
                else:
                    mask = torch.tril(torch.ones(T, T)).view(1, T, T).to(att.device)
                att = att.masked_fill(mask == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_drop(att)
            y = torch.bmm(att, v)  # (B*nh, T, T) X (B*nh, T, hs) -> (B*nh, T, hs)
        y = y.transpose(0, 1).contiguous().view(T, B, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        if use_cache:
            return y.transpose(0, 1).contiguous(), present  # (T, B, C) -> (B, T, C)
        else:
            return y.transpose(0, 1).contiguous()  # (T, B, C) -> (B, T, C)

    def forward_with_context(self, x, context, mask=None):
        B, T, C = x.shape
        x = x.transpose(0, 1).contiguous()  # (B, T, C) -> (T, B, C)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.query(x).view(T, B*self.n_heads, C//self.n_heads).transpose(0, 1)  # (B*nh, T, hs)

        B, T_c, C = context.shape
        k = self.key(context).view(T_c, B * self.n_heads, C // self.n_heads).transpose(0, 1)  # (B*nh, T, hs)
        v = self.value(context).view(T_c, B*self.n_heads, C//self.n_heads).transpose(0, 1)  # (B*nh, T, hs)

        # Tensor shape below: (B * nh, T, hs) X (B * nh, hs, Tc) -> (B * nh, T, Tc)
        att = torch.bmm(q, (k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = torch.bmm(att, v)  # (B*nh, T, T) X (B*nh, T, hs) -> (B*nh, T, hs)
        y = y.transpose(0, 1).contiguous().view(T, B, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y)).transpose(0, 1).contiguous()
        if mask is not None:
            y = y.masked_fill(mask == 0, float('0.0'))
        return y  # (T, B, C) -> (B, T, C)


class Block(nn.Module):

    def __init__(self,
                 ctx_len: int,
                 embed_dim: int,
                 n_heads: int,
                 mlp_bias: bool,
                 attn_bias: bool,
                 resid_pdrop: bool,
                 attn_pdrop: bool,
                 gelu_use_approx: bool):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

        self.attn = MultiHeadSelfAttention(ctx_len=ctx_len,
                                           embed_dim=embed_dim,
                                           n_heads=n_heads,
                                           attn_pdrop=attn_pdrop,
                                           resid_pdrop=resid_pdrop,
                                           attn_bias=attn_bias,
                                           use_mask=True)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim, bias=mlp_bias),
            GELU(gelu_use_approx),
            nn.Linear(4 * embed_dim, embed_dim, bias=mlp_bias),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x, layer_past=None):
        x = x + self.attn(self.ln1(x), layer_past=layer_past)
        x = x + self.mlp(self.ln2(x))
        return x

    def sample(self, x, layer_past=None):
        attn, present = self.attn(self.ln1(x), use_cache=True, layer_past=layer_past)
        x = x + attn
        x = x + self.mlp(self.ln2(x))
        return x, present

    def sample_with_context(self, x, context, context_mask, cross_attn_layer, layer_past=None):
        attn, present = self.attn(self.ln1(x), use_cache=True, layer_past=layer_past)
        x = x + attn
        c_attn = cross_attn_layer(x, context, context_mask)
        x = x + c_attn
        x = x + self.mlp(self.ln2(x))
        return x, present


class CrossAttentionLayer(nn.Module):

    def __init__(self,
                 ctx_len: int,
                 embed_dim: int,
                 n_heads: int,
                 attn_bias: bool,
                 resid_pdrop: bool,
                 attn_pdrop: bool):
        super().__init__()

        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(ctx_len=ctx_len,
                                           embed_dim=embed_dim,
                                           n_heads=n_heads,
                                           attn_pdrop=attn_pdrop,
                                           resid_pdrop=resid_pdrop,
                                           attn_bias=attn_bias,
                                           use_mask=False)

    def forward(self, x, context, context_mask=None):
        attn = self.attn.forward_with_context(self.ln1(x), self.ln2(context), context_mask)
        # x = x + attn
        # return x
        return attn