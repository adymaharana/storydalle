# ------------------------------------------------------------------------------------
# Minimal DALL-E
# Copyright (c) 2021 KakaoBrain. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------
# Modified from minGPT (https://github.com/karpathy/minGPT)
# Copyright (c) 2020 Andrej Karpathy. All Rights Reserved.
# ------------------------------------------------------------------------------------

import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from torch.cuda.amp import autocast
from omegaconf import OmegaConf
from .layers import Block

class Transformer1d(nn.Module):

    def __init__(self,
                 vocab_size_txt: int,
                 vocab_size_img: int,
                 hparams: OmegaConf) -> None:
        super().__init__()
        assert hparams.n_layers == hparams.n_dense_layers

        # input embedding for image and text
        self.tok_emb_img = nn.Embedding(vocab_size_img, hparams.embed_dim)
        self.tok_emb_txt = nn.Embedding(vocab_size_txt, hparams.embed_dim)

        self.pos_emb_img = nn.Embedding(hparams.ctx_len_img, hparams.embed_dim)
        self.pos_emb_txt = nn.Embedding(hparams.ctx_len_txt, hparams.embed_dim)

        self.drop = nn.Dropout(hparams.embd_pdrop)

        # transformer blocks
        self.blocks = [Block(ctx_len=hparams.ctx_len_img + hparams.ctx_len_txt,
                             embed_dim=hparams.embed_dim,
                             n_heads=hparams.n_heads,
                             mlp_bias=hparams.mlp_bias,
                             attn_bias=hparams.attn_bias,
                             resid_pdrop=hparams.resid_pdrop,
                             attn_pdrop=hparams.attn_pdrop,
                             gelu_use_approx=hparams.gelu_use_approx) for i in range(1, hparams.n_layers+1)]
        self.blocks = nn.Sequential(*self.blocks)

        # heads for image and text
        self.ln_f = nn.LayerNorm(hparams.embed_dim)
        self.head_img = nn.Linear(hparams.embed_dim, vocab_size_img, bias=False)
        self.head_txt = nn.Linear(hparams.embed_dim, vocab_size_txt, bias=False)

        self.ctx_len_img = hparams.ctx_len_img
        self.ctx_len_txt = hparams.ctx_len_txt
        self.n_layers = hparams.n_layers

        self.apply(self._init_weights)


    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


    def resize_token_embeddings(self, new_num_tokens):

        old_num_tokens, old_embedding_dim = self.tok_emb_txt.weight.size()
        new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim)
        new_embeddings.to(self.tok_emb_txt.weight.device, dtype=self.tok_emb_txt.weight.dtype)
        self._init_weights(new_embeddings)
        # numbers of tokens to copy
        n = min(old_num_tokens, new_num_tokens)
        new_embeddings.weight.data[:n, :] = self.tok_emb_txt.weight.data[:n, :]
        self.tok_emb_txt = new_embeddings

        self.resize_lm_head(new_num_tokens)
        # TODO: also change config to reflect new vocab size

        return new_embeddings


    def resize_lm_head(
        self, new_num_tokens: Optional[int] = None, transposed: Optional[bool] = False) -> nn.Linear:

        old_num_tokens, old_lm_head_dim = (
            self.head_txt.weight.size() if not transposed else self.head_txt.weight.t().size()
        )
        # Build new lm head
        new_lm_head_shape = (old_lm_head_dim, new_num_tokens) if not transposed else (new_num_tokens, old_lm_head_dim)
        has_new_lm_head_bias = self.head_txt.bias is not None
        new_lm_head = nn.Linear(*new_lm_head_shape, bias=has_new_lm_head_bias)
        new_lm_head = new_lm_head.to(self.head_txt.weight.device, dtype=self.head_txt.weight.dtype)

        # initialize new lm head (in particular added tokens)
        self._init_weights(new_lm_head)
        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
        # Copy old lm head weights to new lm head
        if not transposed:
            new_lm_head.weight.data[:num_tokens_to_copy, :] = self.head_txt.weight.data[:num_tokens_to_copy, :]
        else:
            new_lm_head.weight.data[:, :num_tokens_to_copy] = self.head_txt.weight.data[:, :num_tokens_to_copy]

        # Copy bias weights to new lm head
        if has_new_lm_head_bias:
            new_lm_head.bias.data[:num_tokens_to_copy] = self.head_txt.bias.data[:num_tokens_to_copy]

        self.head_txt = new_lm_head

        return new_lm_head


    def forward(self,
                images: torch.LongTensor,
                texts: torch.LongTensor,
                pos_images: torch.LongTensor,
                pos_texts: torch.LongTensor,
                past: Optional[List[torch.Tensor]] = None,
                prompt: Optional[List[torch.Tensor]] = None,
                pos_prompt: Optional[List[torch.Tensor]] = None) -> Tuple[torch.FloatTensor, torch.FloatTensor]:


        B, T = images.shape
        _, N = texts.shape

        assert T <= self.ctx_len_img, "Already reached the maximum context length (image)."
        assert N == self.ctx_len_txt, "Already reached the maximum context length (text)."

        texts = self.tok_emb_txt(texts)
        images = self.tok_emb_img(images)

        texts = texts + self.pos_emb_txt(pos_texts)
        images = images + self.pos_emb_img(pos_images)

        if prompt is not None:
            prompt = prompt + self.pos_emb_txt(pos_prompt)
            texts = torch.cat([prompt, texts], dim=1).contiguous()
            P = prompt.shape[1]

        x = torch.cat([texts, images], dim=1).contiguous()
        x = self.drop(x)

        # x = self.blocks(x)
        for i, block in enumerate(self.blocks):
            x, _ = block.sample(x, layer_past=None if past is None else past[i])

        x = self.ln_f(x)

        if prompt is not None:
            texts = x[:, P:N+P-1].contiguous()
            images = x[:, N+P-1:-1].contiguous()
        else:
            texts = x[:, :N-1].contiguous()
            images = x[:, N-1:-1].contiguous()

        logits_txt = self.head_txt(texts)
        logits_img = self.head_img(images)
        return logits_img, logits_txt

    def forward_with_context(self,
                images: torch.LongTensor,
                texts: torch.LongTensor,
                pos_images: torch.LongTensor,
                pos_texts: torch.LongTensor,
                src_images: torch.LongTensor,
                src_pos_images: torch.LongTensor,
                cross_attention_idxs: List,
                cross_attention_layers,
                past: Optional[List[torch.Tensor]] = None,
                prompt: Optional[List[torch.Tensor]] = None,
                pos_prompt: Optional[List[torch.Tensor]] = None) -> Tuple[torch.FloatTensor, torch.FloatTensor]:


        B, T = images.shape
        _, N = texts.shape

        assert T <= self.ctx_len_img, "Already reached the maximum context length (image)."
        assert N == self.ctx_len_txt, "Already reached the maximum context length (text)."

        texts = self.tok_emb_txt(texts)
        images = self.tok_emb_img(images)
        src_images = self.tok_emb_img(src_images)

        texts = texts + self.pos_emb_txt(pos_texts)
        images = images + self.pos_emb_img(pos_images)
        src_images = src_images + self.pos_emb_img(src_pos_images)

        if prompt is not None:
            prompt = prompt + self.pos_emb_txt(pos_prompt)
            texts = torch.cat([prompt, texts], dim=1).contiguous()
            P = prompt.shape[1]
        else:
            P = 0

        x = torch.cat([texts, images], axis=1).contiguous()
        x = self.drop(x)

        # prepare mask
        mask = torch.zeros_like(x[0])
        mask[self.ctx_len_txt+P-1:, :].fill_(1.0)
        mask = mask.unsqueeze(0)

        # print(images.shape, texts.shape, src_images.shape, mask.shape, x.shape)

        # x = self.blocks(x)
        for i, block in enumerate(self.blocks):
            if i in cross_attention_idxs:
                x, _ = block.sample_with_context(x, src_images, mask, cross_attention_layers[int(((i+1)/3)-1)], layer_past=None if past is None else past[i])
            else:
                x, _ = block.sample(x, layer_past=None if past is None else past[i])

        x = self.ln_f(x)

        if prompt is not None:
            texts = x[:, P:N+P-1].contiguous()
            images = x[:, N+P-1:-1].contiguous()
        else:
            texts = x[:, :N-1].contiguous()
            images = x[:, N-1:-1].contiguous()

        logits_txt = self.head_txt(texts)
        logits_img = self.head_img(images)
        return logits_img, logits_txt

    @torch.no_grad()
    def sampling(self,
                 images: torch.LongTensor,
                 texts: torch.LongTensor,
                 pos_images: torch.LongTensor,
                 pos_texts: torch.LongTensor,
                 use_fp16: bool = True,
                 past: Optional[List[torch.Tensor]] = None,
                 prompt: Optional[List[torch.Tensor]] = None,
                 pos_prompt: Optional[List[torch.Tensor]] = None) -> Tuple[torch.FloatTensor, List[torch.FloatTensor]]:

        _, N = texts.shape
        assert N == self.ctx_len_txt, "Already reached the maximum context length (text)."

        with autocast(enabled=use_fp16):
            if images is None:
                # assert past is None

                texts = self.tok_emb_txt(texts)
                x = texts + self.pos_emb_txt(pos_texts)

                if prompt is not None:
                    prompt = prompt + self.pos_emb_txt(pos_prompt)
                    texts = torch.cat([prompt, texts], dim=1).contiguous()

                x = self.drop(x)

                if past is not None:
                    past = torch.cat(past, dim=-2)

                presents = []
                for i, block in enumerate(self.blocks):
                    x, present = block.sample(x, layer_past=None if past is None else past[i])
                    presents.append(present)
                x = self.ln_f(x)
                x = x[:, N-1].contiguous()
                logits = self.head_img(x)
            else:
                if past is None:
                    texts = self.tok_emb_txt(texts)
                    images = self.tok_emb_img(images)
                    texts = texts + self.pos_emb_txt(pos_texts)
                    images = images + self.pos_emb_img(pos_images)

                    if prompt is not None:
                        prompt = prompt + self.pos_emb_txt(pos_prompt)
                        texts = torch.cat([prompt, texts], dim=1).contiguous()

                    x = torch.cat([texts, images], axis=1).contiguous()
                else:
                    images = self.tok_emb_img(images)
                    x = images + self.pos_emb_img(pos_images)
                x = self.drop(x)

                # if past is not None and len(past) > 1:
                if past is not None:
                    past = torch.cat(past, dim=-2)
                    # print('Past', past.shape)
                presents = []
                # print(len(past), past[0].shape)
                for i, block in enumerate(self.blocks):
                    x, present = block.sample(x, layer_past=None if past is None else past[i])
                    presents.append(present)
                x = self.ln_f(x)
                x = x[:, -1].contiguous()
                logits = self.head_img(x)
            return logits, presents

    @torch.no_grad()
    def sampling_with_context(self,
                              images: torch.LongTensor,
                              cross_attention_idxs,
                              cross_attention_layers,
                              texts: torch.LongTensor,
                              pos_images: torch.LongTensor,
                              pos_texts: torch.LongTensor,
                              source_image: torch.LongTensor,
                              use_fp16: bool = True,
                              past: Optional[List[torch.Tensor]] = None,
                              prompt: Optional[List[torch.Tensor]] = None,
                              pos_prompt: Optional[List[torch.Tensor]] = None
                              ) -> Tuple[torch.FloatTensor, List[torch.FloatTensor]]:

        _, N = texts.shape
        assert N == self.ctx_len_txt, "Already reached the maximum context length (text)."

        if prompt is not None:
            P = prompt.shape[1]
        else:
            P = 0

        with autocast(enabled=use_fp16):
            if images is None:
                # assert past is None

                texts = self.tok_emb_txt(texts)
                texts = texts + self.pos_emb_txt(pos_texts)

                if prompt is not None:
                    prompt = prompt + self.pos_emb_txt(pos_prompt)
                    texts = torch.cat([prompt, texts], dim=1).contiguous()

                x = self.drop(texts)

                if past is not None:
                    past = torch.cat(past, dim=-2)

                # prepare mask
                mask = torch.zeros_like(x[0])
                mask[self.ctx_len_txt+P - 1:, :].fill_(1.0)
                mask = mask.unsqueeze(0)

                presents = []
                for i, block in enumerate(self.blocks):
                    if i in cross_attention_idxs:
                        x, present = block.sample_with_context(x, source_image, mask,
                                                         cross_attention_layers[int(((i + 1) / 3) - 1)],
                                                         layer_past=None if past is None else past[i])
                    else:
                        x, present = block.sample(x, layer_past=None if past is None else past[i])
                    presents.append(present)
                x = self.ln_f(x)
                x = x[:, N-1].contiguous()
                logits = self.head_img(x)
            else:
                if past is None:
                    texts = self.tok_emb_txt(texts)
                    images = self.tok_emb_img(images)
                    texts = texts + self.pos_emb_txt(pos_texts)
                    images = images + self.pos_emb_img(pos_images)

                    if prompt is not None:
                        prompt = prompt + self.pos_emb_txt(pos_prompt)
                        texts = torch.cat([prompt, texts], dim=1).contiguous()

                    x = torch.cat([texts, images], axis=1).contiguous()
                else:
                    images = self.tok_emb_img(images)
                    x = images + self.pos_emb_img(pos_images)
                x = self.drop(x)

                # if past is not None and len(past) > 1:
                if past is not None:
                    past = torch.cat(past, dim=-2)
                presents = []

                # prepare mask
                mask = torch.zeros_like(x[0])
                mask[self.ctx_len_txt+P - 1:, :].fill_(1.0)
                mask = mask.unsqueeze(0)

                # print(len(past), past[0].shape)
                for i, block in enumerate(self.blocks):
                    if i in cross_attention_idxs:
                        x, present = block.sample_with_context(x, source_image, mask,
                                                         cross_attention_layers[int(((i + 1) / 3) - 1)],
                                                         layer_past=None if past is None else past[i])
                    else:
                        x, present = block.sample(x, layer_past=None if past is None else past[i])
                    presents.append(present)
                x = self.ln_f(x)
                x = x[:, -1].contiguous()
                logits = self.head_img(x)
            return logits, presents

    def from_ckpt(self, path: str) -> None:
        ckpt = torch.load(path, map_location='cpu')['state_dict']
        self.load_state_dict(ckpt, strict=True)
        print(f'{path} succesfully restored..')


class iGPT(nn.Module):
    def __init__(self,
                 vocab_size_img: int,
                 use_cls_cond: bool,
                 hparams: OmegaConf) -> None:
        super().__init__()
        self.use_cls_cond = use_cls_cond

        # sos token embedding
        if self.use_cls_cond:
            self.sos = nn.Embedding(hparams.n_classes, hparams.embed_dim)
        else:
            self.sos = nn.Parameter(torch.randn(1, 1, hparams.embed_dim))

        # input embedding
        self.tok_emb_img = nn.Embedding(vocab_size_img, hparams.embed_dim)
        self.pos_emb_img = nn.Embedding(hparams.ctx_len_img, hparams.embed_dim)

        self.drop = nn.Dropout(hparams.embd_pdrop)

        # transformer blocks
        self.blocks = [Block(ctx_len=hparams.ctx_len_img + 1,
                             embed_dim=hparams.embed_dim,
                             n_heads=hparams.n_heads,
                             mlp_bias=hparams.mlp_bias,
                             attn_bias=hparams.attn_bias,
                             resid_pdrop=hparams.resid_pdrop,
                             attn_pdrop=hparams.attn_pdrop,
                             gelu_use_approx=hparams.gelu_use_approx) for i in range(1, hparams.n_layers+1)]
        self.blocks = nn.Sequential(*self.blocks)

        # head
        self.ln_f = nn.LayerNorm(hparams.embed_dim)
        self.head = nn.Linear(hparams.embed_dim, vocab_size_img, bias=False)

        self.ctx_len_img = hparams.ctx_len_img
        self.n_layers = hparams.n_layers

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    @torch.no_grad()
    def sampling(self,
                 sos: torch.FloatTensor,
                 codes: torch.LongTensor,
                 pos_codes: torch.LongTensor,
                 n_samples: int = 16,
                 use_fp16: bool = True,
                 past: Optional[torch.Tensor] = None) -> Tuple[torch.FloatTensor, List[torch.FloatTensor]]:
        with autocast(enabled=use_fp16):
            if codes is None:
                assert past is None
                xs = self.drop(sos)
                presents = []
                for i, block in enumerate(self.blocks):
                    xs, present = block.sample(xs, layer_past=None)
                    presents.append(present)
                xs = self.ln_f(xs)
                logits = self.head(xs)[:, -1]
            else:
                if past is None:
                    xs = self.tok_emb_img(codes) + self.pos_emb_img(pos_codes)
                    xs = torch.cat([sos, xs], dim=1)
                else:
                    xs = self.tok_emb_img(codes) + self.pos_emb_img(pos_codes)
                xs = self.drop(xs)

                past = torch.cat(past, dim=-2) if past is not None else past
                presents = []
                for i, block in enumerate(self.blocks):
                    xs, present = block.sample(xs, layer_past=None if past is None else past[i])
                    presents.append(present)

                xs = self.ln_f(xs)
                logits = self.head(xs)[:, -1]
            return logits, presents

    def forward(self,
                codes: torch.LongTensor,
                labels: Optional[torch.LongTensor] = None) -> torch.FloatTensor:
        B, T = codes.shape
        xps = torch.arange(T, device=codes.device).repeat((B, 1))
        sos = self.sos.repeat((B, 1, 1)) if labels is None else self.sos(labels).unsqueeze(1)

        h = self.tok_emb_img(codes) + self.pos_emb_img(xps)
        h = torch.cat([sos, h[:, :-1]], dim=1).contiguous()

        h = self.drop(h)
        h = self.blocks(h)
        h = self.ln_f(h)
        logits = self.head(h)
        return logits

    def from_ckpt(self, path: str, strict: bool = True) -> None:
        ckpt = torch.load(path, map_location='cpu')['state_dict']
        self.load_state_dict(ckpt, strict=strict)
        print(f'{path} successfully restored..')
