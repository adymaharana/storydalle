# ------------------------------------------------------------------------------------
# Minimal DALL-E
# Copyright (c) 2021 KakaoBrain. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------

import torch
from typing import Optional
from tqdm import tqdm
from torch.nn import functional as F


torch.set_printoptions(precision=2, threshold=10)
def cutoff_topk_logits(logits: torch.FloatTensor, k: int) -> torch.FloatTensor:
    if k is None:
        return logits
    else:
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[:, [-1]]] = -float('Inf')
        return out


def cutoff_topp_probs(probs: torch.FloatTensor, p: float) -> torch.FloatTensor:
    if p is None:
        return probs
    else:
        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
        cum_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_idx_remove_cond = cum_probs >= p

        sorted_idx_remove_cond[..., 1:] = sorted_idx_remove_cond[..., :-1].clone()
        sorted_idx_remove_cond[..., 0] = 0

        indices_to_remove = sorted_idx_remove_cond.scatter(-1, sorted_indices, sorted_idx_remove_cond)
        probs = probs.masked_fill(indices_to_remove, 0.0)
        norm_probs = probs / torch.sum(probs, dim=-1, keepdim=True)
        return norm_probs


def get_positional_encoding(inputs: torch.LongTensor, mode: str = '1d') -> torch.LongTensor:
    device = inputs.device
    if mode == '1d':
        B, N = inputs.shape
        xs_pos = torch.arange(N, device=device).repeat((B, 1))
    elif mode == '2d':
        B, H, W = inputs.shape
        xs_pos_h = torch.arange(H, device=device).repeat(B, W, 1).transpose(1, 2)
        xs_pos_w = torch.arange(W, device=device).repeat(B, H, 1)
        xs_pos = (xs_pos_h, xs_pos_w)
    else:
        raise ValueError('%s positional encoding invalid' % mode)
    return xs_pos


@torch.no_grad()
def sampling(model: torch.nn.Module,
             tokens: torch.LongTensor,
             top_k: Optional[float] = None,
             top_p: Optional[float] = None,
             softmax_temperature: float = 1.0,
             is_tqdm: bool = True,
             use_fp16: bool = True,
             max_seq_len: int = 256,
             prompt: Optional[torch.tensor] = None,
             pos_prompt: Optional[torch.Tensor] = None) -> torch.LongTensor:

    code = None
    past = None

    pbar = tqdm(range(max_seq_len), total=max_seq_len) if is_tqdm else range(max_seq_len)
    pos_enc_tokens = get_positional_encoding(tokens, mode='1d')

    for cnt, h in enumerate(pbar):
        if code is None:
            code_ = None
            pos_enc_code_ = None
        else:
            code_ = code.clone().detach()
            pos_enc_code_ = get_positional_encoding(code_, mode='1d')
            code_ = code_[:, cnt-1].unsqueeze(-1)
            pos_enc_code_ = pos_enc_code_[:, cnt-1].unsqueeze(-1)

        logits, present = model.sampling(images=code_,
                                         texts=tokens,
                                         pos_images=pos_enc_code_,
                                         pos_texts=pos_enc_tokens,
                                         use_fp16=use_fp16,
                                         past=past,
                                         prompt=prompt,
                                         pos_prompt=pos_prompt)

        logits = logits.to(dtype=torch.float32)
        logits = logits / softmax_temperature

        # print(len(present), present[0].shape)
        present = torch.stack(present).clone().detach()
        if past is None:
            past = [present]
        else:
            past.append(present)

        logits = cutoff_topk_logits(logits, top_k)
        probs = F.softmax(logits, dim=-1)
        probs = cutoff_topp_probs(probs, top_p)
        # print(probs[0])

        idx = torch.multinomial(probs, num_samples=1).clone().detach()
        # print(idx)
        code = idx if code is None else torch.cat([code, idx], axis=1)

    del past
    return code


@torch.no_grad()
def sampling_prefix(model: torch.nn.Module,
                    tokens: torch.LongTensor,
                    past: torch.FloatTensor,
                    top_k: Optional[float] = None,
                    top_p: Optional[float] = None,
                    softmax_temperature: float = 1.0,
                    is_tqdm: bool = True,
                    use_fp16: bool = True,
                    max_seq_len: int = 256,
                    labels = None) -> torch.LongTensor:
    code = None

    pbar = tqdm(range(max_seq_len), total=max_seq_len) if is_tqdm else range(max_seq_len)
    pos_enc_tokens = get_positional_encoding(tokens, mode='1d')

    # print("Entering sampling_prefix; ", past.shape)
    if past is not None:
        past = [past]

    for cnt, h in enumerate(pbar):
        if code is None:
            code_ = None
            pos_enc_code_ = None
        else:
            code_ = code.clone().detach()
            pos_enc_code_ = get_positional_encoding(code_, mode='1d')
            code_ = code_[:, cnt-1].unsqueeze(-1)
            pos_enc_code_ = pos_enc_code_[:, cnt-1].unsqueeze(-1)

        # print("Looop enter")
        # print(cnt, past[0].shape)
        # print("-------------------")
        logits, present = model.sampling(images=code_,
                                         texts=tokens,
                                         pos_images=pos_enc_code_,
                                         pos_texts=pos_enc_tokens,
                                         use_fp16=use_fp16,
                                         past=past)
        logits = logits.to(dtype=torch.float32)
        logits = logits / softmax_temperature

        present = torch.stack(present).clone().detach()

        # print('Present', present.shape)

        if past is None:
            past = [present]
        else:
            # print("Loop end")
            # print(present.shape)
            # print("-----------------")

            # n_layers, temp, _, seq_len, n_dim = present.shape
            # _, _, bs, n_heads, pre_seq_len, n_dim = past[0].shape
            # assert temp == 2
            # past.append(present.view(n_layers, temp, bs, n_heads, seq_len, n_dim))

            past.append(present)

        logits = cutoff_topk_logits(logits, top_k)
        probs = F.softmax(logits, dim=-1)
        probs = cutoff_topp_probs(probs, top_p)
        print(torch.topk(probs, 5, dim=-1))
        if labels is not None:
            print(labels[cnt])
        idx = torch.multinomial(probs, num_samples=1).clone().detach()
        # print(idx)
        code = idx if code is None else torch.cat([code, idx], axis=1)

    del past
    return code


@torch.no_grad()
def sampling_prefix_new(model: torch.nn.Module,
                    tokens: torch.LongTensor,
                    past: torch.FloatTensor,
                    top_k: Optional[float] = None,
                    top_p: Optional[float] = None,
                    softmax_temperature: float = 1.0,
                    is_tqdm: bool = True,
                    use_fp16: bool = True,
                    max_seq_len: int = 256) -> torch.LongTensor:
    code = None

    pbar = tqdm(range(max_seq_len), total=max_seq_len) if is_tqdm else range(max_seq_len)
    pos_enc_tokens = get_positional_encoding(tokens, mode='1d')

    # print("Entering sampling_prefix; ", past.shape)
    if past is not None:
        past = [past]

    for cnt, h in enumerate(pbar):
        if code is None:
            code_ = None
            pos_enc_code_ = None
        else:
            code_ = code.clone().detach()
            pos_enc_code_ = get_positional_encoding(code_, mode='1d')
            # code_ = code_[:, cnt-1].unsqueeze(-1)
            # pos_enc_code_ = pos_enc_code_[:, cnt-1].unsqueeze(-1)

        # print("Looop enter")
        # print(cnt, past[0].shape)
        # print("-------------------")

        if cnt == 0:
            logits, present = model.sampling(images=code_,
                                             texts=tokens,
                                             pos_images=pos_enc_code_,
                                             pos_texts=pos_enc_tokens,
                                             use_fp16=use_fp16,
                                             past=past)
            logits = logits.to(dtype=torch.float32)
            logits = logits / softmax_temperature

            present = torch.stack(present).clone().detach()

            # print('Present', present.shape)

            if past is None:
                past = [present]
            else:
                pass

            logits = cutoff_topk_logits(logits, top_k)
            probs = F.softmax(logits, dim=-1)
            probs = cutoff_topp_probs(probs, top_p)
            # print(torch.topk(probs[0], 5))
            idx = torch.multinomial(probs, num_samples=1).clone().detach()
            # print(idx)
            code = idx if code is None else torch.cat([code, idx], axis=1)

        else:
            pass


    del past
    return code

@torch.no_grad()
def sampling_conditional(model: torch.nn.Module,
                         cross_attention_idxs,
                         cross_attention_layers,
                    tokens: torch.LongTensor,
                    src_codes: torch.FloatTensor,
                    top_k: Optional[float] = None,
                    top_p: Optional[float] = None,
                    softmax_temperature: float = 1.0,
                    is_tqdm: bool = True,
                    use_fp16: bool = True,
                    max_seq_len: int = 256,
                    prompt: Optional[torch.tensor] = None,
                    pos_prompt: Optional[torch.Tensor] = None) -> torch.LongTensor:

    code = None
    past = None

    pbar = tqdm(range(max_seq_len), total=max_seq_len) if is_tqdm else range(max_seq_len)
    pos_enc_tokens = get_positional_encoding(tokens, mode='1d')

    src_pos_tokens = get_positional_encoding(src_codes, mode='1d')
    src_tokens = model.tok_emb_img(src_codes)
    src_tokens = src_tokens + model.pos_emb_img(src_pos_tokens)

    for cnt, h in enumerate(pbar):
        if code is None:
            code_ = None
            pos_enc_code_ = None
        else:
            code_ = code.clone().detach()
            pos_enc_code_ = get_positional_encoding(code_, mode='1d')
            code_ = code_[:, cnt-1].unsqueeze(-1)
            pos_enc_code_ = pos_enc_code_[:, cnt-1].unsqueeze(-1)

        logits, present = model.sampling_with_context(images=code_,
                                                      cross_attention_idxs=cross_attention_idxs,
                                                      cross_attention_layers=cross_attention_layers,
                                                      texts=tokens,
                                                      pos_images=pos_enc_code_,
                                                      pos_texts=pos_enc_tokens,
                                                      source_image=src_tokens,
                                                      use_fp16=use_fp16,
                                                      past=past,
                                                      prompt=prompt,
                                                      pos_prompt=pos_prompt)
        logits = logits.to(dtype=torch.float32)
        logits = logits / softmax_temperature

        present = torch.stack(present).clone().detach()
        if past is None:
            past = [present]
        else:
            past.append(present)

        logits = cutoff_topk_logits(logits, top_k)
        probs = F.softmax(logits, dim=-1)
        probs = cutoff_topp_probs(probs, top_p)

        idx = torch.multinomial(probs, num_samples=1).clone().detach()
        code = idx if code is None else torch.cat([code, idx], axis=1)

    del past
    return code


@torch.no_grad()
def sampling_igpt(model: torch.nn.Module,
                  sos: torch.FloatTensor,
                  top_k: Optional[float] = None,
                  top_p: Optional[float] = None,
                  softmax_temperature: float = 1.0,
                  is_tqdm: bool = True,
                  use_fp16: bool = True,
                  max_seq_len: int = 256) -> torch.LongTensor:
    code = None
    past = None
    pbar = tqdm(range(max_seq_len), total=max_seq_len) if is_tqdm else range(max_seq_len)

    for cnt, h in enumerate(pbar):
        if code is None:
            code_ = None
            pos_enc_code_ = None
        else:
            code_ = code.clone().detach()
            pos_enc_code_ = get_positional_encoding(code_, mode='1d')
            code_ = code_[:, cnt-1].unsqueeze(-1)
            pos_enc_code_ = pos_enc_code_[:, cnt-1].unsqueeze(-1)

        logits, present = model.sampling(sos=sos,
                                         codes=code_,
                                         pos_codes=pos_enc_code_,
                                         use_fp16=use_fp16,
                                         past=past)
        logits = logits.to(dtype=torch.float32)
        logits = logits / softmax_temperature

        present = torch.stack(present).clone().detach()
        if past is None:
            past = [present]
        else:
            past.append(present)

        logits = cutoff_topk_logits(logits, top_k)
        probs = F.softmax(logits, dim=-1)
        probs = cutoff_topp_probs(probs, top_p)

        idx = torch.multinomial(probs, num_samples=1).clone().detach()
        code = idx if code is None else torch.cat([code, idx], axis=1)

    del past
    return code
