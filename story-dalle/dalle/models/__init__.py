# ------------------------------------------------------------------------------------
# Minimal DALL-E
# Copyright (c) 2021 KakaoBrain. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------

import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Optional, Tuple, Union
from omegaconf import OmegaConf
from torch.cuda.amp import autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.nn import functional as F
from .stage1.vqgan import VQGAN
from .stage2.transformer import Transformer1d, iGPT
from .stage2.layers import Block
from .. import utils
from ..utils.config import get_base_config
from ..utils.sampling import sampling, sampling_igpt, get_positional_encoding, sampling_prefix, sampling_conditional
from ..utils.utils import save_image
from .tokenizer import build_tokenizer
import numpy as np
from .stage2.layers import CrossAttentionLayer

_MODELS = {
    'minDALL-E/1.3B': 'https://arena.kakaocdn.net/brainrepo/models/minDALL-E/57b008f02ceaa02b779c8b7463143315/1.3B.tar.gz'
}

class Dalle(pl.LightningModule):
    def __init__(self,
                 config: OmegaConf) -> None:
        super().__init__()
        self.tokenizer = None
        self.stage1 = VQGAN(n_embed=config.stage1.n_embed,
                            embed_dim=config.stage1.embed_dim,
                            hparams=config.stage1.hparams)
        self.stage2 = Transformer1d(vocab_size_txt=config.stage2.vocab_size_txt,
                                    vocab_size_img=config.stage2.vocab_size_img,
                                    hparams=config.stage2.hparams)
        self.config = config
        self.config_stage1 = config.stage1
        self.config_stage2 = config.stage2
        self.config_dataset = config.dataset

        # # make the parameters in stage 1 not trainable
        # self.stage1.eval()
        # for p in self.stage1.parameters():
        #     p.requires_grad = False

    @classmethod
    def from_pretrained(cls, args) -> Tuple[nn.Module, OmegaConf]:

        path = args.model_name_or_path
        config_new = OmegaConf.load(os.path.join(path, 'config.yaml'))
        if args.do_train:
            config_base = get_base_config('finetuning')
            config_update = OmegaConf.merge(config_base, config_new)
            for key, val in vars(args).items():
                if key in config_update.optimizer.keys():
                    OmegaConf.update(config_update, "optimizer.%s" % key, val, merge=False)
                if key in config_update.experiment.keys():
                    OmegaConf.update(config_update, "experiment.%s" % key, val, merge=False)
        else:
            config_base = get_base_config('default')
            config_update = OmegaConf.merge(config_base, config_new)

        model = cls(config_update)
        model.tokenizer = build_tokenizer(os.path.join(path, 'tokenizer'),
                                          context_length=model.config_dataset.context_length,
                                          lowercase=True,
                                          dropout=None)

        print("Loading models from checkpoint %s" % path)

        if hasattr(args, 'dalle_path') and args.dalle_path and args.dalle_path.endswith('.pth'):
            model.load_state_dict(torch.load(args.dalle_path)["model_state_dict"])
        else:
            model.stage1.from_ckpt(os.path.join(path, 'stage1_last.ckpt'))
            model.stage2.from_ckpt(os.path.join(path, 'stage2_last.ckpt'))

        return model, config_update
        

    @torch.no_grad()
    def sampling(self,
                 prompt: Union[str, torch.LongTensor],
                 top_k: int = 256,
                 top_p: Optional[float] = None,
                 softmax_temperature: float = 1.0,
                 num_candidates: int = 96,
                 device: str = 'cuda:0',
                 use_fp16: bool = True) -> torch.FloatTensor:
        self.stage1.eval()
        self.stage2.eval()

        if type(prompt) == str:
            tokens = self.tokenizer.encode(prompt)
            tokens = torch.LongTensor(tokens.ids)
        else:
            tokens = prompt
        tokens = torch.repeat_interleave(tokens.unsqueeze(0), num_candidates, dim=0)

        # Check if the encoding works as intended
        # print(self.tokenizer.decode_batch(tokens.tolist(), skip_special_tokens=True)[0])

        tokens = tokens.to(device)
        codes = sampling(self.stage2,
                         tokens,
                         top_k=top_k,
                         top_p=top_p,
                         softmax_temperature=softmax_temperature,
                         use_fp16=use_fp16)
        codes = codes.view(num_candidates, 16, 16)  # [B, 16, 16]
        pixels = torch.clamp(self.stage1.decode_code(codes) * 0.5 + 0.5, 0, 1)  # [B, 256, 256]
        return pixels

    def forward(self,
                images: torch.FloatTensor,
                texts: Optional[torch.LongTensor],
                past=None
                ) -> tuple:
        B, C, H, W = images.shape
        with torch.no_grad():
            with autocast(enabled=False):
                codes = self.stage1.get_codes(images).detach()
        pos_enc_tokens = get_positional_encoding(texts, mode='1d')
        codes = codes.clone().detach()
        pos_enc_code = get_positional_encoding(codes, mode='1d')
        # codes = codes.unsqueeze(-1)
        # pos_enc_code = pos_enc_code.unsqueeze(-1)
        logits_img, logits_txt = self.stage2(codes, texts, pos_enc_code, pos_enc_tokens, past)
        return logits_img, logits_txt, codes

    def training_step(self, batch, batch_idx):
        images, texts = batch
        logits_img, logits_txt, codes = self(images, texts)

        loss_img = F.cross_entropy(logits_img.view(-1, logits_img.shape[-1]), codes.view(-1))
        loss_txt = F.cross_entropy(logits_txt.view(-1, logits_txt.shape[-1]), texts[:, 1:].reshape(-1))
        self.log("train/loss_img", loss_img, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train/loss_txt", loss_txt, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return loss_img + loss_txt

    def validation_step(self, batch, batch_idx):
        images, texts = batch
        logits_img, logits_txt, codes = self(images, texts)
        # print(logits_img.shape, logits_txt.shape, codes.shape, texts.shape)

        loss_img = F.cross_entropy(logits_img.view(-1, logits_img.shape[-1]), codes.view(-1))
        loss_txt = F.cross_entropy(logits_txt.view(-1, logits_txt.shape[-1]), texts[:, 1:].reshape(-1))
        self.log("val/loss_img", loss_img, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("val/loss_txt", loss_txt, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return loss_img + loss_txt

    def configure_optimizers(self):
        assert self.config.optimizer.opt_type == 'adamW'
        # assert self.config.optimizer.sched_type == 'cosine'

        opt = torch.optim.AdamW(self.parameters(),
                                lr=self.config.optimizer.learning_rate,
                                betas=self.config.optimizer.betas,
                                weight_decay=self.config.optimizer.weight_decay)
        # sched = CosineAnnealingLR(opt,
        #                           T_max=self.config.optimizer.max_steps,
        #                           eta_min=self.config.optimizer.min_lr)

        def lr_lambda(current_step: int):
            return max(
                0.0, float(self.config.optimizer.max_steps - current_step) / float(max(1, self.config.optimizer.max_steps))
            )

        sched = LambdaLR(opt, lr_lambda)
        sched = {
            'scheduler': sched,
            'name': 'linear'
        }
        return [opt], [sched]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure,
                       on_tpu=False, using_native_amp=False, using_lbfgs=False):
        optimizer.step(closure=optimizer_closure)
        self.lr_schedulers().step()
        self.log("lr", self.lr_schedulers().get_last_lr()[0], on_step=True, on_epoch=False, prog_bar=True, logger=True)

    def on_epoch_start(self):
        self.stage1.eval()


class ImageGPT(pl.LightningModule):
    def __init__(self,
                 config: OmegaConf) -> None:
        super().__init__()
        self.stage1 = VQGAN(n_embed=config.stage1.n_embed,
                            embed_dim=config.stage1.embed_dim,
                            hparams=config.stage1.hparams)
        self.stage2 = iGPT(vocab_size_img=config.stage2.vocab_size_img,
                           use_cls_cond=config.stage2.use_cls_cond,
                           hparams=config.stage2.hparams)
        self.config = config
        self.use_cls_cond = config.stage2.use_cls_cond

        # make the parameters in stage 1 not trainable
        self.stage1.eval()
        for p in self.stage1.parameters():
            p.requires_grad = False

    @classmethod
    def from_pretrained(cls,
                        path_upstream: str,
                        path_downstream: str) -> Tuple[nn.Module, OmegaConf]:
        config_base = get_base_config(use_default=False)
        config_down = OmegaConf.load(path_downstream)
        config_down = OmegaConf.merge(config_base, config_down)

        model = cls(config_down)
        model.stage1.from_ckpt(os.path.join(path_upstream, 'stage1_last.ckpt'), strict=True)
        model.stage2.from_ckpt(os.path.join(path_upstream, 'stage2_last.ckpt'), strict=False)
        return model, config_down

    def sample(self,
               cls_idx: Optional[int] = None,
               top_k: int = 256,
               top_p: Optional[float] = None,
               softmax_temperature: float = 1.0,
               num_candidates: int = 16,
               device: str = 'cuda:0',
               use_fp16: bool = True,
               is_tqdm: bool = True) -> torch.FloatTensor:
        self.stage1.eval()
        self.stage2.eval()

        if cls_idx is None:
            sos = self.stage2.sos.repeat(num_candidates, 1, 1)
        else:
            sos = torch.LongTensor([cls_idx]).to(device=device)
            sos = sos.repeat(num_candidates)
            sos = self.stage2.sos(sos).unsqueeze(1)

        codes = sampling_igpt(self.stage2,
                              sos=sos,
                              top_k=top_k,
                              top_p=top_p,
                              softmax_temperature=softmax_temperature,
                              use_fp16=use_fp16,
                              is_tqdm=is_tqdm)
        codes = codes.view(num_candidates, 16, 16)  # [B, 16, 16]
        pixels = torch.clamp(self.stage1.decode_code(codes) * 0.5 + 0.5, 0, 1)  # [B, 256, 256]
        return pixels

    def forward(self,
                images: torch.FloatTensor,
                labels: Optional[torch.LongTensor] = None) -> torch.FloatTensor:
        B, C, H, W = images.shape
        with torch.no_grad():
            with autocast(enabled=False):
                codes = self.stage1.get_codes(images).detach()
        logits = self.stage2(codes, labels)
        return logits, codes

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits, codes = self(images, labels=labels if self.use_cls_cond else None)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), codes.view(-1))
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits, codes = self(images, labels=labels if self.use_cls_cond else None)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), codes.view(-1))
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return loss

    def configure_optimizers(self):
        assert self.config.optimizer.opt_type == 'adamW'
        assert self.config.optimizer.sched_type == 'cosine'

        opt = torch.optim.AdamW(self.parameters(),
                                lr=self.config.optimizer.base_lr,
                                betas=self.config.optimizer.betas,
                                weight_decay=self.config.optimizer.weight_decay)
        sched = CosineAnnealingLR(opt,
                                  T_max=self.config.optimizer.max_steps,
                                  eta_min=self.config.optimizer.min_lr)
        sched = {
            'scheduler': sched,
            'name': 'cosine'
        }
        return [opt], [sched]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure,
                       on_tpu=False, using_native_amp=False, using_lbfgs=False):
        optimizer.step(closure=optimizer_closure)
        self.lr_schedulers().step()
        self.log("lr", self.lr_schedulers().get_last_lr()[0], on_step=True, on_epoch=False, prog_bar=True, logger=True)

    def on_epoch_start(self):
        self.stage1.eval()


class PromptDalle(Dalle):
    """Classification Head for  transformer encoders"""
    def __init__(self, config):
        super().__init__(config)
        print('Initializing the PromptTuning model')

        self.config = config
        self.n_embd = config.stage2.hparams.embed_dim
        self.preseqlen = config.prompt.preseqlen
        self.prefix_dropout = config.prompt.prefix_dropout

        # DIFFERENT PARAMETRIZATION:

        print('[Full prompt-tuning Setting :) ]')
        self.input_tokens = torch.arange(self.preseqlen).long()
        self.wte = nn.Embedding(self.preseqlen, self.n_embd)
        self.control_trans = nn.Sequential(
            nn.Linear(self.n_embd, self.n_embd),
            nn.Tanh(),
            nn.Linear(self.n_embd, self.n_embd))
        self.get_prompt = self.get_prompt_p5
        self.dropout = nn.Dropout(self.prefix_dropout)

        ###### NUM PARAMS #########
        total_param = 0
        for name, param in self.named_parameters():
            # print(param.shape)
            total_param += param.numel()
        print('Total parameters is {}'.format(total_param))


    @classmethod
    def from_pretrained(cls, args) -> Tuple[nn.Module, OmegaConf]:

        # if not args.model_name_or_path:
        #     args.model_name_or_path = args.prefix_model_name_or_path

        path = args.prefix_model_name_or_path
        path = _MODELS[path] if path in _MODELS else path
        path = utils.realpath_url_or_path(path, root=os.path.expanduser("~/.cache/minDALL-E"))

        config_base = get_base_config('prompt_tuning')
        config_new = OmegaConf.load(os.path.join(path, 'config.yaml'))
        config_update = OmegaConf.merge(config_base, config_new)

        for key, val in vars(args).items():
            if key in config_update.prompt.keys():
                OmegaConf.update(config_update, "prompt.%s" % key, val, merge=False)
            if key in config_update.optimizer.keys():
                OmegaConf.update(config_update, "optimizer.%s" % key, val, merge=False)
            if key in config_update.experiment.keys():
                OmegaConf.update(config_update, "experiment.%s" % key, val, merge=False)

        model = cls(config_update)
        model.tokenizer = build_tokenizer(os.path.join(path, 'tokenizer'),
                                          context_length=model.config_dataset.context_length,
                                          lowercase=True,
                                          dropout=None)

        if args.model_name_or_path:
            print("Loading model from pretrained checkpoint %s" % args.model_name_or_path)
            # model.from_ckpt(args.model_name_or_path)
            try:
                model.load_state_dict(torch.load(args.model_name_or_path)['state_dict'])
            except KeyError:
                model.load_state_dict(torch.load(args.model_name_or_path)['model_state_dict'])

        else:
            print("Loading models from checkpoint %s" % path)
            model.stage1.from_ckpt(os.path.join(path, 'stage1_last.ckpt'))
            model.stage2.from_ckpt(os.path.join(path, 'stage2_last.ckpt'))

        return model, config_update

    def get_prompt_p5(self, bsz=None, eval=False):
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1).to(self.device)
        temp_control = self.wte(input_tokens)
        past_key_values = self.control_trans(temp_control) #bsz, seqlen, layer*emb
        if not eval:
            past_key_values = self.dropout(past_key_values)
        return past_key_values

    def forward(self,
                images: torch.FloatTensor,
                texts: Optional[torch.LongTensor],
                **kwargs,
        ):

        #{"input_ids": batch, "labels": labels, 'src_attn': src_attn, 'tgt_attn':tgt_attn, 'src':src}

        B, C, H, W = images.shape
        prompt = self.get_prompt(bsz=B)
        pos_enc_prompt = get_positional_encoding(self.input_tokens.unsqueeze(0).expand(B, -1).to(self.device), mode='1d')

        # if self.mode_para == 2 and src_attn is not None and tgt_attn is not None:
        #     attention_mask = torch.cat([src_attn, tgt_attn], dim=1)


        with torch.no_grad():
            with autocast(enabled=False):
                codes = self.stage1.get_codes(images).detach()

        pos_enc_tokens = get_positional_encoding(texts, mode='1d')
        codes = codes.clone().detach()
        pos_enc_code = get_positional_encoding(codes, mode='1d')
        # codes = codes.unsqueeze(-1)
        # pos_enc_code = pos_enc_code.unsqueeze(-1)
        # print(images.shape, codes.shape, texts.shape)
        logits_img, logits_txt = self.stage2(codes, texts, pos_enc_code, pos_enc_tokens, prompt=prompt, pos_prompt=pos_enc_prompt)
        return logits_img, logits_txt, codes


    @torch.no_grad()
    def sampling(self,
                 tokens: torch.LongTensor,
                 prompt: torch.FloatTensor,
                 top_k: int = 256,
                 top_p: Optional[float] = None,
                 softmax_temperature: float = 1.0,
                 num_candidates: int = 96,
                 device: str = 'cuda:0',
                 use_fp16: bool = True,
                 labels = None) -> torch.FloatTensor:
        self.stage1.eval()
        self.stage2.eval()

        # tokens = torch.repeat_interleave(tokens.unsqueeze(0), num_candidates, dim=0)

        tokens = tokens.to(device)
        pos_enc_prompt = get_positional_encoding(self.input_tokens.unsqueeze(0).expand(num_candidates, -1).to(self.device), mode='1d')

        codes = sampling(self.stage2,
                         tokens,
                         top_k=top_k,
                         top_p=top_p,
                         softmax_temperature=softmax_temperature,
                         use_fp16=use_fp16,
                         prompt=prompt,
                         pos_prompt=pos_enc_prompt)

        codes = codes.view(-1, 16, 16)  # [B, 16, 16]
        pixels = torch.clamp(self.stage1.decode_code(codes) * 0.5 + 0.5, 0, 1)  # [B, 256, 256]
        return pixels


    @torch.no_grad()
    def predict_step(self, batch, batch_idx, return_images=False):
        orig_images, texts = batch

        # extra for checks
        logits_img, logits_txt, codes = self(orig_images, texts)
        pred = torch.argmax(logits_img.view(-1, logits_img.shape[-1]), dim=-1)
        bs = orig_images.shape[0]
        pred = pred.view(bs, 16, 16)  # [B, 16, 16]
        pixels = torch.clamp(self.stage1.decode_code(pred) * 0.5 + 0.5, 0, 1).cpu().numpy()  # [B, 256, 256]
        pixels = np.transpose(pixels, (0, 2, 3, 1))

        # print(texts.shape, orig_images.shape)
        prompt = self.get_prompt(bsz=5, eval=True)

        images = []
        for i, t in enumerate(texts):
            pixels = self.sampling(t, prompt, top_k=16, num_candidates=5, labels=codes[i]).cpu().numpy()
            pixels = np.transpose(pixels, (0, 2, 3, 1))
            images.append(pixels)

        if return_images:
            return images
        else:
            save_image(orig_images, pixels, './out/images/pororo_prompt', batch_idx+10)
            save_image(orig_images, images, './out/images/pororo_prompt', batch_idx)


class PrefixTuningDalle(Dalle):
    """Classification Head for  transformer encoders"""
    def __init__(self, config):
        super().__init__(config)
        print('Initializing the PrefixTuning model')

        self.config = config

        self.match_n_layer = config.stage2.hparams.n_layers
        self.match_n_head = config.stage2.hparams.n_heads
        self.match_n_embd = config.stage2.hparams.embed_dim // config.stage2.hparams.n_heads
        self.n_embd = config.stage2.hparams.embed_dim

        self.optim_prefix = config.prefix.optim_prefix
        self.preseqlen = config.prefix.preseqlen
        self.prefix_dropout = config.prefix.prefix_dropout
        self.init_random = config.prefix.init_random
        self.hidden_dim_prefix = config.prefix.hidden_dim_prefix

        self.lowdata_token = config.prefix.lowdata_token
        self.init_shallow = config.prefix.init_shallow
        self.init_shallow_word = config.prefix.init_shallow_word
        self.mode_para = 0

        print('PrefixTuning')
        print('preseqlen is {}, optimizing the prefix directly'.format(self.preseqlen))

        # DIFFERENT PARAMETRIZATION:

        print('[Full prefix-tuning Setting :) ]')
        self.input_tokens = torch.arange(self.preseqlen).long()
        self.wte = nn.Embedding(self.preseqlen, self.n_embd)
        self.control_trans = nn.Sequential(
            nn.Linear(self.n_embd, self.hidden_dim_prefix),
            nn.Tanh(),
            nn.Linear(self.hidden_dim_prefix, self.match_n_layer * 2 * self.n_embd))
        self.get_prompt = self.get_prompt_p5
        self.dropout = nn.Dropout(self.prefix_dropout)

        ###### NUM PARAMS #########
        total_param = 0
        for name, param in self.named_parameters():
            # print(param.shape)
            total_param += param.numel()
        print('Total parameters is {}'.format(total_param))


    @classmethod
    def from_pretrained(cls, args) -> Tuple[nn.Module, OmegaConf]:

        # if not args.model_name_or_path:
        #     args.model_name_or_path = args.prefix_model_name_or_path

        path = args.prefix_model_name_or_path
        path = _MODELS[path] if path in _MODELS else path
        path = utils.realpath_url_or_path(path, root=os.path.expanduser("~/.cache/minDALL-E"))

        config_base = get_base_config('prefixtuning')
        config_new = OmegaConf.load(os.path.join(path, 'config.yaml'))
        config_update = OmegaConf.merge(config_base, config_new)

        for key, val in vars(args).items():
            if key in config_update.prefix.keys():
                OmegaConf.update(config_update, "prefix.%s" % key, val, merge=False)
            if key in config_update.optimizer.keys():
                OmegaConf.update(config_update, "optimizer.%s" % key, val, merge=False)
            if key in config_update.experiment.keys():
                OmegaConf.update(config_update, "experiment.%s" % key, val, merge=False)

        model = cls(config_update)
        model.tokenizer = build_tokenizer(os.path.join(path, 'tokenizer'),
                                          context_length=model.config_dataset.context_length,
                                          lowercase=True,
                                          dropout=None)

        if args.model_name_or_path:
            print("Loading model from pretrained checkpoint %s" % args.model_name_or_path)
            # model.from_ckpt(args.model_name_or_path)
            try:
                model.load_state_dict(torch.load(args.model_name_or_path)['state_dict'])
            except KeyError:
                model.load_state_dict(torch.load(args.model_name_or_path)['model_state_dict'])

        else:
            print("Loading models from checkpoint %s" % path)
            model.stage1.from_ckpt(os.path.join(path, 'stage1_last.ckpt'))
            model.stage2.from_ckpt(os.path.join(path, 'stage2_last.ckpt'))

        return model, config_update

    def get_prompt_p5(self, bsz=None, eval=False):
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1).to(self.device)
        temp_control = self.wte(input_tokens)
        past_key_values = self.control_trans(temp_control) #bsz, seqlen, layer*emb
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                               self.match_n_embd)
        if not eval:
            past_key_values = self.dropout(past_key_values)
        # past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4])
        # print(past_key_values.shape)
        return past_key_values.split(2)

    def forward(self,
                images: torch.FloatTensor,
                texts: Optional[torch.LongTensor],
                **kwargs,
        ):

        #{"input_ids": batch, "labels": labels, 'src_attn': src_attn, 'tgt_attn':tgt_attn, 'src':src}

        B, C, H, W = images.shape

        if self.mode_para == 2:
            past_key_values_prompt = self.get_prompt(bsz=B)
        else:
            past_key_values_prompt = self.get_prompt(bsz=B)

        # if self.mode_para == 2 and src_attn is not None and tgt_attn is not None:
        #     attention_mask = torch.cat([src_attn, tgt_attn], dim=1)


        with torch.no_grad():
            with autocast(enabled=False):
                codes = self.stage1.get_codes(images).detach()

        pos_enc_tokens = get_positional_encoding(texts, mode='1d')
        codes = codes.clone().detach()
        pos_enc_code = get_positional_encoding(codes, mode='1d')
        # codes = codes.unsqueeze(-1)
        # pos_enc_code = pos_enc_code.unsqueeze(-1)
        # print(images.shape, codes.shape, texts.shape)
        logits_img, logits_txt = self.stage2(codes, texts, pos_enc_code, pos_enc_tokens, past_key_values_prompt)
        return logits_img, logits_txt, codes

    @torch.no_grad()
    def sampling(self,
                 tokens: torch.LongTensor,
                 past: torch.FloatTensor,
                 top_k: int = 256,
                 top_p: Optional[float] = None,
                 softmax_temperature: float = 1.0,
                 num_candidates: int = 96,
                 device: str = 'cuda:0',
                 use_fp16: bool = True,
                 labels = None) -> torch.FloatTensor:
        self.stage1.eval()
        self.stage2.eval()

        if len(past.shape) == 6:
            n_layers, temp, bs, n_heads, seq_len, n_dim = past.shape
            past = past.view(n_layers, temp, bs*n_heads, seq_len, n_dim)

        tokens = torch.repeat_interleave(tokens.unsqueeze(0), num_candidates, dim=0)

        # Check if the encoding works as intended
        # print(self.tokenizer.decode_batch(tokens.tolist(), skip_special_tokens=True)[0])

        tokens = tokens.to(device)
        codes = sampling_prefix(self.stage2,
                                tokens,
                                past,
                                top_k=top_k,
                                top_p=top_p,
                                softmax_temperature=softmax_temperature,
                                use_fp16=use_fp16,
                                labels = None if labels is None else labels.view(-1))

        # codes = sampling(self.stage2,
        #                  tokens,
        #                  top_k=top_k,
        #                  top_p=top_p,
        #                  softmax_temperature=softmax_temperature,
        #                  use_fp16=use_fp16)

        codes = codes.view(num_candidates, 16, 16)  # [B, 16, 16]
        pixels = torch.clamp(self.stage1.decode_code(codes) * 0.5 + 0.5, 0, 1)  # [B, 256, 256]
        return pixels

    def training_step(self, batch, batch_idx):
        images, texts = batch
        logits_img, logits_txt, codes = self(images, texts)

        loss_img = F.cross_entropy(logits_img.view(-1, logits_img.shape[-1]), codes.view(-1))
        loss_txt = F.cross_entropy(logits_txt.view(-1, logits_txt.shape[-1]), texts[:, 1:].reshape(-1))
        self.log("train/loss_img", loss_img, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train/loss_txt", loss_txt, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return loss_img + loss_txt

    def validation_step(self, batch, batch_idx):
        images, texts = batch
        logits_img, logits_txt, codes = self(images, texts)
        # print(logits_img.shape, logits_txt.shape, codes.shape, texts.shape)

        loss_img = F.cross_entropy(logits_img.view(-1, logits_img.shape[-1]), codes.view(-1))
        loss_txt = F.cross_entropy(logits_txt.view(-1, logits_txt.shape[-1]), texts[:, 1:].reshape(-1))
        self.log("val/loss_img", loss_img, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("val/loss_txt", loss_txt, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return loss_img + loss_txt

    @torch.no_grad()
    def predict_step(self, batch, batch_idx, return_images=False):
        orig_images, texts = batch

        # extra for checks
        logits_img, logits_txt, codes = self(orig_images, texts)
        pred = torch.argmax(logits_img.view(-1, logits_img.shape[-1]), dim=-1)
        bs = orig_images.shape[0]
        pred = pred.view(bs, 16, 16)  # [B, 16, 16]
        pixels = torch.clamp(self.stage1.decode_code(pred) * 0.5 + 0.5, 0, 1).cpu().numpy()  # [B, 256, 256]
        pixels = np.transpose(pixels, (0, 2, 3, 1))


        # print(texts.shape, orig_images.shape)
        # concatenate the list of prompts (split by n_head) for better downstream processing
        past_key_values_prompt = self.get_prompt(bsz=5, eval=True)
        # print(past_key_values_prompt[0].shape, past_key_values_prompt[1].shape, len(past_key_values_prompt))
        past_key_values_prompt = torch.cat([x.unsqueeze(0) for x in past_key_values_prompt], dim=0)
        n_layers, temp, bs, n_heads, seq_len, n_dim = past_key_values_prompt.shape
        past_key_values_prompt = past_key_values_prompt.view(n_layers, temp, bs*n_heads, seq_len, n_dim)
        # print(past_key_values_prompt.shape)
        images = []
        for i, t in enumerate(texts):
            pixels = self.sampling(t, past_key_values_prompt, top_k=16, num_candidates=5, labels=codes[i]).cpu().numpy()
            pixels = np.transpose(pixels, (0, 2, 3, 1))
            images.append(pixels)
            # images.extend([p for p in pixels])
            # print([i.shape for i in images])


        if return_images:
            return images
        else:
            save_image(orig_images, pixels, './out/images/pororo_prefix', batch_idx+10)
            save_image(orig_images, images, './out/images/pororo_prefix', batch_idx)


class ConditionalDalle(Dalle):
    """Classification Head for  transformer encoders"""
    def __init__(self, config):
        super().__init__(config)
        print('Initializing the Conditional Dalle model')

        self.config = config

        print('Setting up Cross-attention Layers')
        self.init_cross_attention(list(range(2,42,3)), config.stage2.hparams)

        ###### NUM PARAMS #########
        total_param = 0
        for name, param in self.named_parameters():
            # print(param.shape)
            total_param += param.numel()
        print('Total parameters is {}'.format(total_param))

    @classmethod
    def from_pretrained(cls, args) -> Tuple[nn.Module, OmegaConf]:

        # if not args.model_name_or_path:
        #     args.model_name_or_path = args.prefix_model_name_or_path

        path = args.model_name_or_path
        config_new = OmegaConf.load(os.path.join(path, 'config.yaml'))
        if args.do_train:
            config_base = get_base_config('finetuning')
            config_update = OmegaConf.merge(config_base, config_new)
            for key, val in vars(args).items():
                if key in config_update.optimizer.keys():
                    OmegaConf.update(config_update, "optimizer.%s" % key, val, merge=False)
                if key in config_update.experiment.keys():
                    OmegaConf.update(config_update, "experiment.%s" % key, val, merge=False)
        else:
            config_base = get_base_config('default')
            config_update = OmegaConf.merge(config_base, config_new)

        model = cls(config_update)
        model.tokenizer = build_tokenizer(os.path.join(path, 'tokenizer'),
                                          context_length=model.config_dataset.context_length,
                                          lowercase=True,
                                          dropout=None)
        print(model.cross_attention_idxs)
        # print(next(model.cross_attention_layers[0].parameters()).is_cuda)

        if args.dalle_path:
            print("Loading model from pretrained checkpoint %s" % args.dalle_path)
            # model.from_ckpt(args.model_name_or_path)
            model.load_state_dict(torch.load(args.dalle_path)['model_state_dict'])
        else:
            print("Loading models from checkpoint %s" % path)
            model.stage1.from_ckpt(os.path.join(path, 'stage1_last.ckpt'))
            model.stage2.from_ckpt(os.path.join(path, 'stage2_last.ckpt'))

        return model, config_update


    def init_cross_attention(self, cross_attention_layers, hparams):
        self.cross_attention_idxs = cross_attention_layers
        self.cross_attention_layers = [CrossAttentionLayer(ctx_len=hparams.ctx_len_img + hparams.ctx_len_txt,
                                         embed_dim=hparams.embed_dim,
                                         n_heads=hparams.n_heads,
                                         attn_bias=hparams.attn_bias,
                                         resid_pdrop=hparams.resid_pdrop,
                                         attn_pdrop=hparams.attn_pdrop) for i in cross_attention_layers]


    def forward(self,
                images: torch.FloatTensor,
                src_images: Optional[torch.FloatTensor],
                texts: Optional[torch.LongTensor],
                **kwargs,
        ):

        #{"input_ids": batch, "labels": labels, 'src_attn': src_attn, 'tgt_attn':tgt_attn, 'src':src}

        # print(images.shape, src_images.shape, texts.shape)
        with torch.no_grad():
            with autocast(enabled=False):
                codes = self.stage1.get_codes(images).detach()
                src_codes = self.stage1.get_codes(src_images).detach()

        pos_enc_tokens = get_positional_encoding(texts, mode='1d')
        codes = codes.clone().detach()
        pos_enc_code = get_positional_encoding(codes, mode='1d')
        src_codes = src_codes.clone().detach()
        src_pos_enc_code = get_positional_encoding(src_codes, mode='1d')
        # codes = codes.unsqueeze(-1)
        # pos_enc_code = pos_enc_code.unsqueeze(-1)
        # print(images.shape, codes.shape, texts.shape)
        logits_img, logits_txt = self.stage2.forward_with_context(codes, texts,
                                                                  pos_enc_code, pos_enc_tokens, src_codes, src_pos_enc_code,
                                                                  self.cross_attention_idxs, self.cross_attention_layers)
        # print(logits_img.shape, logits_txt.shape, codes.shape, texts.shape)
        return logits_img, logits_txt, codes

    @torch.no_grad()
    def sampling(self,
                 prompt: torch.LongTensor,
                 source: torch.FloatTensor,
                 top_k: int = 256,
                 top_p: Optional[float] = None,
                 softmax_temperature: float = 1.0,
                 num_candidates: int = 96,
                 device: str = 'cuda:0',
                 use_fp16: bool = True) -> torch.FloatTensor:
        self.stage1.eval()
        self.stage2.eval()

        if type(prompt) == str:
            tokens = self.tokenizer.encode(prompt)
            tokens = torch.LongTensor(tokens.ids)
        else:
            tokens = prompt

        tokens = torch.repeat_interleave(tokens.unsqueeze(0), num_candidates, dim=0)

        # Check if the encoding works as intended
        # print(self.tokenizer.decode_batch(tokens.tolist(), skip_special_tokens=True)[0])

        tokens = tokens.to(device)
        source = source.to(device)

        with autocast(enabled=False):
            src_codes = self.stage1.get_codes(source).detach()
        src_codes = torch.repeat_interleave(src_codes, num_candidates, dim=0)

        codes = sampling_conditional(self.stage2,
                                     self.cross_attention_idxs,
                                     self.cross_attention_layers,
                                     tokens,
                                     src_codes,
                                     top_k=top_k,
                                     top_p=top_p,
                                     softmax_temperature=softmax_temperature,
                                     use_fp16=use_fp16)
        codes = codes.view(num_candidates, 16, 16)  # [B, 16, 16]
        pixels = torch.clamp(self.stage1.decode_code(codes) * 0.5 + 0.5, 0, 1)  # [B, 256, 256]
        return pixels

    def training_step(self, batch, batch_idx):
        images, texts = batch
        logits_img, logits_txt, codes = self(images, texts)

        loss_img = F.cross_entropy(logits_img.view(-1, logits_img.shape[-1]), codes.view(-1))
        loss_txt = F.cross_entropy(logits_txt.view(-1, logits_txt.shape[-1]), texts[:, 1:].reshape(-1))
        self.log("train/loss_img", loss_img, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train/loss_txt", loss_txt, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return loss_img + loss_txt

    def validation_step(self, batch, batch_idx):
        images, texts = batch
        logits_img, logits_txt, codes = self(images, texts)
        # print(logits_img.shape, logits_txt.shape, codes.shape, texts.shape)

        loss_img = F.cross_entropy(logits_img.view(-1, logits_img.shape[-1]), codes.view(-1))
        loss_txt = F.cross_entropy(logits_txt.view(-1, logits_txt.shape[-1]), texts[:, 1:].reshape(-1))
        self.log("val/loss_img", loss_img, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("val/loss_txt", loss_txt, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return loss_img + loss_txt

    @torch.no_grad()
    def predict_step(self, batch, batch_idx):
        orig_images, texts = batch
        # concatenate the list of prompts (split by n_head) for better downstream processing
        past_key_values_prompt = self.get_prompt(bsz=5)
        past_key_values_prompt = torch.cat([x.unsqueeze(0) for x in past_key_values_prompt], dim=0)
        images = []
        for t in texts:
            pixels = self.sampling(t, past_key_values_prompt, top_k=64, num_candidates=5).cpu().numpy()
            pixels = np.transpose(pixels, (0, 2, 3, 1))
            images.append(pixels)
            # images.extend([p for p in pixels])
            # print([i.shape for i in images])

        save_image(orig_images, images, './out/images/', batch_idx)


class PromptConditionalDalle(Dalle):
    """Classification Head for  transformer encoders"""
    def __init__(self, config):
        super().__init__(config)
        print('Initializing the Conditional Dalle model')

        self.config = config

        print('Setting up Cross-attention Layers')
        self.init_cross_attention(list(range(2,42,3)), config.stage2.hparams)

        self.n_embd = config.stage2.hparams.embed_dim
        self.preseqlen = config.story.preseqlen
        self.prefix_dropout = config.story.prefix_dropout

        # DIFFERENT PARAMETRIZATION:

        print('[Full prompt-tuning Setting :) ]')
        self.input_tokens = torch.arange(self.preseqlen).long()
        self.wte = nn.Embedding(self.preseqlen, self.n_embd)
        self.control_trans = nn.Sequential(
            nn.Linear(self.n_embd, self.n_embd),
            nn.Tanh(),
            nn.Linear(self.n_embd, self.n_embd))
        self.get_prompt = self.get_prompt_p5
        self.dropout = nn.Dropout(self.prefix_dropout)

        ###### NUM PARAMS #########
        total_param = 0
        for name, param in self.named_parameters():
            # print(param.shape)
            total_param += param.numel()
        print('Total parameters is {}'.format(total_param))

    @classmethod
    def from_pretrained(cls, args) -> Tuple[nn.Module, OmegaConf]:

        # if not args.model_name_or_path:
        #     args.model_name_or_path = args.prefix_model_name_or_path

        path = args.prefix_model_name_or_path
        path = _MODELS[path] if path in _MODELS else path
        path = utils.realpath_url_or_path(path, root=os.path.expanduser("~/.cache/minDALL-E"))

        config_new = OmegaConf.load(os.path.join(path, 'config.yaml'))
        if args.do_train:
            config_base = get_base_config('story')
            config_update = OmegaConf.merge(config_base, config_new)
            for key, val in vars(args).items():
                if key in config_update.story.keys():
                    OmegaConf.update(config_update, "story.%s" % key, val, merge=False)
                if key in config_update.optimizer.keys():
                    OmegaConf.update(config_update, "optimizer.%s" % key, val, merge=False)
                if key in config_update.experiment.keys():
                    OmegaConf.update(config_update, "experiment.%s" % key, val, merge=False)
        else:
            config_base = get_base_config('default')
            config_update = OmegaConf.merge(config_base, config_new)

        model = cls(config_update)
        model.tokenizer = build_tokenizer(os.path.join(path, 'tokenizer'),
                                          context_length=model.config_dataset.context_length,
                                          lowercase=True,
                                          dropout=None)
        print(model.cross_attention_idxs)
        # print(next(model.cross_attention_layers[0].parameters()).is_cuda)

        if args.model_name_or_path:
            print("Loading model from pretrained checkpoint %s" % args.model_name_or_path)
            # model.from_ckpt(args.model_name_or_path)
            try:
                model.load_state_dict(torch.load(args.model_name_or_path)['state_dict'])
            except KeyError:
                model.load_state_dict(torch.load(args.model_name_or_path)['model_state_dict'])

        else:
            print("Loading models from checkpoint %s" % path)
            model.stage1.from_ckpt(os.path.join(path, 'stage1_last.ckpt'))
            model.stage2.from_ckpt(os.path.join(path, 'stage2_last.ckpt'))

        return model, config_update


    def init_cross_attention(self, cross_attention_layers, hparams):
        self.cross_attention_idxs = cross_attention_layers
        self.cross_attention_layers = [CrossAttentionLayer(ctx_len=hparams.ctx_len_img + hparams.ctx_len_txt,
                                         embed_dim=hparams.embed_dim,
                                         n_heads=hparams.n_heads,
                                         attn_bias=hparams.attn_bias,
                                         resid_pdrop=hparams.resid_pdrop,
                                         attn_pdrop=hparams.attn_pdrop) for i in cross_attention_layers]

    def get_prompt_p5(self, bsz=None, eval=False):
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1).to(self.device)
        temp_control = self.wte(input_tokens)
        past_key_values = self.control_trans(temp_control) #bsz, seqlen, layer*emb
        if not eval:
            past_key_values = self.dropout(past_key_values)
        return past_key_values

    def forward(self,
                images: torch.FloatTensor,
                src_images: Optional[torch.FloatTensor],
                texts: Optional[torch.LongTensor],
                **kwargs,
        ):

        #{"input_ids": batch, "labels": labels, 'src_attn': src_attn, 'tgt_attn':tgt_attn, 'src':src}

        # print(images.shape, src_images.shape, texts.shape)
        with torch.no_grad():
            with autocast(enabled=False):
                codes = self.stage1.get_codes(images).detach()
                src_codes = self.stage1.get_codes(src_images).detach()

        B, C, H, W = images.shape
        prompt = self.get_prompt(bsz=B)
        pos_enc_prompt = get_positional_encoding(self.input_tokens.unsqueeze(0).expand(B, -1).to(self.device), mode='1d')

        pos_enc_tokens = get_positional_encoding(texts, mode='1d')
        codes = codes.clone().detach()
        pos_enc_code = get_positional_encoding(codes, mode='1d')
        src_codes = src_codes.clone().detach()
        src_pos_enc_code = get_positional_encoding(src_codes, mode='1d')
        # codes = codes.unsqueeze(-1)
        # pos_enc_code = pos_enc_code.unsqueeze(-1)
        # print(images.shape, codes.shape, texts.shape)
        logits_img, logits_txt = self.stage2.forward_with_context(codes, texts,
                                                                  pos_enc_code, pos_enc_tokens, src_codes, src_pos_enc_code,
                                                                  self.cross_attention_idxs, self.cross_attention_layers,
                                                                  prompt=prompt, pos_prompt=pos_enc_prompt)
        # print(logits_img.shape, logits_txt.shape, codes.shape, texts.shape)
        return logits_img, logits_txt, codes

    @torch.no_grad()
    def sampling(self,
                 tokens: torch.LongTensor,
                 prompt: torch.LongTensor,
                 source: torch.FloatTensor,
                 top_k: int = 256,
                 top_p: Optional[float] = None,
                 softmax_temperature: float = 1.0,
                 num_candidates: int = 96,
                 device: str = 'cuda:0',
                 use_fp16: bool = True,
                 labels=None) -> torch.FloatTensor:

        self.stage1.eval()
        self.stage2.eval()

        if type(tokens) == str:
            tokens = self.tokenizer.encode(prompt)
            tokens = torch.LongTensor(tokens.ids)
        else:
            pass

        tokens = torch.repeat_interleave(tokens.unsqueeze(0), num_candidates, dim=0)

        # Check if the encoding works as intended
        # print(self.tokenizer.decode_batch(tokens.tolist(), skip_special_tokens=True)[0])

        tokens = tokens.to(device)
        source = source.to(device)

        pos_enc_prompt = get_positional_encoding(self.input_tokens.unsqueeze(0).expand(num_candidates, -1).to(self.device), mode='1d')

        with autocast(enabled=False):
            src_codes = self.stage1.get_codes(source).detach()
        src_codes = torch.repeat_interleave(src_codes, num_candidates, dim=0)

        codes = sampling_conditional(self.stage2,
                                     self.cross_attention_idxs,
                                     self.cross_attention_layers,
                                     tokens,
                                     src_codes,
                                     top_k=top_k,
                                     top_p=top_p,
                                     softmax_temperature=softmax_temperature,
                                     use_fp16=use_fp16,
                                     prompt=prompt,
                                     pos_prompt=pos_enc_prompt)

        codes = codes.view(num_candidates, 16, 16)  # [B, 16, 16]
        pixels = torch.clamp(self.stage1.decode_code(codes) * 0.5 + 0.5, 0, 1)  # [B, 256, 256]
        return pixels


    @torch.no_grad()
    def predict_step(self, batch, batch_idx, return_images=False):
        orig_images, texts = batch
        # concatenate the list of prompts (split by n_head) for better downstream processing

        # extra for checks
        logits_img, logits_txt, codes = self(orig_images, texts)
        pred = torch.argmax(logits_img.view(-1, logits_img.shape[-1]), dim=-1)
        bs = orig_images.shape[0]
        pred = pred.view(bs, 16, 16)  # [B, 16, 16]
        pixels = torch.clamp(self.stage1.decode_code(pred) * 0.5 + 0.5, 0, 1).cpu().numpy()  # [B, 256, 256]
        pixels = np.transpose(pixels, (0, 2, 3, 1))

        prompt = self.get_prompt(bsz=5, eval=True)

        images = []
        for t in texts:
            pixels = self.sampling(t, prompt, top_k=64, num_candidates=5, labels=codes[i]).cpu().numpy()
            pixels = np.transpose(pixels, (0, 2, 3, 1))
            images.append(pixels)
            # images.extend([p for p in pixels])
            # print([i.shape for i in images])

        if return_images:
            return images
        else:
            save_image(orig_images, pixels, './out/images/pororo_story', batch_idx+10)
            save_image(orig_images, images, './out/images/pororo_story', batch_idx)


class StoryDalle(Dalle):
    """Base model with story block"""
    def __init__(self, config):
        super().__init__(config)
        print('Initializing the Conditional Dalle model')

        self.config = config

        self.story_linear = nn.Linear(config.story.sent_embed, config.stage2.hparams.embed_dim)
        self.story_block = Block(ctx_len=config.story.story_len,
                             embed_dim=config.stage2.hparams.embed_dim,
                             n_heads=config.stage2.hparams.n_heads,
                             mlp_bias=config.stage2.hparams.mlp_bias,
                             attn_bias=config.stage2.hparams.attn_bias,
                             resid_pdrop=config.stage2.hparams.resid_pdrop,
                             attn_pdrop=config.stage2.hparams.attn_pdrop,
                             gelu_use_approx=config.stage2.hparams.gelu_use_approx)

        if self.config.story.prompt:
            self.n_embd = config.stage2.hparams.embed_dim
            self.preseqlen = config.story.preseqlen
            self.prefix_dropout = config.story.prefix_dropout

            # DIFFERENT PARAMETRIZATION:

            print('[Full prompt-tuning Setting :) ]')
            self.input_tokens = torch.arange(self.preseqlen).long()
            self.wte = nn.Embedding(self.preseqlen, self.n_embd)
            self.control_trans = nn.Sequential(
                nn.Linear(self.n_embd, self.n_embd),
                nn.Tanh(),
                nn.Linear(self.n_embd, self.n_embd))
            self.get_prompt = self.get_prompt_p5
            self.dropout = nn.Dropout(self.prefix_dropout)

        if self.config.story.condition:
            print('Setting up Cross-attention Layers')
            self.init_cross_attention(list(range(2,42,3)), config.stage2.hparams)

        ###### NUM PARAMS #########
        total_param = 0
        for name, param in self.named_parameters():
            # print(param.shape)
            total_param += param.numel()
        print('Total parameters is {}'.format(total_param))

    @classmethod
    def from_pretrained(cls, args) -> Tuple[nn.Module, OmegaConf]:

        # if not args.model_name_or_path:
        #     args.model_name_or_path = args.prefix_model_name_or_path

        path = args.prefix_model_name_or_path
        path = _MODELS[path] if path in _MODELS else path
        path = utils.realpath_url_or_path(path, root=os.path.expanduser("~/.cache/minDALL-E"))

        config_new = OmegaConf.load(os.path.join(path, 'config.yaml'))
        # if args.do_train:
        config_base = get_base_config('story')
        config_update = OmegaConf.merge(config_base, config_new)
        for key, val in vars(args).items():
            if key in config_update.story.keys():
                OmegaConf.update(config_update, "story.%s" % key, val, merge=False)
            if key in config_update.optimizer.keys():
                OmegaConf.update(config_update, "optimizer.%s" % key, val, merge=False)
            if key in config_update.experiment.keys():
                OmegaConf.update(config_update, "experiment.%s" % key, val, merge=False)
        # else:
        #     config_base = get_base_config('story')
        #     config_update = OmegaConf.merge(config_base, config_new)
        # print(next(model.cross_attention_layers[0].parameters()).is_cuda)

        if args.model_name_or_path:
            if 'pororo' in args.model_name_or_path:
                config_update.stage2.vocab_size_txt = config_update.stage2.vocab_size_txt + 9
            elif 'flintstones' in args.model_name_or_path:
                config_update.stage2.vocab_size_txt = config_update.stage2.vocab_size_txt + 7
            model = cls(config_update)
            model_dir = os.path.dirname(args.model_name_or_path)
            print(model_dir)
            model.tokenizer = build_tokenizer(model_dir,
                                              context_length=model.config_dataset.context_length,
                                              lowercase=True,
                                              dropout=None)
            print("Loaded tokenizer from finetuned checkpoint")
            print(model.cross_attention_idxs)
            print("Loading model from pretrained checkpoint %s" % args.model_name_or_path)
            # model.from_ckpt(args.model_name_or_path)
            try:
                model.load_state_dict(torch.load(args.model_name_or_path)['state_dict'])
            except KeyError:
                model.load_state_dict(torch.load(args.model_name_or_path)['model_state_dict'])
        else:
            model = cls(config_update)
            print(model.cross_attention_idxs)
            print("Loading models from checkpoint %s" % path)
            model.stage1.from_ckpt(os.path.join(path, 'stage1_last.ckpt'))
            model.stage2.from_ckpt(os.path.join(path, 'stage2_last.ckpt'))

            model.tokenizer = build_tokenizer(os.path.join(path, 'tokenizer'),
                                              context_length=model.config_dataset.context_length,
                                              lowercase=True,
                                              dropout=None)


        return model, config_update


    def init_cross_attention(self, cross_attention_layers, hparams):
        self.cross_attention_idxs = cross_attention_layers
        self.cross_attention_layers = [CrossAttentionLayer(ctx_len=hparams.ctx_len_img + hparams.ctx_len_txt,
                                         embed_dim=hparams.embed_dim,
                                         n_heads=hparams.n_heads,
                                         attn_bias=hparams.attn_bias,
                                         resid_pdrop=hparams.resid_pdrop,
                                         attn_pdrop=hparams.attn_pdrop) for i in cross_attention_layers]

    def get_prompt_p5(self, bsz=None, eval=False):
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1).to(self.device)
        temp_control = self.wte(input_tokens)
        past_key_values = self.control_trans(temp_control) #bsz, seqlen, layer*emb
        if not eval:
            past_key_values = self.dropout(past_key_values)
        return past_key_values

    def forward(self,
                images: torch.FloatTensor,
                src_images: Optional[torch.FloatTensor],
                texts: Optional[torch.LongTensor],
                sent_embeds: Optional[torch.FloatTensor],
                **kwargs,
        ):

        # print(images.shape, src_images.shape, texts.shape, sent_embeds.shape)

        B, L, C, H, W = images.shape
        images = images.view(B*L, C, H, W)
        src_images = src_images.unsqueeze(1).expand(-1, L, -1, -1, -1).reshape(B*L, C, H, W)
        sent_embeds = self.story_block(self.story_linear(sent_embeds)).view(B * L, -1).unsqueeze(1)
        texts = texts.view(B * L, -1)

        #{"input_ids": batch, "labels": labels, 'src_attn': src_attn, 'tgt_attn':tgt_attn, 'src':src}

        with torch.no_grad():
            with autocast(enabled=False):
                codes = self.stage1.get_codes(images).detach()
                src_codes = self.stage1.get_codes(src_images).detach()

        B, C, H, W = images.shape

        if self.config.story.prompt:
            prompt = self.get_prompt(bsz=B)
            prompt = torch.cat([prompt, sent_embeds], dim=1)
        else:
            prompt = sent_embeds

        # dim = 0 for full-model finetuning??
        pos_enc_prompt = get_positional_encoding(torch.arange(prompt.shape[1]).long().unsqueeze(0).expand(B, -1).to(self.device),
                                                 mode='1d')

        pos_enc_tokens = get_positional_encoding(texts, mode='1d')
        codes = codes.clone().detach()
        pos_enc_code = get_positional_encoding(codes, mode='1d')
        src_codes = src_codes.clone().detach()
        src_pos_enc_code = get_positional_encoding(src_codes, mode='1d')
        # codes = codes.unsqueeze(-1)
        # pos_enc_code = pos_enc_code.unsqueeze(-1)
        # print(images.shape, codes.shape, texts.shape)
        if self.config.story.condition:
            logits_img, logits_txt = self.stage2.forward_with_context(codes, texts,
                                                                      pos_enc_code, pos_enc_tokens, src_codes, src_pos_enc_code,
                                                                      self.cross_attention_idxs, self.cross_attention_layers,
                                                                      prompt=prompt, pos_prompt=pos_enc_prompt)
        else:
            logits_img, logits_txt = self.stage2(codes, texts, pos_enc_code, pos_enc_tokens, prompt=prompt,
                                                 pos_prompt=pos_enc_prompt)

        # print(logits_img.shape, logits_txt.shape, codes.shape, texts.shape)
        return logits_img, logits_txt, codes

    @torch.no_grad()
    def sampling(self,
                 tokens: torch.LongTensor,
                 source: torch.FloatTensor,
                 sent_embeds: torch.FloatTensor,
                 top_k: int = 256,
                 top_p: Optional[float] = None,
                 softmax_temperature: float = 1.0,
                 num_candidates: int = 96,
                 device: str = 'cuda:0',
                 use_fp16: bool = True,
                 labels=None,
                 prompt = None) -> torch.FloatTensor:

        self.stage1.eval()
        self.stage2.eval()

        if type(tokens) == str:
            tokens = self.tokenizer.encode(tokens)
            tokens = torch.LongTensor(tokens.ids)

        # tokens = torch.repeat_interleave(tokens.unsqueeze(0), num_candidates, dim=0)

        # Check if the encoding works as intended
        # print(self.tokenizer.decode_batch(tokens.tolist(), skip_special_tokens=True)[0])

        tokens = tokens.to(device)
        source = source.to(device)

        # print(tokens.shape, sent_embeds.shape, prompt.shape)
        B, L, _ = sent_embeds.shape
        sent_embeds = self.story_block(self.story_linear(sent_embeds)).view(B * L, -1).unsqueeze(1)
        if prompt is not None:
            prompt = torch.cat([prompt, sent_embeds], dim=1)
        else:
            prompt = sent_embeds
        pos_enc_prompt = get_positional_encoding(torch.arange(prompt.shape[1]).long().unsqueeze(0).expand(B*L, -1).to(self.device), mode='1d')

        with autocast(enabled=False):
            src_codes = self.stage1.get_codes(source).detach()
        src_codes = torch.repeat_interleave(src_codes, self.config.story.story_len, dim=0)
        print(tokens.shape, src_codes.shape, prompt.shape)
        if self.config.story.condition:
            codes = sampling_conditional(self.stage2,
                                         self.cross_attention_idxs,
                                         self.cross_attention_layers,
                                         tokens,
                                         src_codes,
                                         top_k=top_k,
                                         top_p=top_p,
                                         softmax_temperature=softmax_temperature,
                                         use_fp16=use_fp16,
                                         prompt=prompt,
                                         pos_prompt=pos_enc_prompt)
        else:
            codes = sampling(self.stage2,
                             tokens,
                             top_k=top_k,
                             top_p=top_p,
                             softmax_temperature=softmax_temperature,
                             use_fp16=use_fp16,
                             prompt=prompt,
                             pos_prompt=pos_enc_prompt)

        codes = codes.view(self.config.story.story_len, 16, 16)  # [B, 16, 16]
        pixels = torch.clamp(self.stage1.decode_code(codes) * 0.5 + 0.5, 0, 1)  # [B, 256, 256]
        return pixels

    @torch.no_grad()
    def sampling_batch(self,
                 tokens: torch.LongTensor,
                 source: torch.FloatTensor,
                 sent_embeds: torch.FloatTensor,
                 top_k: int = 256,
                 top_p: Optional[float] = None,
                 softmax_temperature: float = 1.0,
                 num_candidates: int = 96,
                 device: str = 'cuda:0',
                 use_fp16: bool = True,
                 labels=None,
                 prompt=None, n_candidates=1) -> torch.FloatTensor:

        self.stage1.eval()
        self.stage2.eval()

        if type(tokens) == str:
            tokens = self.tokenizer.encode(tokens)
            tokens = torch.LongTensor(tokens.ids)

        # tokens = torch.repeat_interleave(tokens.unsqueeze(0), num_candidates, dim=0)

        # Check if the encoding works as intended
        # print(self.tokenizer.decode_batch(tokens.tolist(), skip_special_tokens=True)[0])

        tokens = tokens.to(device)
        source = source.to(device)

        # print(tokens.shape, sent_embeds.shape, prompt.shape)
        B, L, _ = sent_embeds.shape
        sent_embeds = self.story_block(self.story_linear(sent_embeds)).view(B * L, -1).unsqueeze(1)
        if prompt is not None:
            prompt = torch.cat([prompt, sent_embeds], dim=1)
        else:
            prompt = sent_embeds
        pos_enc_prompt = get_positional_encoding(
            torch.arange(prompt.shape[1]).long().unsqueeze(0).expand(B * L, -1).to(self.device), mode='1d')

        with autocast(enabled=False):
            src_codes = self.stage1.get_codes(source).detach()

        # repeat inputs to adjust to n_candidates and story length
        src_codes = torch.repeat_interleave(src_codes, self.config.story.story_len * n_candidates, dim=0)
        prompt = prompt.repeat(n_candidates, 1, 1)
        pos_enc_prompt = pos_enc_prompt.repeat(n_candidates, 1)
        tokens = tokens.repeat(n_candidates, 1)
        print(tokens.shape, src_codes.shape, prompt.shape, pos_enc_prompt.shape)
        if self.config.story.condition:
            codes = sampling_conditional(self.stage2,
                                         self.cross_attention_idxs,
                                         self.cross_attention_layers,
                                         tokens,
                                         src_codes,
                                         top_k=top_k,
                                         top_p=top_p,
                                         softmax_temperature=softmax_temperature,
                                         use_fp16=use_fp16,
                                         prompt=prompt,
                                         pos_prompt=pos_enc_prompt)
        else:
            codes = sampling(self.stage2,
                             tokens,
                             top_k=top_k,
                             top_p=top_p,
                             softmax_temperature=softmax_temperature,
                             use_fp16=use_fp16,
                             prompt=prompt,
                             pos_prompt=pos_enc_prompt)

        codes = codes.view(self.config.story.story_len * n_candidates, 16, 16)  # [B, 16, 16]
        print(codes.shape)
        pixels = torch.clamp(self.stage1.decode_code(codes) * 0.5 + 0.5, 0, 1)  # [B, 3, 256, 256]
        print(pixels.shape)
        return pixels.view(n_candidates, self.config.story.story_len, pixels.shape[-3], pixels.shape[-2], pixels.shape[-1])


    @torch.no_grad()
    def predict_step(self, batch, batch_idx, return_images=False):
        orig_images, texts = batch
        # concatenate the list of prompts (split by n_head) for better downstream processing

        # extra for checks
        logits_img, logits_txt, codes = self(orig_images, texts)
        pred = torch.argmax(logits_img.view(-1, logits_img.shape[-1]), dim=-1)
        bs = orig_images.shape[0]
        pred = pred.view(bs, 16, 16)  # [B, 16, 16]
        pixels = torch.clamp(self.stage1.decode_code(pred) * 0.5 + 0.5, 0, 1).cpu().numpy()  # [B, 256, 256]
        pixels = np.transpose(pixels, (0, 2, 3, 1))

        prompt = self.get_prompt(bsz=5, eval=True)

        images = []
        for t in texts:
            pixels = self.sampling(t, prompt, top_k=64, num_candidates=5, labels=codes[i]).cpu().numpy()
            pixels = np.transpose(pixels, (0, 2, 3, 1))
            images.append(pixels)
            # images.extend([p for p in pixels])
            # print([i.shape for i in images])

        if return_images:
            return images
        else:
            save_image(orig_images, pixels, './out/images/pororo_story', batch_idx+10)
            save_image(orig_images, images, './out/images/pororo_story', batch_idx)