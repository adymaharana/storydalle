# ------------------------------------------------------------------------------------
# Minimal DALL-E
# Copyright (c) 2021 KakaoBrain. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------

from typing import Optional, List
from dataclasses import dataclass, field
from omegaconf import OmegaConf


@dataclass
class DataConfig:
    dataset: Optional[str] = None
    tokenizer_type: str = 'CharBPE'
    context_length: int = 64
    image_resolution: int = 256
    transforms: str = 'dalle-vqvae'
    bpe_pdrop: Optional[float] = None


@dataclass
class Stage1Hparams:
    double_z: bool = False
    z_channels: int = 256
    resolution: int = 256
    in_channels: int = 3
    out_ch: int = 3
    ch: int = 128
    ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    num_res_blocks: int = 2
    attn_resolutions: List[int] = field(default_factory=lambda: [16])
    pdrop: float = 0.0


@dataclass
class Stage2Hparams:
    embed_dim: int = 1536
    n_layers: int = 42
    n_heads: int = 24
    n_dense_layers: int = 42
    ctx_len_img: int = 256
    ctx_len_txt: int = 64
    embd_pdrop: float = 0.0
    resid_pdrop: float = 0.0
    attn_pdrop: float = 0.0
    mlp_bias: bool = True
    attn_bias: bool = True
    gelu_use_approx: bool = False
    use_head_txt: bool = True
    n_classes: Optional[int] = None


@dataclass
class Stage1Config:
    type: str = 'vqgan'
    embed_dim: int = 256
    n_embed: int = 16384
    hparams: Stage1Hparams = Stage1Hparams()


@dataclass
class Stage2Config:
    type: str = 'transformer1d'
    vocab_size_txt: int = 16384
    vocab_size_img: int = 16384
    use_cls_cond: Optional[bool] = None
    hparams: Stage2Hparams = Stage2Hparams()


@dataclass
class WarmupConfig:
    epoch: int = 1
    multiplier: int = 1
    buffer_epoch: int = 0
    min_lr: float = 0.0
    mode: str = 'fix'
    peak_lr: float = 1e-4
    start_from_zero: bool = True


@dataclass
class OptConfig:
    opt_type: str = 'adamW'
    learning_rate: float = 5e-5
    weight_decay: float = 1e-4
    betas: List[float] = field(default_factory=lambda: [0.9, 0.99])
    grad_clip_norm: float = 1.0

    sched_type: str = 'cosine'
    max_steps: int = 0
    min_lr: float = 1e-6


@dataclass
class ExpConfig:
    per_gpu_train_batch_size: int = 4
    per_gpu_eval_batch_size: int = 32
    num_train_epochs: int = 10
    save_ckpt_freq: int = 1
    test_freq: int = 10
    use_amp: bool = True


@dataclass
class PrefixModelConfig:
    model_name_or_path: Optional[str] = ''
    prefix_model_name_or_path: str = ''
    prefix_mode: str = 'activation'
    tuning_mode: str = 'finetune'
    top_k_layers: int = 2
    parameterize_mode: str = 'mlp'
    optim_prefix: bool = False
    preseqlen: int = 10
    prefix_dropout: float = 0.1
    init_random: bool = False
    hidden_dim_prefix: int = 512
    lowdata: bool = False
    lowdata_token: str = ''
    init_shallow: bool = False
    init_shallow_word: bool = False
    teacher_dropout: float = 0.1
    gumbel: bool = False
    replay_buffer: bool = False


@dataclass
class PromptModelConfig:
    model_name_or_path: Optional[str] = ''
    prefix_model_name_or_path: str = ''
    tuning_mode: str = 'prompt'
    preseqlen: int = 10
    prefix_dropout: float = 0.1


@dataclass
class StoryModelConfig:
    model_name_or_path: Optional[str] = ''
    prefix_model_name_or_path: str = ''
    tuning_mode: str = 'story'
    preseqlen: int = 10
    prefix_dropout: float = 0.1
    prompt: bool = False
    story_len: int = 4
    sent_embed: int = 256
    condition: bool = False
    clip_embed: bool = False


@dataclass
class DefaultConfig:
    dataset: DataConfig = DataConfig()
    stage1: Stage1Config = Stage1Config()
    stage2: Stage2Config = Stage2Config()


@dataclass
class FineTuningConfig:
    dataset: DataConfig = DataConfig()
    stage1: Stage1Config = Stage1Config()
    stage2: Stage2Config = Stage2Config()
    optimizer: OptConfig = OptConfig()
    experiment: ExpConfig = ExpConfig()


@dataclass
class PrefixTuningConfig:
    dataset: DataConfig = DataConfig()
    stage1: Stage1Config = Stage1Config()
    stage2: Stage2Config = Stage2Config()
    prefix: PrefixModelConfig = PrefixModelConfig()
    optimizer: OptConfig = OptConfig()
    experiment: ExpConfig = ExpConfig()


@dataclass
class PromptTuningConfig:
    dataset: DataConfig = DataConfig()
    stage1: Stage1Config = Stage1Config()
    stage2: Stage2Config = Stage2Config()
    prompt: PromptModelConfig = PromptModelConfig()
    optimizer: OptConfig = OptConfig()
    experiment: ExpConfig = ExpConfig()


@dataclass
class StoryConfig:
    dataset: DataConfig = DataConfig()
    stage1: Stage1Config = Stage1Config()
    stage2: Stage2Config = Stage2Config()
    story: StoryModelConfig = StoryModelConfig()
    optimizer: OptConfig = OptConfig()
    experiment: ExpConfig = ExpConfig()


def get_base_config(mode):
    if mode == 'default':
        return OmegaConf.structured(DefaultConfig)
    elif mode == 'finetuning':
        return OmegaConf.structured(FineTuningConfig)
    elif mode == 'prefixtuning':
        return OmegaConf.structured(PrefixTuningConfig)
    elif mode == 'prompt_tuning':
        return OmegaConf.structured(PromptTuningConfig)
    elif mode == 'story':
        return OmegaConf.structured(StoryConfig)
    else:
        raise ValueError
    # return OmegaConf.structured(DefaultConfig if use_default else FineTuningConfig)
