# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, CTRL, BERT, RoBERTa, XLNet).
GPT, GPT-2 and CTRL are fine-tuned using a causal language modeling (CLM) loss. BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss. XLNet is fine-tuned using a permutation language modeling (PLM) loss.
"""

import logging
import os, torch
from dalle.models import PrefixTuningDalle, Dalle, ConditionalDalle
import torchvision
import torchvision.transforms as transforms
import pytorch_lightning as pl
from tqdm import tqdm

import torch.nn.functional as F
import argparse
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

console_logger = logging.getLogger(__name__)

def get_cosine_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def evaluate(args, model, valid_loader, device, writer, iter, eval_iters=1, generate=False, num_candidates=16):
    val_loss, img_loss, txt_loss = 0.0, 0.0, 0.0
    model.eval()
    with torch.no_grad():
        for i, batch in tqdm(enumerate(valid_loader), desc="Validating"):  # loop through dataset by minibatch
            # text = torch.LongTensor(batch[1])  # a minibatch of text (numerical tokens)

            images = batch[0]
            texts = batch[1]
            if len(images.shape) == 5:
                B, V, C, H, W = images.shape
                images = images.view(B * V, C, H, W)
                B, V, S = texts.shape
                texts = texts.view(B * V, S)

            images = images.to(device)  # placeholder for images
            texts = texts.to(device)

            if args.conditional:
                src_images = batch[2]
                # print(src_images.shape)
                # making the source and target tensors of same size (in case story version is used)
                if len(src_images.shape) == 4 and src_images.shape[0] != B * V:
                    src_images = src_images.unsqueeze(1).repeat(1, V, 1, 1, 1)
                    B, V, C, H, W = src_images.shape
                    src_images = src_images.view(B * V, C, H, W)
                src_images = src_images.to(device)

            if args.conditional:
                logits_img, logits_txt, codes = model(images, src_images, texts)
            else:
                logits_img, logits_txt, codes = model(images, texts)
            # print(logits_img.shape, logits_txt.shape, codes.shape, texts.shape)

            loss_img = F.cross_entropy(logits_img.view(-1, logits_img.shape[-1]), codes.view(-1))
            loss_txt = F.cross_entropy(logits_txt.view(-1, logits_txt.shape[-1]), texts[:, 1:].reshape(-1))

            if i == 0:
                pred = torch.argmax(logits_img.view(-1, logits_img.shape[-1]), dim=-1)
                bs = images.shape[0]
                pred = pred.view(bs, 16, 16)  # [B, 16, 16]
                pixels = torch.clamp(model.stage1.decode_code(pred) * 0.5 + 0.5, 0, 1)  # [B, 256, 256]
                #
                writer.add_images('gt_batch', images.cpu().numpy(), iter)
                writer.add_images('pred_batch', pixels, iter)

                if args.conditional:
                    writer.add_images('src_batch', src_images.cpu().numpy(), iter)

                if generate:
                    images = []
                    for j, t in enumerate(texts):
                        if args.conditional:
                            pixels = model.sampling(t, src_images[j].unsqueeze(0), top_k=256,
                                                    num_candidates=num_candidates).cpu().numpy()
                        else:
                            pixels = model.sampling(t, top_k=256,
                                                    num_candidates=num_candidates).cpu().numpy()
                            # pixels = np.transpose(pixels, (0, 2, 3, 1))
                        images.append(pixels[0])
                    writer.add_images('gen_batch', np.stack(images, axis=0), iter)

            loss = loss_img + loss_txt
            val_loss += loss.item()
            img_loss += loss_img.item()
            txt_loss += loss_txt.item()

            if eval_iters == i+1:
                break

    writer.add_scalar('Loss/val_img_loss', img_loss/eval_iters, iter)
    writer.add_scalar('Loss/val_txt_loss', txt_loss/eval_iters, iter)

    return val_loss/eval_iters


def get_dataset(args, tokenizer, preprocess: transforms, mode = 'train'):

    if args.dataset_name == 'mscoco':
        import mscoco_dataloader as data
    elif args.dataset_name == 'pororo':
        import pororo_dataloader as data
    elif args.dataset_name == 'flintstones':
        import flintstones_dataloader as data
    elif args.dataset_name == 'didemo':
        import didemo_dataloader as data
    elif args.dataset_name == 'mpii':
        import mpii_dataloader as data
    else:
        raise ValueError

    if args.conditional:
        dataset = data.CopyImageDataset(args.data_dir, tokenizer, preprocess, mode=mode)
    else:
        dataset = data.ImageDataset(args.data_dir, tokenizer, preprocess, mode=mode)
    return dataset


def main(args):
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    if (
            os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
            and args.do_train
            and not args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Set seed
    pl.seed_everything(args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    grad_accm_steps = args.gradient_accumulation_steps
    args.max_steps = (84000 * args.num_train_epochs) // int(grad_accm_steps * args.per_gpu_train_batch_size)

    # Initiate config and tokenizer
    if args.conditional:
        model, config = ConditionalDalle.from_pretrained(args)
    else:
        model, config = Dalle.from_pretrained(args)

    # print(next(model.cross_attention_layers[0].parameters()).is_cuda)

    for param in model.stage1.parameters():
        param.requires_grad = False

    ##############################################################
    #################LOADING DATASETS ###########################

    train_transform = transforms.Compose(
        [transforms.Resize(config.dataset.image_resolution),
         transforms.RandomCrop(config.dataset.image_resolution),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    )

    valid_transform = transforms.Compose(
        [transforms.Resize(config.dataset.image_resolution),
         transforms.CenterCrop(config.dataset.image_resolution),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    )

    train_dataset = (
        get_dataset(args, model.tokenizer, train_transform, mode='train')  # if training_args.do_train else None
    )
    eval_dataset = (
        get_dataset(args, model.tokenizer, valid_transform, mode='val')
    )

    print("Training dataset size: %s", len(train_dataset))
    print("Validation dataset size: %s", len(eval_dataset))

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.per_gpu_train_batch_size * args.n_gpu,
        drop_last=True,
        shuffle=True,
        num_workers=int(args.dataloader_num_workers))

    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.per_gpu_eval_batch_size * args.n_gpu,
        drop_last=True,
        shuffle=False,
        num_workers=int(args.dataloader_num_workers))


    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    console_logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank, args.device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    log_dir = os.path.join(args.log_dir, args.dataset_name + '_' + 'full')
    if args.conditional:
        log_dir += '_cond'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    model_dir = os.path.join(args.output_dir, 'Model')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    writer = SummaryWriter(log_dir=log_dir, flush_secs=60)

    args.n_gpus = torch.cuda.device_count()
    # Calculate how many batches are accumulated
    grad_accm_steps = args.gradient_accumulation_steps
    config.optimizer.max_steps = (len(train_dataset) * args.num_train_epochs) // (args.per_gpu_train_batch_size * args.n_gpus)

    print("Maximum optimizer steps : %s" % config.optimizer.max_steps)

    print("Moving model to CUDA")
    model.to(device)

    # print(next(model.cross_attention_layers[0].parameters()).is_cuda)

    if args.conditional:
        for i in range(len(model.cross_attention_layers)):
            model.cross_attention_layers[i].to(device)
        print("Cross-attention layers are in cuda:", next(model.cross_attention_layers[0].parameters()).is_cuda)

    start_epoch = 0

    optimizer = torch.optim.AdamW(model.parameters(),
                            lr=config.optimizer.learning_rate,
                            betas=config.optimizer.betas,
                            weight_decay=config.optimizer.weight_decay)

    # sched = CosineAnnealingLR(opt,
    #                           T_max=self.config.optimizer.max_steps,
    #                           eta_min=self.config.optimizer.min_lr)

    def lr_lambda(current_step: int):
        return max(
            0.0, float(config.optimizer.max_steps - current_step) / float(max(1, config.optimizer.max_steps))
        )

    scheduler = LambdaLR(optimizer, lr_lambda)
    log_interval = 20
    eval_interval = 200
    gen_interval = 500

    for epoch in range(start_epoch, config.experiment.num_train_epochs):

        batch_idx = 0
        train_loss = 0.0

        for batch in tqdm(train_loader, desc="Training"):  # loop through dataset by minibatch
            model.train()
            # text = torch.LongTensor(batch[1])  # a minibatch of text (numerical tokens)

            images = batch[0]
            texts = batch[1]
            if len(images.shape) == 5:
                B, V, C, H, W = images.shape
                images = images.view(B * V, C, H, W)
                B, V, S = texts.shape
                texts = texts.view(B * V, S)

            images = images.to(device)  # placeholder for images
            texts = texts.to(device)

            if args.conditional:
                src_images = batch[2]
                # print(src_images.shape)
                # making the source and target tensors of same size (in case story version is used)
                if len(src_images.shape) == 4 and src_images.shape[0] != B * V:
                    src_images = src_images.unsqueeze(1).repeat(1, V, 1, 1, 1)
                    B, V, C, H, W = src_images.shape
                    src_images = src_images.view(B * V, C, H, W)
                src_images = src_images.to(device)

            # mask = torch.ones_like(text).bool().to(device) # ???
            # train and optimize a single minibatch
            optimizer.zero_grad()
            if args.conditional:
                logits_img, logits_txt, codes = model(images, src_images, texts)
            else:
                logits_img, logits_txt, codes = model(images, texts)

            loss_img = F.cross_entropy(logits_img.view(-1, logits_img.shape[-1]), codes.view(-1))
            loss_txt = F.cross_entropy(logits_txt.view(-1, logits_txt.shape[-1]), texts[:, 1:].reshape(-1))

            loss = loss_img + loss_txt

            train_loss += loss.item()
            loss.backward()
            optimizer.step()

            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * config.experiment.per_gpu_train_batch_size * args.n_gpus, len(train_dataset),
                           100. * batch_idx / len(train_loader), loss.item()))
                writer.add_scalar('Loss/train_img_loss', loss_img.item(), (epoch * len(train_loader)) + batch_idx)
                writer.add_scalar('Loss/train_txt_loss', loss_txt.item(), (epoch * len(train_loader)) + batch_idx)

            #
            if batch_idx % eval_interval == 0 and batch_idx > 0:
                val_loss = evaluate(args, model, eval_loader, device, writer,  (epoch * len(train_loader)) + batch_idx,
                                    eval_iters=50,
                                    generate= batch_idx % gen_interval == 0 and batch_idx > 0,
                                    num_candidates=2)
                print('====> Step: {}, Average val loss: {:.4f}'.format((epoch * len(train_loader)) + batch_idx, val_loss))

            batch_idx += 1

        print('====> Epoch: {} Average train loss: {:.4f}'.format(
            epoch, train_loss / len(train_loader)))

        scheduler.step(epoch)
        writer.flush()

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'iteration': (epoch * len(train_loader)) + batch_idx,
            'scheduler': scheduler.state_dict(),
            'loss': loss.item()},
            os.path.join(args.output_dir, 'Model', str(epoch) + ".pth"))

    return


# def _mp_fn(index):
#     # For xla_spawn (TPUs)
#     main()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='arguments for training/evaluating prefix-tuning DALLE')

    # Model Arguments
    parser.add_argument('--model_name_or_path', type=str, default=None, help='The model checkpoint for weights initialization.')
    parser.add_argument('--dalle_path', type=str, default='', help='The model checkpoint for weights initialization.')
    parser.add_argument("--conditional", action="store_true", help="Whether to use conditional dalle.")

    # Data Arguments
    parser.add_argument('--data_dir', type=str, default=None, help="Path to data directory")
    parser.add_argument('--dataset_name', type=str, default='mscoco', help="NAme of the t2i dataset")
    parser.add_argument('--train_embeddings', action="store_true", help="Whether to train word embeddings")
    parser.add_argument('--train_max_target_length', type=int, default=100, help='the max target length for training data.')
    parser.add_argument('--val_max_target_length', type=int, default=100, help='the max target length for dev data.')
    parser.add_argument('--dataloader_num_workers', type=int, default=8, help='number of workers when loading data')

    # Training Arguments
    parser.add_argument('--output_dir', type=str, default=None, help="Path to data directory")
    parser.add_argument('--log_dir', type=str, default=None, help="Path to log directory for logging events (Tensorboard or Wandb)")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run evaluation.")
    parser.add_argument("--do_test", action="store_true", help="Whether to run test.")
    parser.add_argument('--seed', type=int, default=42, help='seed for reproducibility')
    parser.add_argument("--overwrite_output_dir", action="store_true", help="Whether to overwrite output dir.")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--learning_rate_2", default=2e-4, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3, type=int, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )

    args = parser.parse_args()

    main(args)