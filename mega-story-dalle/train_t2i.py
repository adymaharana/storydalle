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
from min_dalle import MinDalle
import torchvision.transforms as transforms
import pytorch_lightning as pl
from tqdm import tqdm
import sys
from torch.autograd import Variable
from PIL import Image

import torch.nn.functional as F
import argparse
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.tensorboard import SummaryWriter
import numpy as np
# import wandb
# from dalle.utils.utils import save_image
# wandb.init(project="storydalle-project", entity="adymaharana")

console_logger = logging.getLogger(__name__)


def evaluate(args, model, valid_loader, device, writer, iter, eval_iters=1, generate=False, num_candidates=16):
    val_loss, img_loss, txt_loss = 0.0, 0.0, 0.0
    model.eval()
    with torch.no_grad():
        for i, batch in tqdm(enumerate(valid_loader), desc="Validating"):  # loop through dataset by minibatch
            # text = torch.LongTensor(batch[1])  # a minibatch of text (numerical tokens)
            images = batch[0]  # placeholder for images
            texts = batch[1].to(device)
            images = images.to(device)

            if len(images.shape) == 5:
                B, V, C, H, W = images.shape
                images = images.view(B * V, C, H, W)
                B, V, S = texts.shape
                texts = texts.view(B * V, S)
                # images = images[:, 0].squeeze()
                # texts = texts[:, 0].squeeze()
                # pass

            if args.tuning_mode == 'story':
                src_images = batch[2]
                src_images = src_images.to(device)
            #     sent_embeds = batch[3]
            #     sent_embeds = sent_embeds.to(device)
            #     # print(images.shape, src_images.shape, sent_embeds.shape)
                logits_img, codes = model(images, texts, src_images)
            else:
                logits_img, codes = model(images, texts, src_images)
                # print(logits_img.shape, logits_txt.shape, codes.shape, texts.shape)

            loss_img = F.cross_entropy(logits_img.view(-1, logits_img.shape[-1]), codes.view(-1))

            if i == 0:
                C, H, W = images.shape[-3:]
                pred = torch.argmax(logits_img.view(-1, logits_img.shape[-1]), dim=-1).view(-1, 16, 16)
                # if len(images.shape) == 5:
                #     B, L, C, H, W = images.shape
                #     pred = pred.view(B*L, 16, 16)
                # else:
                #     B = images.shape[0]
                #     pred = pred.view(B, 16, 16)  # [B, 16, 16]
                # print(pred.shape)
                # pixels = model.detokenizer.forward(False, pred).cpu().transpose(1, -1).numpy()  # [B, 256, 256]
                pixels = model.detokenizer.forward(False, pred).cpu().transpose(1, -1).transpose(-2, -1)  # [B, 256, 256]
                # print(pred.shape, pixels.shape)
                # saved_img = Image.fromarray(pixels.to(torch.uint8).to('cpu').numpy())
                # saved_img.save('pred.png')

                # pixels = pixels.cpu().transpose(1, -1).numpy()

                writer.add_images('gt_batch', images.cpu().view(-1, C, H, W).numpy(), iter)
                writer.add_images('pred_batch', pixels.to(torch.uint8), iter)

                if generate:
                    images = []
                    # concatenate the list of prompts (split by n_head) for better downstream processing
                    if args.tuning_mode == 'prefixtune':
                        past_key_values_prompt = model.get_prompt(bsz=num_candidates)
                        past_key_values_prompt = torch.cat([x.unsqueeze(0) for x in past_key_values_prompt], dim=0)
                        for t in texts:
                            pixels = model.sampling(t, past_key_values_prompt, top_k=256,
                                                    num_candidates=num_candidates).cpu().numpy()
                            # pixels = np.transpose(pixels, (0, 2, 3, 1))
                            images.append(pixels[0])
                        writer.add_images('gen_batch', np.stack(images, axis=0), iter)

                    elif args.tuning_mode == 'prompt_tune':
                        prompt = model.get_prompt(bsz=num_candidates)
                        for j, t in enumerate(texts):
                            pixels = model.sampling(t, prompt, top_k=256,
                                                    num_candidates=num_candidates).cpu().numpy()
                            # pixels = np.transpose(pixels, (0, 2, 3, 1))
                            images.append(pixels[0])
                        writer.add_images('gen_batch', np.stack(images, axis=0), iter)

                    elif args.tuning_mode == 'story':
                        # if args.condition:
                        writer.add_images('src_batch', src_images.cpu().numpy(), iter)
                        # if args.prompt:
                        #     prompt = model.get_prompt(bsz=texts.shape[1])
                        # else:
                        #     prompt = None
                        # pixels = model.sampling(texts[0], src_images[0].unsqueeze(0), sent_embeds[0].unsqueeze(0), top_k=256,
                        #                         num_candidates=num_candidates, prompt=prompt).cpu().numpy()
                        # pixels = np.transpose(pixels, (0, 2, 3, 1))
                        # images.append(pixels)


                        # pixels = model.sampling(texts[0], src_images[0].unsqueeze(0), sent_embeds[0].unsqueeze(0), top_k=256,
                        #                         num_candidates=num_candidates, prompt=prompt).cpu().numpy()
                        #
                        pixels = model.sample_images(texts, src_images).cpu().transpose(1, -1).transpose(-1, -2)
                        # pixels = model.sample_images(texts)

                        # saved_img = Image.fromarray(pixels.to(torch.uint8).to('cpu').numpy())
                        # saved_img.save('gen.png')

                        # pixels = pixels.cpu().transpose(1, -1).numpy()
                        writer.add_images('gen_batch', pixels.to(torch.uint8), iter)
                    else:
                        for t in texts:
                            pixels = model.sampling(t, top_k=256,
                                                    num_candidates=num_candidates).cpu().numpy()
                            # pixels = np.transpose(pixels, (0, 2, 3, 1))
                            images.append(pixels[0])
                        writer.add_images('gen_batch', np.stack(images, axis=0), iter)

            loss = loss_img
            val_loss += loss.item()
            img_loss += loss_img.item()

            if eval_iters == i+1:
                break

    writer.add_scalar('Loss/val_img_loss', img_loss/eval_iters, iter)
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
    else:
        raise ValueError

    if args.tuning_mode == 'story':
        # dataset = data.CopyImageDataset(args.data_dir, tokenizer, preprocess, mode=mode)
        dataset = data.StoryImageDataset(args.data_dir, tokenizer, preprocess, mode=mode)
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

    model = MinDalle(args.model_name_or_path,
                     dtype=torch.float32,
                     device='cuda',
                     is_mega=args.is_mega,
                     is_reusable=True,
                     is_train=True,
                     condition=args.tuning_mode == 'story')

    for param in model.vqgan_tokenizer.parameters():
        param.requires_grad = False
    for param in model.detokenizer.parameters():
        param.requires_grad = False
    print("Frozen VQGAN model parameters")

    # print("*******Encoder Parameters********")
    # for name, param in model.encoder.named_parameters():
    #     print(name, param.requires_grad)
        # if not param.requires_grad:
        #     print(name)
    # print("*******Decoder Parameters********")
    # for name, param in model.decoder.named_parameters():
    #     print(name, param.requires_grad)
        # if not param.requires_grad:
        #     print(name)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Total number of parameters: %s", pytorch_total_params)
    print("Total number of trainable parameters: %s", pytorch_total_params_trainable)

    # for name, param in model.named_parameters():
    #     print(name, param.requires_grad)
        # if not param.requires_grad:
        #     print(name)

    #############################################################
    # Add character names to tokenizer dictionary and resize token embeddings
    # TODO: Fix token embeddings
    if args.dataset_name == 'mscoco':
        pass
    elif args.dataset_name == 'pororo':
        n_new_tokens = 9
        with torch.no_grad():
            print(len(model.tokenizer.token_from_subword), model.encoder.embed_tokens.weight.shape)
            model.tokenizer.add_tokens(['pororo', 'loopy', 'eddy', 'harry', 'poby', 'tongtong', 'crong', 'rody', 'petty'])
            model.encoder.resize_token_embeddings(len(model.tokenizer.token_from_subword) + n_new_tokens)
    elif args.dataset_name == 'flintstones':
        n_new_tokens = 7
        with torch.no_grad():
            model.tokenizer.add_tokens(
                ['fred', 'barney', 'wilma', 'betty', 'pebbles', 'dino', 'slate'])
            model.encoder.resize_token_embeddings(len(model.tokenizer.token_from_subword) + n_new_tokens)
    elif args.dataset_name == 'didemo':
        pass
    else:
        raise ValueError

    ##############################################################
    #################LOADING DATASETS ###########################

    image_resolution = 256

    train_transform = transforms.Compose(
        [transforms.Resize(image_resolution),
         transforms.RandomCrop(image_resolution),
         transforms.ToTensor()]
         # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    )

    valid_transform = transforms.Compose(
        [transforms.Resize(image_resolution),
         transforms.CenterCrop(image_resolution),
         transforms.ToTensor()]
         # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
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

    if args.is_mega:
        suffix = '_mega'
    else:
        suffix = '_min'
    log_dir = os.path.join(args.log_dir, args.dataset_name + '_' + args.tuning_mode + suffix)
    if args.prompt:
        log_dir = log_dir + '_' + 'prompt'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    model_dir = os.path.join(args.output_dir, 'Model')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    writer = SummaryWriter(log_dir=log_dir, flush_secs=60)

    args.n_gpus = torch.cuda.device_count()
    # Calculate how many batches are accumulated
    grad_accm_steps = args.gradient_accumulation_steps
    # TODO: Adafactor optimizer / cyclic for conditional layers

    train_max_steps = (len(train_dataset) * args.num_train_epochs) // (args.per_gpu_train_batch_size * args.n_gpus)
    print("Maximum optimizer steps : %s" % train_max_steps)
    print("Moving model to CUDA")
    model.to(device)

    # if args.tuning_mode == 'story':
    #     if model.config.story.condition:
    #         for i in range(len(model.cross_attention_layers)):
    #             model.cross_attention_layers[i].to(device)
    #         print("Cross-attention layers are in cuda:", next(model.cross_attention_layers[0].parameters()).is_cuda)

    start_epoch = 0
    optimizer = torch.optim.AdamW(model.parameters(),
                            lr=args.learning_rate,
                            betas=(0.9, 0.99),
                            weight_decay=args.weight_decay)

    # sched = CosineAnnealingLR(opt,
    #                           T_max=self.config.optimizer.max_steps,
    #                           eta_min=self.config.optimizer.min_lr)

    # def lr_lambda(current_step: int):
    #     return max(
    #         0.0, float(config.optimizer.max_steps - current_step) / float(max(1, config.optimizer.max_steps))
    #     )

    def lr_lambda(current_step: int):
        warmup_steps = 1000
        return max(
            1.0, float(warmup_steps - current_step) / float(warmup_steps)
        )

    scheduler = LambdaLR(optimizer, lr_lambda)

    log_interval = args.logging_steps
    eval_interval = args.eval_steps
    gen_interval = args.generate_steps

    optimizer.zero_grad()

    for epoch in range(start_epoch, args.num_train_epochs):

        batch_idx = 0
        train_loss = 0.0

        pytorch_total_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Total number of trainable parameters: %s", pytorch_total_params_trainable)

        for batch in tqdm(train_loader, desc="Training"):  # loop through dataset by minibatch
            model.train()
            # text = torch.LongTensor(batch[1])  # a minibatch of text (numerical tokens)
            images = batch[0]  # placeholder for images
            texts = batch[1].to(device)
            # text_masks = batch[2].to(device)
            images = images.to(device)
            # mask = torch.ones_like(text).bool().to(device) # ???
            # train and optimize a single minibatch

            if args.tuning_mode == 'story':
                if len(images.shape) == 5:
                    B, V, C, H, W = images.shape
                    images = images.view(B * V, C, H, W)
                    B, V, S = texts.shape
                    texts = texts.view(B * V, S)
                    # images = images[:, 0].squeeze()
                    # texts = texts[:, 0].squeeze()
                    pass
                src_images = batch[2]
                src_images = src_images.to(device)
                # sent_embeds = sent_embeds.to(device)

                for i in [0, 2]:
                    # breaking it down to 2 passes to fit into GPU
                    logits_img, codes = model(images[i:i+2], texts[i:i+2], src_images)
                    loss = F.cross_entropy(logits_img.view(-1, logits_img.shape[-1]), codes.view(-1))
                    train_loss += loss.item()
                    loss.backward()

            else:
                if len(images.shape) == 5:
                    B, V, C, H, W = images.shape
                    images = images.view(B * V, C, H, W)
                    B, V, S = texts.shape
                    texts = texts.view(B * V, S)
                    # images = images[:, 0].squeeze()
                    # texts = texts[:, 0].squeeze()
                    pass
                logits_img, codes = model(images[:2], texts[:2])

                loss = F.cross_entropy(logits_img.view(-1, logits_img.shape[-1]), codes.view(-1))
                train_loss += loss.item()
                loss.backward()

            if batch_idx % args.gradient_accumulation_steps == 0:
                # every 10 iterations of batches of size 10
                optimizer.step()
                optimizer.zero_grad()

                # print(model.decoder.lm_head.weight.grad.shape)

            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * args.per_gpu_train_batch_size * args.n_gpus, len(train_dataset),
                           100. * batch_idx / len(train_loader), loss.item()))
                # with torch.no_grad():
                #     probs = F.softmax(logits_img, dim=-1)
                #     print(torch.topk(probs[0, 0, :], 10), codes.view(-1)[0])

                writer.add_scalar('Loss/train_img_loss', loss.item(), (epoch * len(train_loader)) + batch_idx)
                # writer.add_scalar('Loss/train_txt_loss', loss_txt.item(), (epoch * len(train_loader)) + batch_idx)

            if batch_idx % eval_interval == 0:
                pass

                val_loss = evaluate(args, model, eval_loader, device, writer, (epoch * len(train_loader)) + batch_idx,
                                    eval_iters=50,
                                    generate=batch_idx % gen_interval == 0,
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
        # model.tokenizer.save_model(os.path.join(args.output_dir, 'Model'))

    return


# def _mp_fn(index):
#     # For xla_spawn (TPUs)
#     main()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='arguments for training/evaluating prefix-tuning DALLE')

    # Model Arguments
    parser.add_argument('--model_name_or_path', type=str, default=None, help='The model checkpoint for weights initialization.')
    parser.add_argument("--is_mega", action="store_true", help="Whether to run training.")
    parser.add_argument('--prefix_mode', type=str, default='activation', help='activation or embedding')
    parser.add_argument('--preseqlen', type=int, default=0, help='how many tokens of prefix should we include.')
    parser.add_argument('--optim_prefix', action="store_true", help='set to True if optimizing prefix directly; no if through amortized function')
    parser.add_argument('--tuning_mode', type=str, default='prefixtune', help='prefixtune or finetune')
    parser.add_argument('--top_k_layers', type=int, default=2, help='In finetuning setting, if we only tune the top k layers.')
    parser.add_argument('--parameterize_mode', type=str, default='mlp', help="mlp or emb to parametrize when we optimize for the embeddings.")
    parser.add_argument('--prefix_dropout', type=float, default=0.0, help='dropout rate for the prefix tuning model.')
    parser.add_argument('--teacher_dropout', type=float, default=0.0, help='dropout rate for the teacher model.')
    parser.add_argument('--init_random', action="store_true", help="set True if initializing random embeddings")
    parser.add_argument('--init_shallow', action="store_true", help="set True if not using reparameterization")
    parser.add_argument('--init_shallow_word', type=bool, default=False, help="set True if init_shallow and specify words")
    parser.add_argument('--replay_buffer', action="store_true", help="set True if using replay buffer in training")
    parser.add_argument('--gumbel', action="store_true", help="set True if using the gumbel softmax in training")
    parser.add_argument('--hidden_dim_prefix', type=float, default=512, help="hidden dim of MLP for generating prefix?")

    # Data Arguments
    parser.add_argument('--data_dir', type=str, default=None, help="Path to data directory")
    parser.add_argument('--dataset_name', type=str, default='mscoco', help="NAme of the t2i dataset")
    parser.add_argument('--lowdata_token', type=str, default='story', help="The token to be prepended at initialization time.")
    parser.add_argument('--use_lowdata_token', type=bool, default=True, help="Whether we should use the lowdata token for prefix-tuning")
    parser.add_argument('--train_embeddings', action="store_true", help="Whether to train word embeddings")
    parser.add_argument('--train_max_target_length', type=int, default=100, help='the max target length for training data.')
    parser.add_argument('--val_max_target_length', type=int, default=100, help='the max target length for dev data.')
    parser.add_argument('--dataloader_num_workers', type=int, default=8, help='number of workers when loading data')

    # new arguments for story
    parser.add_argument('--prompt', action="store_true", help="set True if using prompts in StoryDALLE")
    parser.add_argument('--story_len', type=int, default=4, help='the max target length for dev data.')
    parser.add_argument('--sent_embed', type=int, default=384, help='the max target length for dev data.')
    parser.add_argument('--condition', action="store_true", help="set True if using prompts in StoryDALLE")
    parser.add_argument('--clip_embed', action="store_true", help="set True if using prompts in StoryDALLE")

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
    parser.add_argument("--eval_steps", type=int, default=50, help="Log every X updates steps.")
    parser.add_argument("--generate_steps", type=int, default=50, help="Log every X updates steps.")
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