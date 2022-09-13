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
from dalle.models import PrefixTuningDalle, StoryDalle, PromptDalle
import torchvision
import torchvision.transforms as transforms
import pytorch_lightning as pl
from datetime import datetime
import argparse
import random
from tqdm import tqdm
import torchvision.utils as vutils
from torchvision.utils import save_image

console_logger = logging.getLogger(__name__)


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

    if args.tuning_mode == 'story' or args.tuning_mode == 'prompt':
        dataset = data.StoryImageDataset(args.data_dir, tokenizer, preprocess, mode=mode)
    else:
        dataset = data.ImageDataset(args.data_dir, tokenizer, preprocess, mode=mode)
    return dataset


def images_to_numpy(tensor):
    generated = tensor.data.cpu().numpy().transpose(1,2,0)
    generated[generated < -1] = -1
    generated[generated > 1] = 1
    generated = (generated + 1) / 2 * 255
    return generated.astype('uint8')


def inverse_normalize(tensor, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device)
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    tensor.mul_(std).add_(mean)
    return tensor


def save_story_results(images, texts, image_dir, epoch,
                       mode='val', dataset = 'flint',
                       ground_truth=None, video_len=4, source=None):
    # print("Generated Images shape: ", images.shape)

    all_images = []
    for i in range(len(images)):
        all_images.append(vutils.make_grid(images[i], video_len, padding=0))
    all_images = vutils.make_grid(all_images, 1, padding=0)

    # all_images = images_to_numpy(all_images)
    if ground_truth is not None:
        ground_truth = inverse_normalize(ground_truth)
        imgs = []
        for gt in ground_truth:
            img = vutils.make_grid(gt, video_len, padding=0)
            imgs.append(img)
        ground_truth = vutils.make_grid(imgs, 1, padding=0)
        print(ground_truth.shape, all_images.shape)
        all_images = vutils.make_grid(torch.cat([ground_truth, all_images], dim=-1), 1, padding=5)

    if source is not None:
        source = inverse_normalize(source)
        source = vutils.make_grid(source, 1, padding=0)
        print(source.shape, all_images.shape)
        all_images = vutils.make_grid(torch.cat([source, all_images], dim=-1), 1, padding=5)

    save_image(all_images, '%s/%s_gen_samples_%s_batch_%03d.png' % (image_dir, dataset, mode, epoch))

    if False:
    # if texts is not None:
        fid = open('%s/%s_fake_samples_%s_batch_%03d.txt' % (image_dir, dataset, mode, epoch), 'w')
        for i, story in enumerate(texts):
            fid.write(str(i) + '--------------------------------------------------------\n')
            for j in range(len(story)):
                fid.write(story[j] +'\n' )
            fid.write('\n\n')
        fid.close()
    return


def main(args):

    # Set seed
    pl.seed_everything(args.seed)

    device = 'cuda:0'

    # Initiate config and tokenizer
    if args.tuning_mode == 'story':
        model, config = StoryDalle.from_pretrained(args)
    elif args.tuning_mode == 'prompt':
        model, config = PromptDalle.from_pretrained(args)
    else:
        model, config = PrefixTuningDalle.from_pretrained(args)
    model.eval()
    model.to(device=device)

    # Add character names to tokenizer dictionary and resize token embeddings
    if args.dataset_name == 'mscoco':
        pass
    elif args.dataset_name == 'pororo':
        n_new_tokens = 9
        with torch.no_grad():
            model.tokenizer.add_tokens(['pororo', 'loopy', 'eddy', 'harry', 'poby', 'tongtong', 'crong', 'rody', 'petty'])
            # model.stage2.resize_token_embeddings(model.stage2.tok_emb_txt.weight.shape[0] + n_new_tokens)
    elif args.dataset_name == 'flintstones':
        n_new_tokens = 7
        with torch.no_grad():
            model.tokenizer.add_tokens(
                ['fred', 'barney', 'wilma', 'betty', 'pebbles', 'dino', 'slate'])
            # model.stage2.resize_token_embeddings(model.stage2.tok_emb_txt.weight.shape[0] + n_new_tokens)
    elif args.dataset_name == 'didemo':
        pass
    else:
        raise ValueError

    if args.tuning_mode == 'story':
        if model.config.story.condition:
            for i in range(len(model.cross_attention_layers)):
                model.cross_attention_layers[i].to(device)
            print("Cross-attention layers are in cuda:", next(model.cross_attention_layers[0].parameters()).is_cuda)

    valid_transform = transforms.Compose(
        [transforms.Resize(config.dataset.image_resolution),
         transforms.CenterCrop(config.dataset.image_resolution),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    )

    eval_dataset = (
        get_dataset(args, model.tokenizer, valid_transform, mode=args.mode)
    )

    print("%s Dataset size: %s" % (args.mode, len(eval_dataset)))

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.per_gpu_eval_batch_size * args.n_gpu,
        drop_last=False,
        shuffle=False,
        num_workers=int(args.dataloader_num_workers))

    console_logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank, args.device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    gts = []
    debug = False
    args.n_gpus = torch.cuda.device_count()
    # idxs = random.sample(list(range(len(eval_dataset))), k=10)
    for idx, batch in tqdm(enumerate(eval_loader)):

        # if args.dataset_name == 'flintstones' and args.mode == 'test':
        #     if idx < 38:
        #         continue
        # elif args.dataset_name == 'pororo' and args.mode == 'test':
        #     if idx < 5:
        #         continue
        #     if idx > 10:
        #         break

        # img, prompt = eval_dataset[idx]
        # if args.dataset == 'pororo':
        #     prompts = [p[0] for p in prompts]

        # Sampling
        # pred = []
        # gts.append(img)
        with torch.no_grad():
            if args.tuning_mode == 'story':

                stories = []
                images = batch[0]  # placeholder for images
                texts = batch[1]
                src_images = batch[2]
                sent_embeds = batch[3]
                if args.prompt:
                    prompt = model.get_prompt(bsz=texts.shape[1]).to(device)
                else:
                    prompt = None
                print(batch[0].shape[0])
                for i in range(texts.shape[0]):
                    print(i)
                    # pixels = model.sampling(texts[i].to(device),
                    # src_images[i].unsqueeze(0).to(device),
                    # sent_embeds[i].unsqueeze(0).to(device),
                    # top_k=32, top_p=0.2, prompt=prompt).cpu() # for pororo arxiv, k=32, p=0.2; for pororo/flintstones/didemo camera-ready, k=256
                    # pixels = model.sampling(texts[i].to(device), src_images[i].unsqueeze(0).to(device), sent_embeds[i].unsqueeze(0).to(device), top_k=96, top_p=0.5, prompt=prompt).cpu() # flintstones arxiv
                    pixels = model.sampling(texts[i].to(device), src_images[i].unsqueeze(0).to(device), sent_embeds[i].unsqueeze(0).to(device), top_k=96, top_p=0.5, prompt=prompt).cpu() # flintstones arxiv
                    # print(pixels.shape)
                    stories.append(pixels)

                save_story_results(stories, texts, args.output_dir, idx, mode=args.mode, dataset='mscoco', video_len=args.story_len, ground_truth=images, source = src_images)
                torch.save(torch.stack(stories), os.path.join(args.output_dir, 'sdalle_story_%s_batch_%s.pt' % (args.mode, idx)))
            elif args.tuning_mode == 'prompt':
                stories = []
                images = batch[0]  # placeholder for images
                texts = batch[1]
                src_images = batch[2]
                sent_embeds = batch[3]
                prompt = model.get_prompt(bsz=texts.shape[1]).to(device)

                for i in range(texts.shape[0]):
                    pixels = model.sampling(texts[i].to(device), prompt, top_k=256, num_candidates=args.story_len).cpu()
                    # print(pixels.shape)
                    stories.append(pixels)

                save_story_results(stories, texts, args.output_dir, idx, mode=args.mode, dataset='mscoco', video_len=args.story_len, ground_truth=images, source = src_images)
                torch.save(torch.stack(stories), os.path.join(args.output_dir, 'sdalle_story_%s_batch_%s.pt' % (args.mode, idx)))
            else:
                # imgs, prompts = batch
                # images = model.predict_step([imgs.to(device=device), prompts.to(device=device)], idx)
                pass

            # images = model.predict(prompt=prompt,
            #                         top_k=256, # It is recommended that top_k is set lower than 256.
            #                         top_p=None,
            #                         softmax_temperature=1.0,
            #                         num_candidates=4,
            #                         device=device).cpu()

            # pred.append(images[0])
            # print(images.shape)


        # images = torch.transpose(images, (0, 2, 3, 1))

        # # CLIP Re-ranking
        # model_clip, preprocess_clip = clip.load("ViT-B/32", device=device)
        # model_clip.to(device=device)
        # rank = clip_score(prompt=prompt,
        #                   images=images,
        #                   model_clip=model_clip,
        #                   preprocess_clip=preprocess_clip,
        #                   device=device)
        #
        # # Plot images
        # images = images[rank]
        # # plt.imshow(images[0])

        # texts.append(prompt)

    #     if i%5 == 0:
    #         save_story_results(torch.stack(story), texts, '/nas-ssd/adyasha/out', 0, mode='val', dataset = 'mscoco', video_len=1, ground_truth=gts)
    #         torch.save(torch.stack(story), '/nas-ssd/adyasha/out/sdalle_prefix_%s_val.pt' % 'mscoco')
    #
    # save_story_results(torch.stack(story), texts, '/nas-ssd/adyasha/out', 0, mode='val', dataset= 'mscoco', video_len=1, ground_truth=gts)
    # torch.save(torch.stack(story), '/nas-ssd/adyasha/out/sdalle_prefix_%s_val.pt' % 'mscoco')

    # # Build trainer
    # trainer = pl.Trainer(accelerator='gpu',
    #                      gpus=[0])
    # # Infer
    # trainer.predict(model, eval_loader)
    return


# def _mp_fn(index):
#     # For xla_spawn (TPUs)
#     main()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='arguments for training/evaluating prefix-tuning DALLE')

    # Model Arguments
    parser.add_argument('--model_name_or_path', type=str, default=None, help='The model checkpoint for weights initialization.')
    parser.add_argument('--prefix_model_name_or_path', type=str, default=None, help='The prefix model checkpoint for weights initialization.')
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
    parser.add_argument('--supercondition', type=float, default=1.0, help="hidden dim of MLP for generating prefix?")

    # Data Arguments
    parser.add_argument('--dataset_name', type=str, default='pororo', help="dataset name")
    parser.add_argument('--data_dir', type=str, default=None, help="Path to data directory")
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

    parser.add_argument('--mode', type=str, default='val', help="mval or test.")

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