import os
import json
from matplotlib import pyplot as plt
import clip
import random
import torch
from dalle.models import Dalle, ConditionalDalle, PrefixTuningDalle
from dalle.utils.utils import set_seed, clip_score
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm
from collections import defaultdict
from torchvision.datasets import ImageFolder
# torch.multiprocessing.set_start_method('spawn', force=True)
import torchvision.utils as vutils
from torchvision.utils import save_image
import torchvision.transforms as transforms
from torch.multiprocessing import Pool, Process, set_start_method
from PIL import Image

try:
     set_start_method('spawn')
except RuntimeError:
    pass

device = 'cuda:0'
set_seed(0)

parser = ArgumentParser(description='arguments for training/evaluating prefix-tuning DALLE')

# Model Arguments
parser.add_argument('--model_name_or_path', type=str, default=None, help='The model checkpoint for weights initialization.')
parser.add_argument('--dalle_path', type=str, default=None, help='The model checkpoint for weights initialization.')
parser.add_argument('--dataset', type=str, default='pororo', help='Dataset name.')
parser.add_argument("--do_train", action="store_true", help="Whether to run evaluation.")
parser.add_argument("--do_eval", action="store_true", help="Whether to run evaluation.")
parser.add_argument("--conditional", action="store_true", help="Whether to use the conditional model.")
parser.add_argument("--prefix", action="store_true", help="Whether to use the conditional model.")
args = parser.parse_args()

def images_to_numpy(tensor):
    generated = tensor.data.cpu().numpy().transpose(1,2,0)
    generated[generated < -1] = -1
    generated[generated > 1] = 1
    generated = (generated + 1) / 2 * 255
    return generated.astype('uint8')

def save_story_results(images, texts, image_dir, epoch, mode='val', dataset = 'flint', ground_truth=None, video_len=4):
    # print("Generated Images shape: ", images.shape)

    print(images.shape)
    all_images = []
    for i in range(images.shape[0]):
        all_images.append(vutils.make_grid(images[i], video_len, padding=0))
    all_images = vutils.make_grid(all_images, 1, padding=0)

    # all_images = images_to_numpy(all_images)
    if ground_truth:
        imgs = []
        for gt in ground_truth:
            img = vutils.make_grid(gt, video_len+1, padding=0)
            imgs.append(img)
        ground_truth = vutils.make_grid(imgs, 1, padding=0)
        print(ground_truth.shape, all_images.shape)
        all_images = vutils.make_grid(torch.cat([all_images, ground_truth], dim=-1), 1, padding=5)

    save_image(all_images, '%s/%s_fake_samples_%s_epoch_%03d.png' % (image_dir, dataset, mode, epoch))

    if texts is not None:
        fid = open('%s/%s_fake_samples_%s_epoch_%03d.txt' % (image_dir, dataset, mode, epoch), 'w')
        for i, story in enumerate(texts):
            fid.write(str(i) + '--------------------------------------------------------\n')
            for j in range(len(story)):
                fid.write(story[j] +'\n' )
            fid.write('\n\n')
        fid.close()
    return


class MSCOCOImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, mode='train', splits=1, process_idx=0):

        self.data_dir = data_dir
        if mode == 'train':
            self.img_dir = os.path.join(self.data_dir, 'train2014')
            annotations = json.loads(open(os.path.join(self.data_dir, 'annotations', 'captions_train2014.json')).read())
        elif mode == 'val':
            self.img_dir = os.path.join(self.data_dir, 'val2014')
            annotations = json.loads(open(os.path.join(self.data_dir, 'annotations', 'captions_val2014.json')).read())
        elif mode == 'test':
            pass
        else:
            raise ValueError

        self.mode = mode
        self.id2name = {}
        for ann in tqdm(annotations['images'], desc='Reading image filenames'):
            self.id2name[ann['id']] = ann['file_name']
        self.ids = list(set(self.id2name.keys()))
        self.ids.sort()

        if mode in ['val', 'test']:
            # self.ids = random.sample(self.ids, k=200)
            total_len = len(self.ids)
            include_idxs = list(range(total_len))[int(process_idx*total_len/splits):int((process_idx+1)*total_len/splits)]
            self.ids = [self.ids[idx] for idx in include_idxs]

        self.id2captions = defaultdict(lambda: [])
        for cap in tqdm(annotations['annotations'], desc='Reading captions'):
            self.id2captions[cap['image_id']].append(cap['caption'])
        keys = list(set(self.id2captions.keys()))
        # assert all([img_id in keys for img_id in keys]), "Key not found in annotations"
        # assert len(keys) == len(self.ids), (len(keys), len(self.ids))


    def __getitem__(self, item):

        img_id = self.ids[item]
        img_path = os.path.join(self.img_dir, self.id2name[img_id])
        caption = random.sample(self.id2captions[img_id], k=1)[0]
        return img_path, self.id2name[img_id], caption

    def __len__(self):
        return len(self.ids)


def infer_model(splits=1, process_idx=0):
    # prompt = "A painting of a monkey with sunglasses in the frame"
    # prompt = "A group of elephants walking in muddy water"

    debug = False

    if not debug:
        if args.conditional:
            model, _ = ConditionalDalle.from_pretrained(args)
        elif args.prefix:
            model, _ = PrefixTuningDalle.from_pretrained(args)
        else:
            model, _ = Dalle.from_pretrained(args)  # This will automatically download the pretrained model.
        model.eval()
        model.to(device=device)

        if args.conditional:
            for i in range(len(model.cross_attention_layers)):
                model.cross_attention_layers[i].to(device)
            print("Cross-attention layers are in cuda:", next(model.cross_attention_layers[0].parameters()).is_cuda)


    if args.dataset == 'pororo':
        from pororo_dataloader import StoryImageDataset
        dataset = StoryImageDataset('/nas-ssd/adyasha/datasets/pororo_png', 256, mode='test')
        video_len = 4
    elif args.dataset == 'flintstones':
        from flintstones_dataloader import StoryImageDataset
        dataset = StoryImageDataset('/nas-ssd/adyasha/datasets/flintstones', mode='val')
        video_len = 4
    elif args.dataset == 'mpii':
        from mpii_dataloader import StoryImageDataset
        dataset = StoryImageDataset('/nas-ssd/adyasha/datasets/mpii', mode='val')
        video_len = 2
    else:
        # dataset = MSCOCOImageDataset('./mscoco2014', mode='val', splits=splits, process_idx=process_idx)
        from didemo_dataloader import StoryImageDataset
        dataset = StoryImageDataset('/nas-ssd/adyasha/datasets/didemo', mode='val')
        video_len = 3

    print("Found %s images in dataset" % len(dataset))

    stories = []
    texts = []
    gts = []

    idxs = random.sample(list(range(len(dataset))), k=10)

    # with open('/nas-ssd/adyasha/out/sdalle_%s_human_eval_idxs.txt' % args.dataset, 'w') as f:
    #     f.write('\n'.join([str(n) for n in idxs]))

    for i, idx in tqdm(enumerate(idxs)):

        imgs, prompts = dataset[idx]
        if args.dataset == 'pororo':
            prompts = [p[0] for p in prompts]

        # Sampling
        story = []
        gts.append(imgs)
        print(prompts)
        for prompt in prompts[1:]:
            if not debug:

                if args.conditional:
                    images = model.sampling(prompt=prompt,
                                            source=transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(imgs[0]).unsqueeze(0),
                                            top_k=256, # It is recommended that top_k is set lower than 256.
                                            top_p=None,
                                            softmax_temperature=1.0,
                                            num_candidates=4,
                                            device=device).cpu()
                else:
                    images = model.sampling(prompt=prompt,
                                            top_k=256, # It is recommended that top_k is set lower than 256.
                                            top_p=None,
                                            softmax_temperature=1.0,
                                            num_candidates=4,
                                            device=device).cpu()

                story.append(images[0])
                # print(images.shape)
            else:
                story.append(torch.rand(3, 256, 256))

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

        stories.append(torch.stack(story))
        texts.append([p for p in prompts])

        if i%5 == 0:
            save_story_results(torch.stack(stories), texts, '/nas-ssd/adyasha/out', 0, mode='val', dataset =args.dataset, video_len=video_len, ground_truth=gts)
            torch.save(torch.stack(stories), '/nas-ssd/adyasha/out/sdalle_%s_val.pt' % args.dataset)

    save_story_results(torch.stack(stories), texts, '/nas-ssd/adyasha/out', 0, mode='val', dataset=args.dataset, video_len=video_len, ground_truth=gts)
    torch.save(torch.stack(stories), '/nas-ssd/adyasha/out/sdalle_%s_val.pt' % args.dataset)


def infer_images(n_processes=1):

    if n_processes > 1:

        jobs = []
        for i in range(0, n_processes):
            process = Process(target=infer_model,
                                              args=(n_processes, i))
            jobs.append(process)

        # Start the processes (i.e. calculate the random number lists)
        for j in jobs:
            j.start()

        # Ensure all of the processes have finished
        for j in jobs:
            j.join()

    else:

        infer_model()

if __name__ == "__main__":
    infer_images(1)