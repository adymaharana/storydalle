# ------------------------------------------------------------------------------------
# Minimal DALL-E
# Copyright (c) 2021 KakaoBrain. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------

import os
import random
import urllib
import hashlib
import tarfile
import torch
import clip
import numpy as np
from PIL import Image
from torch.nn import functional as F
from tqdm import tqdm
import torchvision.utils as vutils
import matplotlib.pyplot as plt


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def clip_score(prompt: str,
               images: np.ndarray,
               model_clip: torch.nn.Module,
               preprocess_clip,
               device: str) -> np.ndarray:
    images = [preprocess_clip(Image.fromarray((image*255).astype(np.uint8))) for image in images]
    images = torch.stack(images, dim=0).to(device=device)
    texts = clip.tokenize(prompt).to(device=device)
    texts = torch.repeat_interleave(texts, images.shape[0], dim=0)

    image_features = model_clip.encode_image(images)
    text_features = model_clip.encode_text(texts)

    scores = F.cosine_similarity(image_features, text_features).squeeze()
    rank = torch.argsort(scores, descending=True).cpu().numpy()
    return rank


def download(url: str, root: str) -> str:
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)
    pathname = filename[:-len('.tar.gz')]

    expected_md5 = url.split("/")[-2]
    download_target = os.path.join(root, filename)
    result_path = os.path.join(root, pathname)

    if os.path.isfile(download_target) and (os.path.exists(result_path) and not os.path.isfile(result_path)):
        return result_path

    with urllib.request.urlopen(url) as source, open(download_target, 'wb') as output:
        with tqdm(total=int(source.info().get('Content-Length')), ncols=80, unit='iB', unit_scale=True,
                  unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.md5(open(download_target, 'rb').read()).hexdigest() != expected_md5:
        raise RuntimeError(f'Model has been downloaded but the md5 checksum does not not match')

    with tarfile.open(download_target, 'r:gz') as f:
        pbar = tqdm(f.getmembers(), total=len(f.getmembers()))
        for member in pbar:
            pbar.set_description(f'extracting: {member.name} (size:{member.size // (1024 * 1024)}MB)')
            f.extract(member=member, path=root)

    return result_path


def realpath_url_or_path(url_or_path: str, root: str = None) -> str:
    if urllib.parse.urlparse(url_or_path).scheme in ('http', 'https'):
        return download(url_or_path, root)
    return url_or_path


def images_to_numpy(tensor):
    generated = tensor.data.cpu().numpy().transpose(1,2,0)
    generated[generated < -1] = -1
    generated[generated > 1] = 1
    generated = (generated + 1) / 2 * 255
    return generated.astype('uint8')


def save_image(ground_truth, images, out_dir, batch_idx):

    for i, im in enumerate(images):
        if len(im.shape) == 3:
            plt.imsave(os.path.join(out_dir, 'test_%s_%s.png' % (batch_idx, i)), im)
        else:
            bs = im.shape[0]
            # plt.imsave()
            for j in range(bs):
                plt.imsave(os.path.join(out_dir, 'test_%s_%s_%s.png' % (batch_idx, i, j)), im[j])


    # print("Ground truth Images shape: ", ground_truth.shape, len(images))

    # images = vutils.make_grid(images, nrow=ground_truth.shape[0])
    # images = images_to_numpy(images)
    #
    # if ground_truth is not None:
    #     ground_truth = vutils.make_grid(ground_truth, 5)
    #     ground_truth = images_to_numpy(ground_truth)
    #     print("Ground Truth shape, Generated Images shape: ", ground_truth.shape, images.shape)
    #     images = np.concatenate([ground_truth, images], axis=0)
    #
    # output = Image.fromarray(images)
    # output.save('%s/fake_samples_epoch_%03d.png' % (out_dir, batch_idx))

    # if texts is not None:
    #     fid = open('%s/fake_samples_epoch_%03d_%03d.txt' % (image_dir, epoch, idx), 'w')
    #     for idx in range(images.shape[0]):
    #         fid.write(str(idx) + '--------------------------------------------------------\n')
    #         for i in range(len(texts)):
    #             fid.write(texts[i][idx] + '\n')
    #         fid.write('\n\n')
    #     fid.close()
    return