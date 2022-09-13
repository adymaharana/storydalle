import os
import torch
import torchvision.utils as vutils
from torchvision.utils import save_image
from tqdm import tqdm
import numpy as np
import PIL

def save_as_image(tensor, out_dir, suffix):
    img = vutils.make_grid(tensor, video_len=1, padding=0)
    save_image(img, '%s/gen_sample_%s.png' % (out_dir, suffix))

def acc_tensors_to_images(tensor_dir, key, out_dir):

    files = [f for f in os.listdir(tensor_dir) if f.endswith('pt') and key in f]
    sorted_files = sorted(files, key=lambda x: int(x[:-3].split('_')[-1]))
    print(files[:10])
    print(sorted_files[:20])
    all_tensors = []
    for f in tqdm(files, desc="eading tensors"):
        t = torch.load(os.path.join(tensor_dir, f))
        # print(t[0].shape)
        # print(t.shape)
        all_tensors.append(t)
    all_tensors = torch.cat(all_tensors, dim=0)
    print(all_tensors.shape)

    torch.save(all_tensors, os.path.join(tensor_dir, 'sdalle_story_%s.pt' % key))

    # for i in tqdm(range(0, all_tensors.shape[0]), desc='Preapring images'):
    #     for j in range(0, all_tensors.shape[1]):
    #         save_as_image(all_tensors[i, j], out_dir, '%s_%s' % (i, j))

def images_to_numpy(tensor):
    generated = tensor.data.cpu().numpy().transpose(1, 2, 0)
    generated[generated < -1] = -1
    generated[generated > 1] = 1
    generated = (generated + 1) / 2 * 255
    return generated.astype('uint8')

def numpy_to_img(numpy_file, outdir, img_size):

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    x = np.load(numpy_file)
    print("Numpy image file shape: ", x.shape)
    for i in tqdm(range(x.shape[0])):
        frames = x[i, :, :, :, :]
        frames = np.swapaxes(frames, 0, 1)
        # frames = torch.Tensor(frames).view(-1, 3, 64, 64)
        # frames = torch.nn.functional.upsample(frames, size=(img_size, img_size), mode="bilinear")

        # vutils.save_image(vutils.make_grid(torch.Tensor(frames).view(-1, 3, 64, 64), 1, padding=0), 'sequence-2.png')
        all_images = images_to_numpy(vutils.make_grid(torch.Tensor(frames).view(-1, 3, 64, 64), 1, padding=0))
        # all_images = images_to_numpy(vutils.make_grid(frames, 1, padding=0))
        # print(all_images.shape)
        for j, idx in enumerate(range(64, all_images.shape[0] + 1, 64)):
            output = PIL.Image.fromarray(all_images[idx-64: idx, :, :])
            output.save(os.path.join(outdir, 'img-%s-%s.png' % (i, j)))
            img = PIL.Image.open(os.path.join(outdir, 'img-%s-%s.png' % (i, j)))
            if img_size != 64:
                img = img.resize((img_size, img_size,))
            img.save(os.path.join(outdir, 'img-%s-%s.png' % (i, j)))

if __name__ == "__main__":

    acc_tensors_to_images('/nas-ssd/adyasha/out/minDALLEs/pororo/', 'test', '/nas-ssd/adyasha/out/minDALLEs/pororo/test_images')
    # acc_tensors_to_images('/nas-ssd/adyasha/out/minDALLEs/didemo/', 'test', '/nas-ssd/adyasha/out/minDALLEs/didemo/test_images/images')
    # acc_tensors_to_images('/nas-ssd/adyasha/out/minDALLEs/flintstones/', 'test', '/nas-ssd/adyasha/out/minDALLEs/flintstones/test_images/images')

    # numpy_to_img('/nas-ssd/adyasha/out/SGANc/didemo/val-images-epoch-120.npy', '/nas-ssd/adyasha/out/SGANc/didemo/val_images/', 299)