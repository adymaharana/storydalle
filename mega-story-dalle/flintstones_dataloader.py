import os, pickle
from tqdm import tqdm
import numpy as np
import torch.utils.data
import PIL
from random import randrange
import json
from torchvision import transforms
from PIL import Image

unique_characters = ["Wilma", "Fred", "Betty", "Barney", "Dino", "Pebbles", "Mr Slate"]

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, dir_path, tokenizer, preprocess, mode='train'):
        self.dir_path = dir_path

        splits = json.load(open(os.path.join(self.dir_path, 'train-val-test_split.json'), 'r'))
        train_id, val_id, test_id = splits["train"], splits["val"], splits["test"]

        if mode == 'train':
            self.orders = train_id
        elif mode =='val':
            self.orders = val_id
        elif mode == 'test':
            self.orders = test_id
        else:
            raise ValueError
        print("Total number of clips {}".format(len(self.orders)))

        annotations = json.load(open(os.path.join(self.dir_path, 'flintstones_annotations_v1-0.json')))
        self.descriptions = {}
        for sample in annotations:
            self.descriptions[sample["globalID"]] = sample["description"]

        self.preprocess = preprocess
        self.tokenizer = tokenizer

    def __getitem__(self, item):

        # single image input
        globalID = self.orders[item]
        path = os.path.join(self.dir_path, 'video_frames_sampled', globalID + '.npy')
        arr = np.load(path)
        n_frames = arr.shape[0]
        random_range = randrange(n_frames)
        im = arr[random_range]
        image = np.array(im)
        image = PIL.Image.fromarray(image.astype('uint8'), 'RGB')
        text = self.descriptions[globalID]
        tokens = self.tokenizer.encode(text.lower())
        tokens = torch.LongTensor(tokens.ids)
        image = self.preprocess(image)

        return image, tokens

    def __len__(self):
        return len(self.orders)


class StoryImageDataset(torch.utils.data.Dataset):
    def __init__(self, dir_path, tokenizer, transform=None, mode='train', im_input_size=128, out_img_folder='', return_labels=False):
        self.dir_path = dir_path

        splits = json.load(open(os.path.join(self.dir_path, 'train-val-test_split.json'), 'r'))
        train_id, val_id, test_id = splits["train"], splits["val"], splits["test"]

        min_len = 4
        if os.path.exists(os.path.join(self.dir_path, 'following_cache' + str(min_len) +  '.pkl')):
            self.followings = pickle.load(open(os.path.join(self.dir_path, 'following_cache' + str(min_len) + '.pkl'), 'rb'))
        else:
            print("Cache does not exist")
            all_clips = train_id + val_id + test_id
            all_clips.sort()
            for idx, clip in enumerate(tqdm(all_clips, desc="Counting total number of frames")):
                season, episode = int(clip.split('_')[1]), int(clip.split('_')[3])
                has_frames = True
                for c in all_clips[idx+1:idx+min_len+1]:
                    s_c, e_c = int(c.split('_')[1]), int(c.split('_')[3])
                    if s_c != season or e_c != episode:
                        has_frames = False
                        break
                if has_frames:
                    self.followings[clip] = all_clips[idx+1:idx+min_len+1]
                else:
                    continue
            pickle.dump(self.followings, open(os.path.join(self.dir_path, 'following_cache' + str(min_len) + '.pkl'), 'wb'))

        train_id = [tid for tid in train_id if tid in self.followings]
        val_id = [vid for vid in val_id if vid in self.followings]
        test_id = [tid for tid in test_id if tid in self.followings]

        self.labels = pickle.load(open(os.path.join(dir_path, 'labels.pkl'), 'rb'))

        if mode == 'train':
            self.orders = train_id
        elif mode =='val':
            val_id = [vid for vid in val_id if len(self.followings[vid]) == 4]
            self.orders = val_id
        elif mode == 'test':
            test_id = [vid for vid in test_id if len(self.followings[vid]) == 4]
            self.orders = test_id[:1900]
        else:
            raise ValueError
        print("Total number of clips {}".format(len(self.orders)))

        annotations = json.load(open(os.path.join(self.dir_path, 'flintstones_annotations_v1-0.json')))
        self.descriptions = {}
        for sample in annotations:
            self.descriptions[sample["globalID"]] = sample["description"]

        self.embeds = np.load(os.path.join(self.dir_path, "flintstones_use_embeddings.npy"))
        self.sent2idx = pickle.load(open(os.path.join(self.dir_path, 'flintstones_use_embed_idxs.pkl'), 'rb'))

        if mode == 'train':
            if transform:
                self.transform = transform
            else:
                self.transform = transforms.Compose([
                    transforms.RandomResizedCrop(im_input_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        else:
            if transform:
                self.transform = transform
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(im_input_size),
                    transforms.CenterCrop(im_input_size),
                    transforms.ToTensor(),
                    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

        self.tokenizer = tokenizer
        self.return_labels = return_labels
        self.out_img_folder = out_img_folder

    def __getitem__(self, item):

        # single image input
        globalIDs = [self.orders[item]] + self.followings[self.orders[item]]
        tokens = []
        images = []
        for idx, globalID in enumerate(globalIDs):
            if self.out_img_folder and idx != 0:
                image = Image.open(os.path.join(self.out_img_folder, 'gen_sample_%s_%s.png' % (item, idx-1))).convert('RGB')
            else:
                path = os.path.join(self.dir_path, 'video_frames_sampled', globalID + '.npy')
                arr = np.load(path)
                n_frames = arr.shape[0]
                random_range = randrange(n_frames)
                im = arr[random_range]
                # image = np.array(im)
                image = PIL.Image.fromarray(im.astype('uint8'), 'RGB')
            images.append(image)
            text = self.descriptions[globalID]
            if idx != 0:
                if self.tokenizer is not None:
                    tokens.append(self.tokenizer.encode(text.lower()))
                else:
                    tokens.append(text)
        if self.tokenizer is not None:
            tokens = torch.stack([torch.LongTensor(token.ids) for token in tokens])

        sent_embeds = [torch.tensor(self.embeds[self.sent2idx[globalID]]) for globalID in globalIDs[1:]]

        if self.return_labels:
            labels = [torch.tensor(self.labels[globalID]) for globalID in globalIDs[1:]]
            return torch.stack([self.transform(im) for im in images[1:]]), torch.stack(labels), tokens, self.transform(
                images[0]), torch.stack(sent_embeds)
        else:
            return torch.stack([self.transform(im) for im in images[1:]]), tokens, self.transform(images[0]), torch.stack(sent_embeds)

    def __len__(self):
        return len(self.orders)


# if __name__ == "__main__":
#
#     dataset = StoryImageDataset('/nas-ssd/adyasha/datasets/flintstones', None, None, 'val')
#     for item in range(len(dataset)):
#         texts = []
#         globalIDs = [dataset.orders[item]] + dataset.followings[dataset.orders[item]]
#         for idx, globalID in enumerate(globalIDs):
#             text = dataset.descriptions[globalID]
#             texts.append(text)
#         if len(texts) != 5:
#             print(item, globalIDs)

