import os, json
import random
from tqdm import tqdm
import torch.utils.data
from PIL import Image
from collections import defaultdict

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, tokenizer, preprocess, mode='train'):

        self.data_dir = data_dir
        self.tokenizer = tokenizer
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

        self.id2name = {}
        for ann in tqdm(annotations['images'], desc='Reading image filenames'):
            self.id2name[ann['id']] = ann['file_name']
        self.ids = list(set(self.id2name.keys()))

        self.id2captions = defaultdict(lambda: [])
        for cap in tqdm(annotations['annotations'], desc='Reading captions'):
            self.id2captions[cap['image_id']].append(cap['caption'])
        keys = list(set(self.id2captions.keys()))
        # assert all([img_id in keys for img_id in keys]), "Key not found in annotations"
        # assert len(keys) == len(self.ids), (len(keys), len(self.ids))

        self.preprocess = preprocess

    def __getitem__(self, item):

        img_id = self.ids[item]
        img_path = os.path.join(self.img_dir, self.id2name[img_id])
        caption = random.sample(self.id2captions[img_id], k=1)[0]
        # print(caption)
        tokens = self.tokenizer.encode(caption)
        tokens = torch.LongTensor(tokens.ids)
        image = self.preprocess(Image.open(img_path).convert('RGB'))
        return image, tokens

    def __len__(self):
        return len(self.ids)