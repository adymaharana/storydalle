import os, json, pickle
from tqdm import tqdm
import numpy as np
import torch.utils.data
from torchvision import transforms
from collections import Counter
import nltk
from PIL import Image

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, img_folder, tokenizer, preprocess, mode='train'):

        self.lengths = []
        self.followings = []
        self.dir_path = img_folder
        if mode == 'train':
            self.file_path = os.path.join(img_folder, 'train.json')
        elif mode == 'val':
            self.file_path = os.path.join(img_folder, 'val.json')
        else:
            self.file_path = os.path.join(img_folder, 'test.json')

        min_len = 4
        self.total_frames = 0
        self.images = []
        if os.path.exists(os.path.join(img_folder, 'img_cache' + str(min_len) + '_' + mode + '.npy')) and os.path.exists(
                os.path.join(img_folder, 'following_cache' + str(min_len) + '_' + mode + '.npy')):
            self.images = np.load(os.path.join(img_folder, 'img_cache' + str(min_len) + '_' + mode + '.npy'), encoding='latin1')
            self.followings = np.load(os.path.join(img_folder, 'following_cache' + str(min_len) + '_' + mode + '.npy'))
        else:
            print("Building image list cache")
            data = json.load(open(self.file_path))
            for ex in data:
                # Set the first image as the frame in the first description
                dont_use = False
                for tup in ex['desc_to_frame']:
                    if not os.path.exists(os.path.join('/'.join(list(os.path.abspath(img_folder).split('/'))[:-1]), tup[1])):
                        dont_use = True
                if dont_use:
                    continue
                self.images.append(ex['desc_to_frame'][0][1])
                # Set remaining images to the rest of the images
                following_imgs = [tup[1] for tup in ex['desc_to_frame'][1:]]
                self.followings.append(following_imgs)
            np.save(os.path.join(img_folder, 'img_cache' + str(min_len) + '_' + mode + '.npy'), self.images)
            np.save(os.path.join(img_folder, 'following_cache' + str(min_len) + '_' + mode + '.npy'), self.followings)
        print("Total number of clips {}".format(len(self.images)))

        self.descriptions_original = {tup[1]: tup[0] for ex in json.load(open(self.file_path)) for tup in ex['desc_to_frame']}
        self.preprocess = preprocess
        self.tokenizer = tokenizer

    def __getitem__(self, item):

        src_img_path = self.images[item]

        src_image_raw = Image.open(os.path.join(os.path.dirname(self.dir_path), src_img_path)).convert('RGB')
        src_image = self.preprocess(src_image_raw)
        # open the target image and caption
        caption = self.descriptions_original[src_img_path]
        tokens = self.tokenizer.encode(caption)
        tokens = torch.LongTensor(tokens.ids)

        return src_image, tokens

    def __len__(self):
        return len(self.images)


class StoryImageDataset(torch.utils.data.Dataset):
    def __init__(self, img_folder, tokenizer, transform=None, mode='train'):

        self.lengths = []
        self.followings = []
        self.dir_path = img_folder
        if mode == 'train':
            self.file_path = os.path.join(img_folder, 'train.json')
        elif mode == 'val':
            self.file_path = os.path.join(img_folder, 'val.json')
        else:
            self.file_path = os.path.join(img_folder, 'test.json')

        min_len = 2
        self.total_frames = 0
        self.images = []
        if os.path.exists(os.path.join(img_folder, 'img_cache' + str(min_len) + '_' + mode + '.npy')) and os.path.exists(
                os.path.join(img_folder, 'following_cache' + str(min_len) + '_' + mode + '.npy')):
            self.images = np.load(os.path.join(img_folder, 'img_cache' + str(min_len) + '_' + mode + '.npy'), encoding='latin1')
            self.followings = np.load(os.path.join(img_folder, 'following_cache' + str(min_len) + '_' + mode + '.npy'))
        else:
            print("Building image list cache")
            data = json.load(open(self.file_path))
            for ex in data:
                # Set the first image as the frame in the first description
                dont_use = False
                for tup in ex['desc_to_frame']:
                    if not os.path.exists(os.path.join('/'.join(list(os.path.abspath(img_folder).split('/'))[:-1]), tup[1])):
                        dont_use = True
                if dont_use:
                    continue
                if len(ex['desc_to_frame']) < min_len+1:
                    continue
                self.images.append(ex['desc_to_frame'][0][1])
                # Set remaining images to the rest of the images
                following_imgs = [tup[1] for tup in ex['desc_to_frame'][1:1+min_len]]
                self.followings.append(following_imgs)
            np.save(os.path.join(img_folder, 'img_cache' + str(min_len) + '_' + mode + '.npy'), self.images)
            np.save(os.path.join(img_folder, 'following_cache' + str(min_len) + '_' + mode + '.npy'), self.followings)
        print("Total number of clips {}".format(len(self.images)))
        self.descriptions_original = {tup[1]: tup[0] for ex in json.load(open(self.file_path)) for tup in ex['desc_to_frame']}

        if mode == 'train':
            if transform:
                self.transform = transform
            else:
                self.transform = transforms.Compose([
                    transforms.RandomResizedCrop(256),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        else:
            if transform:
                self.transform = transform
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(256),
                    transforms.ToTensor(),
                    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

        self.descriptions_vec = pickle.load(open(os.path.join(img_folder, 'didemo_use_embeds.pkl'), 'rb'))
        self.tokenizer = tokenizer


    def __getitem__(self, item):

        frame_path_list = [self.images[item]]
        for i in range(len(self.followings[item])):
            frame_path_list.append(str(self.followings[item][i]))

        images = []
        tokens = []

        for img_path in frame_path_list:
            im = self.transform(Image.open(os.path.join(os.path.dirname(os.path.normpath(self.dir_path)), img_path)))
            images.append(im)
            if self.tokenizer is not None:
                tokens.append(self.tokenizer.encode(self.descriptions_original[img_path]))
            else:
                tokens.append(self.descriptions_original[img_path])

        if self.tokenizer is not None:
            tokens = torch.stack([torch.LongTensor(token.ids) for token in tokens[1:]])

        sent_embeds = [torch.tensor(self.descriptions_vec[frame_path]) for frame_path in frame_path_list[1:]]
        return torch.stack(images[1:]), tokens, images[0], torch.stack(sent_embeds)


    def __len__(self):
        return len(self.images)


class CopyImageDataset(torch.utils.data.Dataset):
    def __init__(self, img_folder, tokenizer, preprocess, mode='train', video_len=2):

        self.lengths = []
        self.followings = []
        self.dir_path = img_folder
        if mode == 'train':
            self.file_path = os.path.join(img_folder, 'train.json')
        elif mode == 'val':
            self.file_path = os.path.join(img_folder, 'val.json')
        else:
            self.file_path = os.path.join(img_folder, 'test.json')
        self.video_len = video_len

        min_len = 4
        self.total_frames = 0
        self.images = []
        if os.path.exists(os.path.join(img_folder, 'img_cache' + str(min_len) + '_' + mode + '.npy')) and os.path.exists(
                os.path.join(img_folder, 'following_cache' + str(min_len) + '_' + mode + '.npy')):
            self.images = np.load(os.path.join(img_folder, 'img_cache' + str(min_len) + '_' + mode + '.npy'), encoding='latin1')
            self.followings = np.load(os.path.join(img_folder, 'following_cache' + str(min_len) + '_' + mode + '.npy'))
        else:
            print("Building image list cache")
            data = json.load(open(self.file_path))
            for ex in data:
                # Set the first image as the frame in the first description
                dont_use = False
                for tup in ex['desc_to_frame']:
                    if not os.path.exists(os.path.join('/'.join(list(os.path.abspath(img_folder).split('/'))[:-1]), tup[1])):
                        dont_use = True
                if dont_use:
                    continue
                self.images.append(ex['desc_to_frame'][0][1])
                # Set remaining images to the rest of the images
                following_imgs = [tup[1] for tup in ex['desc_to_frame'][1:]]
                self.followings.append(following_imgs)
            np.save(os.path.join(img_folder, 'img_cache' + str(min_len) + '_' + mode + '.npy'), self.images)
            np.save(os.path.join(img_folder, 'following_cache' + str(min_len) + '_' + mode + '.npy'), self.followings)
        print("Total number of clips {}".format(len(self.images)))

        self.descriptions_original = {tup[1]: tup[0] for ex in json.load(open(self.file_path)) for tup in ex['desc_to_frame']}
        self.preprocess = preprocess
        self.tokenizer = tokenizer

    def __getitem__(self, item):

        src_img_path = self.images[item]
        src_image_raw = Image.open(os.path.join(os.path.dirname(self.dir_path), src_img_path)).convert('RGB')
        src_image = self.preprocess(src_image_raw)

        tgt_img_paths = [str(self.followings[item][i]) for i in range(self.video_len)]
        # open the target image and caption
        tgt_images = [self.preprocess(Image.open(os.path.join(os.path.dirname(self.dir_path), tgt_img_path)).convert('RGB')) for tgt_img_path in tgt_img_paths]

        captions = [self.descriptions_original[tgt_img_path] for tgt_img_path in tgt_img_paths]
        tokens = [self.tokenizer.encode(caption) for caption in captions]
        tokens = [torch.LongTensor(token.ids) for token in tokens]

        return torch.stack(tgt_images), torch.stack(tokens), src_image

    def __len__(self):
        return len(self.images)


class CopyStoryDataset(torch.utils.data.Dataset):
    def __init__(self, img_folder, preprocess, mode='train', max_t_len=72, video_len=5, resnet=False, condition_seq_len=0):

        self.lengths = []
        self.followings = []
        self.video_len = video_len
        min_len = video_len - 1

        if mode == 'train':
            self.file_path = os.path.join(img_folder, 'train.json')
        elif mode == 'val':
            self.file_path = os.path.join(img_folder, 'val.json')
        else:
            self.file_path = os.path.join(img_folder, 'test.json')

        self.dir_path = img_folder

        if os.path.exists(os.path.join(self.dir_path, 'dalle_vocab.pkl')):
            vocab_from_file = True
            vocab_file = os.path.join(self.dir_path, 'dalle_vocab.pkl')
        else:
            vocab_from_file = False
            vocab_file = os.path.join(self.dir_path, 'dalle_vocab.pkl')

        self.vocab = Vocabulary(vocab_threshold=5,
                                vocab_file=vocab_file,
                                annotations_file=os.path.join(self.dir_path, 'train.json'),
                                vocab_from_file=vocab_from_file)

        print("Length of Vocabulary ", len(self.vocab))

        self.descriptions_original = {tup[1]: tup[0] for ex in json.load(open(self.file_path)) for tup in ex['desc_to_frame']}

        self.total_frames = 0
        self.images = []
        if os.path.exists(os.path.join(img_folder, 'img_cache' + str(min_len) + '_' + mode + '.npy')) and os.path.exists(
                os.path.join(img_folder, 'following_cache' + str(min_len) + '_' + mode + '.npy')):
            self.images = np.load(os.path.join(img_folder, 'img_cache' + str(min_len) + '_' + mode + '.npy'), encoding='latin1')
            self.followings = np.load(os.path.join(img_folder, 'following_cache' + str(min_len) + '_' + mode + '.npy'))
        else:
            print("Building image list cache")
            data = json.load(open(self.file_path))
            for ex in data:
                # Set the first image as the frame in the first description
                dont_use = False
                for tup in ex['desc_to_frame']:
                    if not os.path.exists(os.path.join('/'.join(list(os.path.abspath(img_folder).split('/'))[:-1]), tup[1])):
                        dont_use = True
                if dont_use:
                    continue
                self.images.append(ex['desc_to_frame'][0][1])
                # Set remaining images to the rest of the images
                following_imgs = [tup[1] for tup in ex['desc_to_frame'][1:]]
                self.followings.append(following_imgs)
            np.save(os.path.join(img_folder, 'img_cache' + str(min_len) + '_' + mode + '.npy'), self.images)
            np.save(os.path.join(img_folder, 'following_cache' + str(min_len) + '_' + mode + '.npy'), self.followings)
        print("Total number of clips {}".format(len(self.images)))

        self.preprocess = preprocess
        self.max_t_len = max_t_len

        self.resnet = resnet
        im_input_size = 299
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((im_input_size, im_input_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((im_input_size, im_input_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        self.condition_seq_len = condition_seq_len

    def __getitem__(self, item):

        # source image
        src_img_path = self.images[item].replace('didemo/', '')
        src_image_raw = Image.open(os.path.join(self.dir_path, src_img_path)).convert('RGB')
        # open the source images
        if self.resnet:
            src_image = self.transform(src_image_raw)
        else:
            src_image = self.preprocess(src_image_raw)

        # source caption
        src_caption = self.descriptions_original['didemo/' + src_img_path]
        src_text_tokens, src_text_mask = self.vocab._tokenize_pad_sentence(str(src_caption).lower(), self.max_t_len,
                                                                   condition=self.condition_seq_len)

        tgt_images = []
        tgt_text_tokens = [src_text_tokens]
        tgt_text_masks = [src_text_mask]
        for i in range(0, self.video_len-1):
            tgt_img_path = str(self.followings[item][i]).replace('didemo/', '')
            # open the target image and caption
            caption = self.descriptions_original['didemo/' + tgt_img_path]
            tgt_image = self.preprocess(Image.open(os.path.join(self.dir_path, tgt_img_path)).convert('RGB'))
            tgt_images.append(tgt_image.unsqueeze(0))
            # image = Image.open(os.path.join(self.out_dir, 'img-' + str(item) + '.png')).convert('RGB')
            text_tokens, text_mask = self.vocab._tokenize_pad_sentence(str(caption).lower(), self.max_t_len, condition=self.condition_seq_len)
            tgt_text_tokens.append(text_tokens)
            tgt_text_masks.append(text_mask)

        return torch.cat(tgt_images, dim=0), torch.LongTensor(tgt_text_tokens), torch.LongTensor(tgt_text_masks), src_image

    def __len__(self):
        return len(self.images)


if __name__ == "__main__":

    dataset = StoryImageDataset('/nas-ssd/adyasha/datasets/didemo', None, None, 'val')

    all_captions = {}
    for item in range(len(dataset)):

        frame_path_list = [dataset.images[item]]
        for i in range(len(dataset.followings[item])):
            frame_path_list.append(str(dataset.followings[item][i]))
        captions = [dataset.descriptions_original[img_path] for img_path in frame_path_list]
        all_captions[item] = captions

    with open(os.path.join('/nas-ssd/adyasha/datasets/didemo', 'all_captions_val.json'), 'w') as f:
        json.dump(all_captions, f, indent=4)