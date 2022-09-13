import os, re
import csv
import nltk
import pickle
from collections import Counter
import numpy as np
from tqdm import tqdm
import torch.utils.data
from torchvision.datasets import ImageFolder
from PIL import Image
from torchvision import transforms
import json
import pickle as pkl

class Vocabulary(object):

    def __init__(self,
                 vocab_threshold,
                 vocab_file,
                 annotations_file,
                 vocab_from_file=False,
                 unk_word="[UNK]",
                 pad_word="[PAD]",
                 condition_word="[CON]"):
        """Initialize the vocabulary.
        Args:
          vocab_threshold: Minimum word count threshold.
          vocab_file: File containing the vocabulary.
          start_word: Special word denoting sentence start.
          end_word: Special word denoting sentence end.
          unk_word: Special word denoting unknown words.
          annotations_file: Path for train annotation file.
          vocab_from_file: If False, create vocab from scratch & override any existing vocab_file
                           If True, load vocab from from existing vocab_file, if it exists
        """
        self.vocab_threshold = vocab_threshold
        self.vocab_file = vocab_file
        self.unk_word = unk_word
        self.pad_word = pad_word
        self.condition_word = condition_word
        self.annotations_file = annotations_file
        self.vocab_from_file = vocab_from_file
        self.get_vocab()

    def get_vocab(self):
        """Load the vocabulary from file OR build the vocabulary from scratch."""
        if os.path.exists(self.vocab_file) & self.vocab_from_file:
            print('Reading vocabulary from %s file!' % self.vocab_file)
            with open(self.vocab_file, 'rb') as f:
                vocab = pickle.load(f)
                self.word2idx = vocab['word2idx']
                self.idx2word = vocab['idx2word']
            print('Vocabulary successfully loaded from %s file!' % self.vocab_file)
        else:
            print("Building vocabulary from scratch")
            self.build_vocab()
            with open(self.vocab_file, 'wb') as f:
                pickle.dump({'word2idx': self.word2idx, 'idx2word': self.idx2word}, f)

    def build_vocab(self):
        """Populate the dictionaries for converting tokens to integers (and vice-versa)."""
        self.init_vocab()
        self.add_word(self.pad_word)
        self.add_word(self.unk_word)
        self.add_word(self.condition_word)
        self.add_captions()

    def load_glove(self, filename):
        """ returns { word (str) : vector_embedding (torch.FloatTensor) }
        """
        glove = {}
        with open(filename) as f:
            for line in tqdm(f.readlines()):
                values = line.strip("\n").split(" ")  # space separator
                word = values[0]
                vector = np.asarray([float(e) for e in values[1:]])
                glove[word] = vector
        return glove

    def extract_glove(self, raw_glove_path, vocab_glove_path, glove_dim=300, extra_vocab_len=0):

        if os.path.exists(vocab_glove_path):
            print("Pre-extracted embedding matrix exists at %s" % vocab_glove_path)
        else:
            # Make glove embedding.
            print("Loading glove embedding at path : {}.\n".format(raw_glove_path))
            glove_full = self.load_glove(raw_glove_path)
            print("Glove Loaded, building word2idx, idx2word mapping.\n")
            idx2word = {v: k for k, v in self.word2idx.items()}

            glove_matrix = np.zeros([len(self.word2idx), glove_dim])
            glove_keys = glove_full.keys()
            for i in tqdm(range(len(idx2word))):
                w = idx2word[i]
                w_embed = glove_full[w] if w in glove_keys else np.random.randn(glove_dim) * 0.4
                glove_matrix[i, :] = w_embed
            print("vocab embedding size is :", glove_matrix.shape)

            if extra_vocab_len:
                position_matrix = np.random.normal(0.0, 1.0, (extra_vocab_len, glove_dim))
                glove_matrix = np.concatenate((glove_matrix, position_matrix), axis=0)
                print("with additional vocab, vocab embedding size is :", glove_matrix.shape)

            torch.save(glove_matrix, vocab_glove_path)

    def init_vocab(self):
        """Initialize the dictionaries for converting tokens to integers (and vice-versa)."""
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        """Add a token to the vocabulary."""
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def add_captions(self):
        """Loop over training captions and add all tokens to the vocabulary that meet or exceed the threshold."""
        counter = Counter()
        with open(self.annotations_file, 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            print("Tokenizing captions")
            for i, row in tqdm(enumerate(csv_reader)):
                _, _, caption = row
                tokens = nltk.tokenize.word_tokenize(caption.lower())
                counter.update(tokens)

        words = [word for word, cnt in counter.items() if cnt >= self.vocab_threshold]

        for i, word in enumerate(words):
            self.add_word(word)

    def _tokenize_pad_sentence(self, sentence, max_t_len=None, condition=0):
        """[BOS], [WORD1], [WORD2], ..., [WORDN], [EOS], [PAD], ..., [PAD], len == max_t_len
        All non-PAD values are valid, with a mask value of 1
        """
        if condition:
            sentence_tokens = [self.condition_word]*condition + \
                              nltk.tokenize.word_tokenize(sentence.lower())[:max_t_len-condition]
        else:
            sentence_tokens = nltk.tokenize.word_tokenize(sentence.lower())[:max_t_len]

        # pad
        valid_l = len(sentence_tokens)
        # print(sentence_tokens)
        if max_t_len:
            mask = [1] * valid_l + [0] * (max_t_len - valid_l)
            sentence_tokens += [self.pad_word] * (max_t_len - valid_l)
            text_idxs = [self.word2idx.get(t, self.word2idx[self.unk_word]) for t in sentence_tokens]
            return text_idxs, mask
        else:
            text_idxs = [self.word2idx.get(t, self.word2idx[self.unk_word]) for t in sentence_tokens]
            return text_idxs, None

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx[self.unk_word]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, img_folder, tokenizer, preprocess, mode='train'):
        self.lengths = []
        self.followings = []
        self.img_dataset = ImageFolder(img_folder)
        self.img_folder = img_folder
        self.labels = np.load(os.path.join(img_folder, 'labels.npy'), allow_pickle=True, encoding='latin1').item()
        self.descriptions_original = np.load(os.path.join(img_folder, 'descriptions.npy'), allow_pickle=True, encoding='latin1').item()
        self.tokenizer = tokenizer

        # train_ids, val_ids, test_ids = np.load(os.path.join(img_folder, 'train_val_test_ids.npy'), allow_pickle=True)
        train_ids, val_ids, test_ids = np.load(os.path.join(img_folder, 'train_seen_unseen_ids.npy'), allow_pickle=True)
        if mode == 'train':
            self.ids = np.sort(train_ids)
        elif mode =='val':
            self.ids = np.sort(val_ids)
        elif mode =='test':
            self.ids = np.sort(test_ids)
        else:
            raise ValueError

        self.preprocess = preprocess

    def sample_image(self, im):
        shorter, longer = min(im.size[0], im.size[1]), max(im.size[0], im.size[1])
        video_len = int(longer/shorter)
        se = np.random.randint(0,video_len, 1)[0]
        return im.crop((0, se * shorter, shorter, (se+1)*shorter))

    def __getitem__(self, item):

        img_id = self.ids[item]
        img_path = self.img_dataset.imgs[img_id][0]
        im_id = str(img_path).replace(self.img_folder, '').replace('.png', '')[1:]
        caption = self.descriptions_original[im_id][0]
        tokens = self.tokenizer.encode(caption.lower())
        tokens = torch.LongTensor(tokens.ids)
        image = self.preprocess(self.sample_image(Image.open(img_path).convert('RGB')))
        return image, tokens

    def __len__(self):
        return len(self.ids)


class ConditionalImageDataset(torch.utils.data.Dataset):
    def __init__(self, img_folder, preprocess, mode='train', max_t_len=72, video_len=5):
        self.lengths = []
        self.followings = []
        self.images = []
        self.img_dataset = ImageFolder(img_folder)
        self.img_folder = img_folder
        self.labels = np.load(os.path.join(img_folder, 'labels.npy'), allow_pickle=True, encoding='latin1').item()
        self.video_len = video_len
        self.descriptions_original = np.load(os.path.join(img_folder, 'descriptions.npy'), allow_pickle=True, encoding='latin1').item()
        self.vocab = Vocabulary(vocab_threshold=2,
                                vocab_file='../pororo_png/dalle_vocab.pkl',
                                annotations_file=os.path.join(img_folder, 'descriptions.csv'),
                                vocab_from_file=True)
        print("Length of Vocabulary ", len(self.vocab))
        self.descriptions = np.load(os.path.join(img_folder, 'descriptions_vec.npy'), allow_pickle=True, encoding='latin1').item()

        if os.path.exists(os.path.join(img_folder, 'img_cache4.npy')) and os.path.exists(os.path.join(img_folder, 'following_cache4.npy')):
            self.images = np.load(os.path.join(img_folder, 'img_cache4.npy'), encoding='latin1')
            self.followings = np.load(os.path.join(img_folder, 'following_cache4.npy'))
            self.counter = ''
        else:
            counter = np.load(os.path.join(img_folder, 'frames_counter.npy'), allow_pickle=True).item()
            for idx, (im, _) in enumerate(tqdm(self.img_dataset, desc="Counting total number of frames")):
                img_path, _ = self.img_dataset.imgs[idx]
                v_name = img_path.replace(self.img_folder,'')
                id = v_name.split('/')[-1]
                id = int(id.replace('.png', ''))
                v_name = re.sub(r"[0-9]+.png",'', v_name)
                if id > counter[v_name] - (self.video_len-1):
                    continue
                following_imgs = []
                for i in range(self.video_len-1):
                    following_imgs.append(v_name + str(id+i+1) + '.png')
                self.images.append(img_path.replace(self.img_folder, ''))
                self.followings.append(following_imgs)
            np.save(os.path.join(self.img_folder, 'img_cache4.npy'), self.images)
            np.save(os.path.join(self.img_folder, 'following_cache4.npy'), self.followings)

        train_ids, val_ids, test_ids = np.load(os.path.join(img_folder, 'train_seen_unseen_ids.npy'), allow_pickle=True)
        if mode == 'train':
            self.ids = np.sort(train_ids)
        else:
            self.ids = np.sort(val_ids)

        self.preprocess = preprocess
        self.max_t_len = max_t_len

    def sample_image(self, im):
        shorter, longer = min(im.size[0], im.size[1]), max(im.size[0], im.size[1])
        video_len = int(longer/shorter)
        se = np.random.randint(0,video_len, 1)[0]
        return im.crop((0, se * shorter, shorter, (se+1)*shorter))

    def __getitem__(self, item):

        img_id = self.ids[item]
        img_path = self.img_dataset.imgs[img_id][0]
        im_id = str(img_path).replace(self.img_folder, '').replace('.png', '')
        caption = self.descriptions_original[im_id][0]
        image = self.preprocess(self.sample_image(Image.open(img_path).convert('RGB')))
        # image = Image.open(os.path.join(self.out_dir, 'img-' + str(item) + '.png')).convert('RGB')
        text_tokens, text_mask = self.vocab._tokenize_pad_sentence(str(caption).lower(), self.max_t_len, condition=True)

        im_ids = [str(self.images[img_id])[2:-1].replace(self.img_folder, '').replace('.png', '')] + \
                 [str(self.followings[img_id][k])[2:-1].replace(self.img_folder, '').replace('.png', '') for k in
                                                        range(0, self.video_len - 1)]
        sentence_embeddings = np.concatenate([self.descriptions[id][0] for id in im_ids], axis=-1)
        return image, torch.LongTensor(text_tokens), torch.Tensor(sentence_embeddings)

        # img_id = self.ids[item]
        # img_paths = [str(self.images[img_id])[2:-1]] + [str(self.followings[img_id][k])[2:-1] for k in range(0, self.video_len-1)]
        # captions = [self.descriptions_original[str(img_path).replace(self.img_folder, '').replace('.png', '')][0] for img_path in img_paths]
        # images = [self.preprocess(self.sample_image(Image.open(os.path.join(self.img_folder, path)).convert('RGB'))) for path in img_paths]
        # text_tokens = []
        # for c in captions:
        #     text_tokens.extend(self.vocab._tokenize_pad_sentence(str(c).strip().lower())[0])
        # text_tokens = text_tokens[:self.max_t_len]
        # valid_l = len(text_tokens)
        # text_tokens += [self.vocab.word2idx.get(self.vocab.pad_word)] * (self.max_t_len - valid_l)
        # return torch.cat(images, dim=-1), torch.LongTensor(text_tokens)

    def __len__(self):
        return len(self.ids)


class CopyImageDataset(torch.utils.data.Dataset):
    def __init__(self, img_folder, tokenizer, preprocess, mode='train', video_len=4):
        self.lengths = []
        self.followings = []
        self.images = []
        self.img_dataset = ImageFolder(img_folder)
        self.img_folder = img_folder
        self.labels = np.load(os.path.join(img_folder, 'labels.npy'), allow_pickle=True, encoding='latin1').item()
        self.video_len = video_len
        self.descriptions_original = np.load(os.path.join(img_folder, 'descriptions.npy'), allow_pickle=True, encoding='latin1').item()

        if os.path.exists(os.path.join(img_folder, 'img_cache4.npy')) and os.path.exists(os.path.join(img_folder, 'following_cache4.npy')):
            self.images = np.load(os.path.join(img_folder, 'img_cache4.npy'), encoding='latin1')
            self.followings = np.load(os.path.join(img_folder, 'following_cache4.npy'))
            self.counter = ''
        else:
            counter = np.load(os.path.join(img_folder, 'frames_counter.npy'), allow_pickle=True).item()
            for idx, (im, _) in enumerate(tqdm(self.img_dataset, desc="Counting total number of frames")):
                img_path, _ = self.img_dataset.imgs[idx]
                v_name = img_path.replace(self.img_folder,'')
                id = v_name.split('/')[-1]
                id = int(id.replace('.png', ''))
                v_name = re.sub(r"[0-9]+.png",'', v_name)
                if id > counter[v_name] - (self.video_len-1):
                    continue
                following_imgs = []
                for i in range(self.video_len-1):
                    following_imgs.append(v_name + str(id+i+1) + '.png')
                self.images.append(img_path.replace(self.img_folder, ''))
                self.followings.append(following_imgs)
            np.save(os.path.join(self.img_folder, 'img_cache4.npy'), self.images)
            np.save(os.path.join(self.img_folder, 'following_cache4.npy'), self.followings)

        train_ids, val_ids, test_ids = np.load(os.path.join(img_folder, 'train_seen_unseen_ids.npy'), allow_pickle=True)
        if mode == 'train':
            self.ids = np.sort(train_ids)
        elif mode =='val':
            self.ids = np.sort(val_ids)
        elif mode =='test':
            self.ids = np.sort(test_ids)
        else:
            raise ValueError

        self.preprocess = preprocess
        self.tokenizer = tokenizer

    def sample_image(self, im):
        shorter, longer = min(im.size[0], im.size[1]), max(im.size[0], im.size[1])
        video_len = int(longer/shorter)
        se = np.random.randint(0,video_len, 1)[0]
        return im.crop((0, se * shorter, shorter, (se+1)*shorter))

    def __getitem__(self, item):

        src_img_id = self.ids[item]
        src_img_path = os.path.join(self.img_folder, str(self.images[src_img_id])[2:-1])
        tgt_img_paths = [str(self.followings[src_img_id][i])[2:-1] for i in range(self.video_len)]
        # print(src_img_path, tgt_img_path)

        # open the source images
        src_image = self.preprocess(self.sample_image(Image.open(src_img_path).convert('RGB')))

        # open the target image and caption
        tgt_img_ids = [str(tgt_img_path).replace(self.img_folder, '').replace('.png', '') for tgt_img_path in tgt_img_paths]
        tgt_images = [self.preprocess(self.sample_image(Image.open(os.path.join(self.img_folder, tgt_img_path)).convert('RGB'))) for tgt_img_path in tgt_img_paths]
        # image = Image.open(os.path.join(self.out_dir, 'img-' + str(item) + '.png')).convert('RGB')

        captions = [self.descriptions_original[tgt_img_id][0] for tgt_img_id in tgt_img_ids]
        tokens = [self.tokenizer.encode(caption.lower()) for caption in captions]
        tokens = [torch.LongTensor(token.ids) for token in tokens]

        return torch.stack(tgt_images), torch.stack(tokens), src_image

    def __len__(self):
        return len(self.ids)


class StoryImageDataset(torch.utils.data.Dataset):
    def __init__(self, img_folder, tokenizer, preprocess, mode='train', video_len=4, out_img_folder='', return_labels=False):
        self.lengths = []
        self.followings = []
        self.images = []
        self.img_dataset = ImageFolder(img_folder)
        self.img_folder = img_folder
        self.labels = np.load(os.path.join(img_folder, 'labels.npy'), allow_pickle=True, encoding='latin1').item()
        self.video_len = video_len
        self.descriptions_original = np.load(os.path.join(img_folder, 'descriptions.npy'), allow_pickle=True, encoding='latin1').item()
        # self.descriptions = np.load(os.path.join(img_folder, 'descriptions_vec.npy'), allow_pickle=True, encoding='latin1').item() # used in the eccv camera-ready version
        self.descriptions = pkl.load(open(os.path.join(img_folder, 'descriptions_vec_512.pkl'), 'rb'))

        if os.path.exists(os.path.join(img_folder, 'img_cache4.npy')) and os.path.exists(os.path.join(img_folder, 'following_cache4.npy')):
            self.images = np.load(os.path.join(img_folder, 'img_cache4.npy'), encoding='latin1')
            self.followings = np.load(os.path.join(img_folder, 'following_cache4.npy'))
            self.counter = ''
        else:
            counter = np.load(os.path.join(img_folder, 'frames_counter.npy'), allow_pickle=True).item()
            for idx, (im, _) in enumerate(tqdm(self.img_dataset, desc="Counting total number of frames")):
                img_path, _ = self.img_dataset.imgs[idx]
                v_name = img_path.replace(self.img_folder,'')
                id = v_name.split('/')[-1]
                id = int(id.replace('.png', ''))
                v_name = re.sub(r"[0-9]+.png",'', v_name)
                if id > counter[v_name] - (self.video_len-1):
                    continue
                following_imgs = []
                for i in range(self.video_len-1):
                    following_imgs.append(v_name + str(id+i+1) + '.png')
                self.images.append(img_path.replace(self.img_folder, ''))
                self.followings.append(following_imgs)
            np.save(os.path.join(self.img_folder, 'img_cache4.npy'), self.images)
            np.save(os.path.join(self.img_folder, 'following_cache4.npy'), self.followings)

        train_ids, val_ids, test_ids = np.load(os.path.join(img_folder, 'train_seen_unseen_ids.npy'), allow_pickle=True)
        if mode == 'train':
            self.ids = np.sort(train_ids)
        elif mode =='val':
            self.ids = np.sort(val_ids)
        elif mode =='test':
            self.ids = np.sort(test_ids)
        else:
            raise ValueError

        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.return_labels = return_labels
        self.out_img_folder = out_img_folder

    def sample_image(self, im):
        shorter, longer = min(im.size[0], im.size[1]), max(im.size[0], im.size[1])
        video_len = int(longer/shorter)
        se = np.random.randint(0,video_len, 1)[0]
        return im.crop((0, se * shorter, shorter, (se+1)*shorter))

    def __getitem__(self, item):

        src_img_id = self.ids[item]


        src_img_path = os.path.join(self.img_folder, str(self.images[src_img_id])[2:-1])
        tgt_img_paths = [str(self.followings[src_img_id][i])[2:-1] for i in range(self.video_len)]
        # print(src_img_path, tgt_img_path)

        # open the source images
        src_image = self.preprocess(self.sample_image(Image.open(src_img_path).convert('RGB')))

        # open the target image and caption
        tgt_img_ids = [str(tgt_img_path).replace(self.img_folder, '').replace('.png', '') for tgt_img_path in tgt_img_paths]

        if self.out_img_folder:
            tgt_images = [self.preprocess(Image.open(os.path.join(self.out_img_folder, 'gen_sample_%s_%s.png' % (item, frame_idx))).convert('RGB')) for frame_idx in range(self.video_len)]
        else:
            tgt_images = [self.preprocess(self.sample_image(Image.open(os.path.join(self.img_folder, tgt_img_path)).convert('RGB'))) for tgt_img_path in tgt_img_paths]
        # image = Image.open(os.path.join(self.out_dir, 'img-' + str(item) + '.png')).convert('RGB')

        captions = [self.descriptions_original[tgt_img_id][0] for tgt_img_id in tgt_img_ids]
        if self.tokenizer is not None:
            tokens = [self.tokenizer.encode(caption.lower()) for caption in captions]
            tokens = torch.stack([torch.LongTensor(token.ids) for token in tokens])
        else:
            tokens = captions

        sentence_embeddings = [torch.tensor(self.descriptions[tgt_img_id][0]) for tgt_img_id in tgt_img_ids]

        if self.return_labels:
            labels = [torch.tensor(self.labels[img_id]) for img_id in tgt_img_ids]
            return torch.stack(tgt_images), torch.stack(labels), tokens, src_image, torch.stack(sentence_embeddings)
        else:
            return torch.stack(tgt_images), tokens, src_image, torch.stack(sentence_embeddings)

    def __len__(self):
        return len(self.ids)


# class StoryImageDataset(torch.utils.data.Dataset):
#     def __init__(self, img_folder, im_input_size,
#                  out_img_folder = '',
#                  mode='train',
#                  video_len = 5,
#                  transform=None):
#         self.lengths = []
#         self.followings = []
#         self.images = []
#         self.img_dataset = ImageFolder(img_folder)
#         self.img_folder = img_folder
#         self.labels = np.load(os.path.join(img_folder, 'labels.npy'), allow_pickle=True, encoding='latin1').item()
#         self.video_len = video_len
#         self.descriptions_original = np.load(os.path.join(img_folder, 'descriptions.npy'), allow_pickle=True, encoding='latin1').item()
#
#         if os.path.exists(os.path.join(img_folder, 'img_cache4.npy')) and os.path.exists(os.path.join(img_folder, 'following_cache4.npy')):
#             self.images = np.load(os.path.join(img_folder, 'img_cache4.npy'), encoding='latin1')
#             self.followings = np.load(os.path.join(img_folder, 'following_cache4.npy'))
#             self.counter = ''
#         else:
#             for idx, (im, _) in enumerate(tqdm(self.img_dataset, desc="Counting total number of frames")):
#                 img_path, _ = self.img_dataset.imgs[idx]
#                 v_name = img_path.replace(self.img_folder,'')
#                 id = v_name.split('/')[-1]
#                 id = int(id.replace('.png', ''))
#                 v_name = re.sub(r"[0-9]+.png",'', v_name)
#                 if id > self.counter[v_name] - (self.video_len-1):
#                     continue
#                 following_imgs = []
#                 for i in range(self.video_len-1):
#                     following_imgs.append(v_name + str(id+i+1) + '.png')
#                 self.images.append(img_path.replace(self.img_folder, ''))
#                 self.followings.append(following_imgs)
#             np.save(os.path.join(self.img_folder, 'img_cache4.npy'), self.images)
#             np.save(os.path.join(self.img_folder, 'following_cache4.npy'), self.followings)
#
#         # train_ids, val_ids, test_ids = np.load(os.path.join(img_folder, 'train_val_test_ids.npy'), allow_pickle=True)
#         train_ids, val_ids, test_ids = np.load(os.path.join(img_folder, 'train_seen_unseen_ids.npy'), allow_pickle=True)
#
#         if mode == 'train':
#             self.ids = train_ids
#             if transform:
#                 self.transform = transform
#             else:
#                 self.transform = transforms.Compose([
#                     transforms.RandomResizedCrop(im_input_size),
#                     transforms.RandomHorizontalFlip(),
#                     transforms.ToTensor(),
#                     # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#                 ])
#         else:
#             if mode == 'val':
#                 self.ids = val_ids[:2304]
#             elif mode == 'test':
#                 self.ids = test_ids
#             else:
#                 raise ValueError
#
#             if transform:
#                 self.transform = transform
#             else:
#                 self.transform = transforms.Compose([
#                     transforms.Resize(im_input_size),
#                     transforms.CenterCrop(im_input_size),
#                     transforms.ToTensor(),
#                     # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#                 ])
#
#         self.out_dir = out_img_folder
#
#     def sample_image(self, im):
#         shorter, longer = min(im.size[0], im.size[1]), max(im.size[0], im.size[1])
#         video_len = int(longer/shorter)
#         se = np.random.randint(0,video_len, 1)[0]
#         #print(se*shorter, shorter, (se+1)*shorter)
#         return im.crop((0, se * shorter, shorter, (se+1)*shorter))
#
#     def __getitem__(self, item):
#
#         img_id = self.ids[item]
#         img_paths = [str(self.images[img_id])[2:-1]] + [str(self.followings[img_id][k])[2:-1] for k in range(0, self.video_len-1)]
#         if self.out_dir:
#             images = [Image.open(os.path.join(self.out_dir, 'img-%s-%s.png' % (item, k))).convert('RGB') for k in range(self.video_len)]
#         else:
#             images = [self.sample_image(Image.open(os.path.join(self.img_folder, path)).convert('RGB')) for path in img_paths]
#         captions = [self.descriptions_original[path.replace('.png', '').replace(self.img_folder + '/', '')] for path in img_paths]
#         labels = [self.labels[path.replace('.png', '').replace(self.img_folder + '/', '')] for path in img_paths]
#         # return torch.cat([self.transform(image).unsqueeze(0) for image in images], dim=0), torch.tensor(np.vstack(labels))
#         return torch.stack([self.transform(im) for im in images[1:]]), captions[1:]
#
#     def __len__(self):
#         return len(self.ids)


class CopyStoryDataset(torch.utils.data.Dataset):
    def __init__(self, img_folder, preprocess, mode='train', max_t_len=72, video_len=5, resnet=False, condition_seq_len=0):
        self.lengths = []
        self.followings = []
        self.images = []
        self.img_dataset = ImageFolder(img_folder)
        self.img_folder = img_folder
        self.labels = np.load(os.path.join(img_folder, 'labels.npy'), allow_pickle=True, encoding='latin1').item()
        self.video_len = video_len
        self.descriptions_original = np.load(os.path.join(img_folder, 'descriptions.npy'), allow_pickle=True, encoding='latin1').item()
        self.vocab = Vocabulary(vocab_threshold=2,
                                vocab_file='../pororo_png/dalle_vocab.pkl',
                                annotations_file=os.path.join(img_folder, 'descriptions.csv'),
                                vocab_from_file=True)
        print("Length of Vocabulary ", len(self.vocab))
        self.descriptions = np.load(os.path.join(img_folder, 'descriptions_vec.npy'), allow_pickle=True, encoding='latin1').item()

        if os.path.exists(os.path.join(img_folder, 'img_cache4.npy')) and os.path.exists(os.path.join(img_folder, 'following_cache4.npy')):
            self.images = np.load(os.path.join(img_folder, 'img_cache4.npy'), encoding='latin1')
            self.followings = np.load(os.path.join(img_folder, 'following_cache4.npy'))
            self.counter = ''
        else:
            counter = np.load(os.path.join(img_folder, 'frames_counter.npy'), allow_pickle=True).item()
            for idx, (im, _) in enumerate(tqdm(self.img_dataset, desc="Counting total number of frames")):
                img_path, _ = self.img_dataset.imgs[idx]
                v_name = img_path.replace(self.img_folder,'')
                id = v_name.split('/')[-1]
                id = int(id.replace('.png', ''))
                v_name = re.sub(r"[0-9]+.png",'', v_name)
                if id > counter[v_name] - (self.video_len-1):
                    continue
                following_imgs = []
                for i in range(self.video_len-1):
                    following_imgs.append(v_name + str(id+i+1) + '.png')
                self.images.append(img_path.replace(self.img_folder, ''))
                self.followings.append(following_imgs)
            np.save(os.path.join(self.img_folder, 'img_cache4.npy'), self.images)
            np.save(os.path.join(self.img_folder, 'following_cache4.npy'), self.followings)

        train_ids, val_ids, test_ids = np.load(os.path.join(img_folder, 'train_seen_unseen_ids.npy'), allow_pickle=True)
        if mode == 'train':
            self.ids = np.sort(train_ids)
        else:
            self.ids = np.sort(val_ids)

        self.preprocess = preprocess
        self.max_t_len = max_t_len

        self.resnet = resnet
        im_input_size = 299
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(im_input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(im_input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        self.condition_seq_len = condition_seq_len

    def sample_image(self, im):
        shorter, longer = min(im.size[0], im.size[1]), max(im.size[0], im.size[1])
        video_len = int(longer/shorter)
        se = np.random.randint(0,video_len, 1)[0]
        return im.crop((0, se * shorter, shorter, (se+1)*shorter))

    def __getitem__(self, item):

        # source image
        src_img_id = self.ids[item]
        src_img_path = os.path.join(self.img_folder, str(self.images[src_img_id])[2:-1])

        # print(src_img_path, tgt_img_path)

        # open the source images
        if self.resnet:
            src_image = self.transform(self.sample_image(Image.open(src_img_path).convert('RGB')))
        else:
            src_image = self.preprocess(self.sample_image(Image.open(src_img_path).convert('RGB')))
        # source caption
        src_caption = self.descriptions_original[str(self.images[src_img_id])[2:-1].replace(self.img_folder, '').replace('.png', '')][0]
        src_text_tokens, src_text_mask = self.vocab._tokenize_pad_sentence(str(src_caption).lower(), self.max_t_len,
                                                                   condition=self.condition_seq_len)

        tgt_images = []
        tgt_text_tokens = [src_text_tokens]
        tgt_text_masks = [src_text_mask]
        for i in range(0, self.video_len-1):
            tgt_img_path = str(self.followings[src_img_id][i])[2:-1]
            # open the target image and caption
            tgt_img_id = str(tgt_img_path).replace(self.img_folder, '').replace('.png', '')
            caption = self.descriptions_original[tgt_img_id][0]
            tgt_image = self.preprocess(self.sample_image(Image.open(os.path.join(self.img_folder, tgt_img_path)).convert('RGB')))
            tgt_images.append(tgt_image.unsqueeze(0))
            # image = Image.open(os.path.join(self.out_dir, 'img-' + str(item) + '.png')).convert('RGB')
            text_tokens, text_mask = self.vocab._tokenize_pad_sentence(str(caption).lower(), self.max_t_len, condition=self.condition_seq_len)
            tgt_text_tokens.append(text_tokens)
            tgt_text_masks.append(text_mask)

        return torch.cat(tgt_images, dim=0), torch.LongTensor(tgt_text_tokens), torch.LongTensor(tgt_text_masks), src_image

    def __len__(self):
        return len(self.ids)

# if __name__ == "__main__":
#
#     dataset = StoryImageDataset('/nas-ssd/adyasha/datasets/pororo_png',
#                                 None, None, mode='val')
#
#     all_captions = {}
#     for item in range(len(dataset)):
#
#         src_img_id = dataset.ids[item]
#         tgt_img_paths = [str(dataset.followings[src_img_id][i])[2:-1] for i in range(dataset.video_len)]
#         tgt_img_ids = [str(tgt_img_path).replace(dataset.img_folder, '').replace('.png', '') for tgt_img_path in tgt_img_paths]
#         captions = [dataset.descriptions_original[tgt_img_id][0] for tgt_img_id in tgt_img_ids]
#         all_captions[item] = captions
#
#     with open(os.path.join('/nas-ssd/adyasha/datasets/pororo_png', 'all_captions_val.json'), 'w') as f:
#         json.dump(all_captions, f, indent=4)