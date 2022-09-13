import os, json
import pickle
import numpy as np
from torchvision.utils import save_image


def get_pororo_idxs():
    img_folder = '../StoryGAN/pororo_png/'
    pororo_key = 'Pororo_ENGLISH1_3_ep4/27'
    pororo_images = np.load(os.path.join(img_folder, 'img_cache4.npy'), encoding='latin1')

    _, val_ids, _ = np.load(os.path.join(img_folder, 'train_seen_unseen_ids.npy'), allow_pickle=True)
    for i, item in enumerate(val_ids):
        if pororo_key in str(pororo_images[item]):
            print(i, item, pororo_images[item])

def get_flintstones_idxs():
    key = 'superhero'

    splits = json.load(open(os.path.join(dir_path, 'train-val-test_split.json'), 'r'))
    _, val_id, test_id = splits["train"], splits["val"], splits["test"]
    followings = pickle.load(open(os.path.join(dir_path, 'following_cache' + str(min_len) + '.pkl'), 'rb'))
    val_id = [vid for vid in val_id if vid in followings]
    test_id = [tid for tid in test_id if tid in followings]
    val_id = [vid for vid in val_id if len(self.followings[vid]) == 4]
    test_id = [vid for vid in test_id if len(self.followings[vid]) == 4]
    annotations = json.load(open(os.path.join(dir_path, 'flintstones_annotations_v1-0.json')))
    descriptions = {}
    for sample in annotations:
        descriptions[sample["globalID"]] = sample["description"]

    for ids, name in zip([val_id, test_id], ['val', 'test']):
        for i, id in enumerate(ids):
            caption = descriptions[id]
            if key in caption:
                print(i, id, caption, name)

def get_didemo_idxs():
    pass


def get_pororo_texts():

    img_folder = '../StoryGAN/pororo_png/'
    pororo_key = 'Pororo_ENGLISH1_3_ep4/27'
    images = np.load(os.path.join(img_folder, 'img_cache4.npy'), encoding='latin1')
    _, val_ids, _ = np.load(os.path.join(img_folder, 'train_seen_unseen_ids.npy'), allow_pickle=True)
    val_ids = np.sort(val_ids)
    descriptions_original = np.load(os.path.join(img_folder, 'descriptions.npy'), allow_pickle=True,
                                    encoding='latin1').item()
    followings = np.load(os.path.join(img_folder, 'following_cache4.npy'))
    all_captions = []

    for item in range(len(val_ids)):
        src_img_id = val_ids[item]

        src_img_path = os.path.join(img_folder, str(images[src_img_id])[2:-1])
        tgt_img_paths = [str(followings[src_img_id][i])[2:-1] for i in range(4)]
        # print(src_img_path, tgt_img_path)

        # open the target image and caption
        tgt_img_ids = [str(tgt_img_path).replace(img_folder, '').replace('.png', '') for tgt_img_path in tgt_img_paths]
        captions = [descriptions_original[src_img_path.replace(img_folder, '').replace('.png', '')][0]] + [descriptions_original[tgt_img_id][0] for tgt_img_id in
                                                             tgt_img_ids]
        all_captions.append(captions)

    with open(os.path.join(img_folder, 'descriptions_val.json'), 'w') as f:
        json.dump({i: captions for i, captions in enumerate(all_captions)}, f, indent=2)


def get_flintstones_texts():

    min_len = 4
    dir_path = '../StoryGAN/flintstones'
    splits = json.load(open(os.path.join(dir_path, 'train-val-test_split.json'), 'r'))
    _, val_id, test_id = splits["train"], splits["val"], splits["test"]
    followings = pickle.load(open(os.path.join(dir_path, 'following_cache' + str(min_len) + '.pkl'), 'rb'))
    val_id = [vid for vid in val_id if vid in followings]
    test_id = [tid for tid in test_id if tid in followings]
    val_id = [vid for vid in val_id if len(followings[vid]) == 4]
    test_id = [vid for vid in test_id if len(followings[vid]) == 4]
    annotations = json.load(open(os.path.join(dir_path, 'flintstones_annotations_v1-0.json')))
    descriptions = {}
    for sample in annotations:
        descriptions[sample["globalID"]] = sample["description"]

    all_captions = []
    for ids, name in zip([val_id, test_id], ['val', 'test']):
        for i, id in enumerate(ids):
            captions = [descriptions[id]] + [descriptions[j] for j in followings[id]]
            all_captions.append(captions)

        with open(os.path.join(dir_path, 'flintstones_annotations_%s.json' % name), 'w') as f:
            json.dump({i: captions for i, captions in enumerate(all_captions)}, f, indent=2)

def get_didemo_texts():
    mode = 'val'
    img_folder = '/nas-ssd/adyasha/datasets/didemo'
    file_path = os.path.join(img_folder, 'val.json')
    min_len = 2
    images = np.load(os.path.join(img_folder, 'img_cache' + str(min_len) + '_' + mode + '.npy'), encoding='latin1')
    followings = np.load(os.path.join(img_folder, 'following_cache' + str(min_len) + '_' + mode + '.npy'))
    descriptions_original = {tup[1]: tup[0] for ex in json.load(open(file_path)) for tup in ex['desc_to_frame']}

    all_captions = []
    for item in range(len(images)):
        frame_path_list = [images[item]]
        for i in range(len(followings[item])):
            frame_path_list.append(str(followings[item][i]))
        captions = []
        for img_path in frame_path_list:
            captions.append(descriptions_original[img_path])
        all_captions.append(captions)

    with open(os.path.join(img_folder, 'annotations_val.json'), 'w') as f:
        json.dump({i: captions for i, captions in enumerate(all_captions)}, f, indent=2)


if __name__ == "__main__":
    # get_pororo_idxs()
    # get_flintstones_idxs()
    # get_pororo_texts()
    # get_flintstones_texts()
    get_didemo_texts()