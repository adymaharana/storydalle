import json
import math
import pickle
from tqdm import tqdm
import numpy as np
import sys
import os

def get_embeddings(dataset='didemo', data_dir = ''):


    if dataset == 'flintstones':
        annotations = json.load(open('../flintstones/flintstones_annotations_v1-0.json', 'r'))
        globalIDs = [s["globalID"] for s in annotations]
        descriptions = [s["description"] for s in annotations]
    elif dataset == 'didemo':
        descriptions = {}
        for filepath in ['train.json', 'val.json', 'test.json']:
            d = {tup[1]: tup[0] for ex in json.load(open(os.path.join(data_dir, filepath))) for tup in ex['desc_to_frame']}
            descriptions.update(d)
    elif dataset == 'mpii':
        all_keys = []
        descriptions = {}
        for filepath in ['train.json', 'val.json', 'test.json']:
            data = json.load(open(os.path.join(data_dir, filepath)))
            print(len(data))
            for ex in tqdm(data):
                for tup in ex['desc_to_frame']:
                    k = tup[1].replace('/ssd-playpen/dhannan/StoryDatasets/mpii/', '')
                    all_keys.append(k)
                    descriptions[k] = tup[0]
        print(len(descriptions), len(all_keys), len(set(all_keys)))
        print(set(all_keys))
    else:
        raise ValueError
    sys.exit()

    import tensorflow_hub as hub
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
    all_embeddings = None
    bs = 128
    description_keys = list(descriptions.keys())
    for i in tqdm(range(0, math.ceil(len(description_keys)/bs)), desc="Extraction seed embeddings"):
        # embeddings = embed(descriptions[i*bs:(i+1)*bs]).numpy()
        embeddings = embed([descriptions[k] for k in description_keys[i * bs:(i + 1) * bs]]).numpy()
        if all_embeddings is None:
            all_embeddings = embeddings
        else:
            all_embeddings = np.concatenate([all_embeddings, embeddings], axis=0)
    print(all_embeddings.shape, len(description_keys))
    embeddings = {k: v for v, k in zip(all_embeddings, description_keys)}
    pickle.dump(embeddings, open(os.path.join(data_dir, '%s_use_embed_idxs.pkl' % dataset), 'wb'))

    # np.save(os.path.join(data_dir, '%s_use_embeddings.npy' % dataset), all_embeddings)
    # pickle.dump({key: val for val, key in enumerate(globalIDs)}, open(os.path.join(data_dir, '%s_use_embed_idxs.pkl'), 'wb'))


# get_embeddings('didemo', '/nas-ssd/adyasha/datasets/didemo')
get_embeddings('mpii', '/nas-ssd/adyasha/datasets/mpii')