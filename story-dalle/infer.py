from matplotlib import pyplot as plt
import clip
from dalle.models import Dalle
from dalle.utils.utils import set_seed, clip_score
import numpy as np
from argparse import ArgumentParser

device = 'cuda:0'
set_seed(0)

parser = ArgumentParser(description='arguments for training/evaluating prefix-tuning DALLE')

# Model Arguments
parser.add_argument('--model_name_or_path', type=str, default=None, help='The model checkpoint for weights initialization.')
parser.add_argument("--do_train", action="store_true", help="Whether to run evaluation.")
parser.add_argument("--do_eval", action="store_true", help="Whether to run evaluation.")
args = parser.parse_args()

# prompt = "A painting of a monkey with sunglasses in the frame"
prompt = "A group of elephants walking in muddy water"
model, _ = Dalle.from_pretrained(args)  # This will automatically download the pretrained model.
model.to(device=device)

# Sampling
images = model.sampling(prompt=prompt,
                        top_k=256, # It is recommended that top_k is set lower than 256.
                        top_p=None,
                        softmax_temperature=1.0,
                        num_candidates=16,
                        device=device).cpu().numpy()
images = np.transpose(images, (0, 2, 3, 1))

# CLIP Re-ranking
model_clip, preprocess_clip = clip.load("ViT-B/32", device=device)
model_clip.to(device=device)
rank = clip_score(prompt=prompt,
                  images=images,
                  model_clip=model_clip,
                  preprocess_clip=preprocess_clip,
                  device=device)

# Plot images
images = images[rank]
# plt.imshow(images[0])
plt.imsave('./out/test.png', images[0])