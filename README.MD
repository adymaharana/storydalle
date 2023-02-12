## StoryDALL-E: Adapting Pretrained Text-to-Image Transformers for Story Continuation

PyTorch code for the ECCV 2022 paper "StoryDALL-E: Adapting Pretrained Text-to-Image Transformers for Story Continuation".

\[[Paper](https://arxiv.org/abs/2209.06192)\] \[[Model Card](https://github.com/adymaharana/storydalle/blob/main/MODEL_CARD.MD)\] \[[Spaces Demo](https://huggingface.co/spaces/ECCV2022/storydalle)\] \[[Replicate Demo](https://replicate.com/adymaharana/story-dalle)\]

![image](./assets/story_dalle_predictions.png)

![image](./assets/story_dalle.png)

### Training

#### Prepare Repository:
Download the PororoSV dataset and associated files from [here](https://drive.google.com/file/d/11Io1_BufAayJ1BpdxxV2uJUvCcirbrNc/view?usp=sharing) (updated) and save it as ```./data/pororo/```.<br>
Download the FlintstonesSV dataset and associated files from [here](https://drive.google.com/file/d/1kG4esNwabJQPWqadSDaugrlF4dRaV33_/view?usp=sharing) and save it as ```./data/flintstones```<br>
Download the DiDeMoSV dataset and associated files from [here](https://drive.google.com/file/d/1zgj_bpE6Woyi-G76axF0nO-yzQaLBayc/view?usp=sharing) and save it as ```./data/didemo```<br>

This repository contains separate folders for training StoryDALL-E based on [minDALL-E](https://github.com/kakaobrain/minDALL-E) and [DALL-E Mega](https://github.com/kuprel/min-dalle) models i.e. the ```./story_dalle/``` and ```./mega-story-dalle``` models respectively.

#### Training StoryDALL-E based on minDALL-E:

1. To finetune the minDALL-E model for story continuation, first migrate to the corresponding folder:\
```cd story-dalle```<br>
2. Set the environment variables in ```train_story.sh``` to point to the right locations in your system. Specifically, change the ```$DATA_DIR```, ```$OUTPUT_ROOT``` and ```$LOG_DIR``` if different from the default locations.
3. Download the pretrained checkpoint from [here](https://github.com/kakaobrain/minDALL-E) and save it in ```./1.3B```
4. Run the following command:
```bash train_story.sh <dataset_name>```

   
#### Training StoryDALL-E based on DALL-E Mega:

1. To finetune the DALL-E Mega model for story continuation, first migrate to the corresponding folder:\
```cd mega-story-dalle```<br>
2. Set the environment variables in ```train_story.sh``` to point to the right locations in your system. Specifically, change the ```$DATA_DIR```, ```$OUTPUT_ROOT``` and ```$LOG_DIR``` if different from the default locations.
3. Pretrained checkpoints for generative model and VQGAN detokenizer are automatically downloaded upon initialization. Download the pretrained weights for VQGAN tokenizer from [here](https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/) and place it in the same folder as VQGAN detokenizer.
4. Run the following command:
```bash train_story.sh <dataset_name>```

### Inference
Pretrained checkpoints for minDALL-E based StoryDALL-E can be downloaded from here: [PororoSV](https://drive.google.com/file/d/1lJ6zMZ6qTvFu6H35-VEdFlN13MMslivJ/view?usp=sharing)<br>

For a demo of inference using cog, check out [this repo](https://github.com/daanelson/story-dalle-cog). 

Links to other pretrained checkpoints and inference instructions coming soon!

