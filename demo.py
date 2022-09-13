import os, torch
import gradio as gr
import torchvision.utils as vutils
import torchvision.transforms as transforms
from min_dalle import MinDalle
import argparse
from PIL import Image
import numpy as np
from torchvision.utils import save_image
import tensorflow_hub as hub

source_frame_paths = {
    'Pororo': '/playpen-ssd/adyasha/projects/StoryGAN/pororo_png/Pororo_ENGLISH1_2/Pororo_ENGLISH1_2_ep6/12.png',
    'Loopy': '/playpen-ssd/adyasha/projects/StoryGAN/pororo_png/Pororo_ENGLISH1_1/Pororo_ENGLISH1_1_ep12/26.png',
    'Crong': '/playpen-ssd/adyasha/projects/StoryGAN/pororo_png/Pororo_ENGLISH1_1/Pororo_ENGLISH1_1_ep12/10.png',
    'Poby': '/playpen-ssd/adyasha/projects/StoryGAN/pororo_png/Pororo_ENGLISH1_1/Pororo_ENGLISH1_1_ep9/34.png',
    'Eddy': '/playpen-ssd/adyasha/projects/StoryGAN/pororo_png/Pororo_ENGLISH1_1/Pororo_ENGLISH1_1_ep12/46.png',
    'Petty': '/playpen-ssd/adyasha/projects/StoryGAN/pororo_png/Pororo_ENGLISH2_1/Pororo_ENGLISH2_1_ep1/34.png',
    'Tongtong': '/playpen-ssd/adyasha/projects/StoryGAN/pororo_png/Pororo_ENGLISH3_1/Pororo_ENGLISH3_1_ep7/8.png',
    'Rody': '/playpen-ssd/adyasha/projects/StoryGAN/pororo_png/Pororo_ENGLISH3_1/Pororo_ENGLISH3_1_ep6/66.png',
    'Harry': '/playpen-ssd/adyasha/projects/StoryGAN/pororo_png/Pororo_ENGLISH3_1/Pororo_ENGLISH3_1_ep7/39.png',
}

def inverse_normalize(tensor, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device)
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    tensor.mul_(std).add_(mean)
    return tensor

def save_story_results(images, video_len=4, n_candidates=1, mask=None):
    # print("Generated Images shape: ", images.shape)

    if mask is None:
        mask = [1 for _ in range(len(video_len))]

    all_images = []
    for i in range(len(images)): # batch size = 1
        for j in range(n_candidates):
            story = []
            for k, m in enumerate(mask):
                if m == 1:
                    story.append(images[i][j][k])
            all_images.append(vutils.make_grid(story, sum(mask), padding=0))
    all_images = vutils.make_grid(all_images, 1, padding=20)
    print(all_images.shape)
    return all_images[:, 15:-15, 15:-15]

def main(args):

    device = torch.device('cuda:0')

    if args.debug:
        model = None
        embed = None
    else:
        model = MinDalle(args.model_name_or_path,
                         dtype=torch.float32,
                         device='cuda',
                         is_mega=args.is_mega,
                         is_reusable=True,
                         is_train=True,
                         condition=args.tuning_mode == 'story')
        model.eval()
        embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

        image_resolution = 256
        valid_transform = transforms.Compose(
            [transforms.Resize(image_resolution),
             transforms.CenterCrop(image_resolution),
             transforms.ToTensor()]
            # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
        )

        if args.dataset_name == 'pororo':
            n_new_tokens = 9
            with torch.no_grad():
                print(len(model.tokenizer.token_from_subword), model.encoder.embed_tokens.weight.shape)
                model.tokenizer.add_tokens(
                    ['pororo', 'loopy', 'eddy', 'harry', 'poby', 'tongtong', 'crong', 'rody', 'petty'])
                # model.encoder.resize_token_embeddings(len(model.tokenizer.token_from_subword) + n_new_tokens)
        elif args.dataset_name == 'flintstones':
            n_new_tokens = 7
            with torch.no_grad():
                model.tokenizer.add_tokens(
                    ['fred', 'barney', 'wilma', 'betty', 'pebbles', 'dino', 'slate'])
                model.encoder.resize_token_embeddings(len(model.tokenizer.token_from_subword) + n_new_tokens)
        elif args.dataset_name == 'didemo':
            pass
        else:
            raise ValueError

    max_length = 64
    def predict(caption_1, caption_2, caption_3, caption_4, source='Pororo', top_k=32, top_p=0.2, n_candidates=4, supercondition=False):

        if not args.debug:
            captions = [caption_1, caption_2, caption_3, caption_4]
            mask = [1 if caption != '' else 0 for caption in captions]
            print(captions, mask, source)
            for i, caption in enumerate(captions):
                if caption == "":
                    captions[i] = "Pororo is reading a book."
            tokens = [model.tokenizer.tokenize(caption.lower()) for caption in captions]

            if supercondition:
                # superconditioning during generation
                text_tokens = np.ones((2 * len(tokens) * n_candidates, max_length), dtype=np.int32)
                for j, words in enumerate(tokens):
                    text_tokens[i, :2] = [words[0], words[-1]]
                    text_tokens[len(tokens) + i, :len(words)] = words
                tokens = torch.tensor(
                    text_tokens,
                    dtype=torch.long,
                )
                # tokens = torch.stack([torch.LongTensor(token.ids) for token in tokens])
            else:
                text_tokens = np.ones((len(tokens) * n_candidates, max_length), dtype=np.int32)
                for i, words in enumerate(tokens):
                    text_tokens[i, :len(words)] = words
                tokens = torch.tensor(text_tokens, dtype=torch.long)

            # texts = torch.stack([torch.LongTensor(token.ids) for token in tokens]).unsqueeze(0)
            # sent_embeds = torch.tensor(embed(captions).numpy())
            # sent_embeds = torch.tensor(description_vecs[source_frame_paths[source].
            #                            replace('/playpen-ssd/adyasha/projects/StoryGAN/pororo_png/', '')[:-4]][0]).unsqueeze(0).repeat(4, 1)

            # todo repeat
            src_image = valid_transform(Image.open('./demo/%s.png' % source).convert('RGB')).unsqueeze(0)

            stories = []
            with torch.no_grad():
                pixels = model.sample_images(tokens.to(device),
                                             src_image.to(device),
                                             top_k=top_k,
                                             top_p=top_p,
                                             supercondition=supercondition).cpu().transpose(1, -1).transpose(-1, -2)
                stories.append(pixels)

            # img = save_story_results(stories, video_len=4, n_candidates=n_candidates, mask=mask)
            # image = Image.fromarray(img.to('cpu').to(torch.uint8).numpy())
            # image.save('gradio_demo_pororo.png')
            torch.save(pixels, 'pixels.pt')
            print(pixels.shape)
            pixels = pixels.transpose(0, 1).flatten(1, 2)
            print(pixels.shape)
            image = Image.fromarray(pixels.to('cpu').to(torch.uint8).numpy())
            image.save('gradio_demo_pororo.png')

        return "gradio_demo_pororo.png"



    # demo = gr.Interface(
    #     predict,
    #     [
    #         "text",
    #         "text",
    #         "text",
    #         "text",
    #         gr.Radio(["Pororo", "Loopy", "Crong", "Poby", "Eddy", "Petty", "Tongtong", "Rody", "Harry"]),
    #         gr.Slider(16, 128),
    #         gr.Slider(0.01, 1.0),
    #         gr.Dropdown([1, 2, 3])
    #     ],
    #     "image",
    #     title="StoryDALL-E",
    #     description=gr.Markdown("StoryDALL-E is a model trained for the task of Story Visualization \[1\]. The model receives a sequence of captions as input and generates a corresponding sequence of images which form a visual story depicting the narrative in the captions. It is based on the [mega-dalle](https://github.com/borisdayma/dalle-mini) model and is adapted from the corresponding [PyTorch codebase](https://github.com/kuprel/min-dalle). This model has been developed for academic purposes only. "
    #                 "### Dataset\n\nThis model has been trained using the Pororo story visualization dataset \[1\]. The data was adapted from the popular cartoon series *Pororo the Little Penguin* and originally released by \[2\]. The Pororo dataset contains 9 recurring characters, as shown below, in the decreasing order of their frequency in the training data. The training dataset contains nearly 10,000 samples in the training set. Most of the scenes occur in a snowy village, surrounded by hills, trees and houses. A few episodes are located in gardens or water bodies. All the captions are in the English language and predominantly contain verbs in the present tense. Additionally, the training of this model starts from the pretrained checkpoint of mega-dalle, which is trained on the Conceptual Captions dataset. ![](file/demo_pororo_good.png)"),
    #
    # )
    # demo.launch(share=True)

    #         <p style="text-align: center;font-size:40px;"><b>StoryDALL-E: Adapting Pretrained Text-to-Image Transformers for Story Continuation</b></p>
    # <body style="text-align: center;font-size:40px;"> Adyasha Maharana, Darryl Hannan and Mohit Bansal [UNC Chapel Hill]</b> Published at ECCV 2022</body>

    with gr.Blocks(css = '#output {width:750px; height:750px; float:left;}') as demo:
        gr.Markdown('''
        <p style="text-align: center;font-size:40px;"><b>StoryDALL-E: Adapting Pretrained Text-to-Image Transformers for Story Continuation</b><br><font size="6">Adyasha Maharana, Darryl Hannan and Mohit Bansal (UNC Chapel Hill)<br>Published at <b>ECCV 2022</b></font></p>
        
        StoryDALL-E is a model trained for the task of Story Visualization \[1\].
        The model receives a sequence of captions as input and generates a corresponding sequence of images which form a visual story depicting the narrative in the captions.
        It is based on the [mega-dalle](https://github.com/borisdayma/dalle-mini) model and is adapted from the corresponding [PyTorch codebase](https://github.com/kuprel/min-dalle).
        **This model has been developed for academic purposes only.**
        
        \[[Paper]()\]  \[[Code]()\] \[[Model Card]()\]
        
        ### Dataset
        This model has been trained using the Pororo story visualization dataset \[1\].
        The data was adapted from the popular cartoon series *Pororo the Little Penguin* and originally released by \[2\].
        The Pororo dataset contains 9 recurring characters, as shown below, in the decreasing order of their frequency in the training data.
        <p align="center">
            <img src="file/pororo_characters.png" width="800">
        </p>
        The training dataset contains nearly 10,000 samples in the training set. Most of the scenes occur in a snowy village, surrounded by hills, trees and houses. A few episodes are located in gardens or water bodies. All the captions are in the English language and predominantly contain verbs in the present tense. Additionally, the training of this model starts from the pretrained checkpoint of mega-dalle, which is trained on the Conceptual Captions dataset."),
        
        ### Intended Use
        This model is intended for generating visual stories containing the 9 characters in the Pororo dataset. This version of the StoryDALL-E model is reasonable at the following scenarios:
        * Frames containing a single character.
        * Overtly visual actions such as *making cookies*, *walking*, *reading a book*, *sitting*.
        * Scenes taking place in snowy settings, indoors and gardens.
        * Visual stories contaning 1-3 characters across all frames.
        * Scene transitions e.g. from day to night.
        
        Here are some examples of generated visual stories for the above-mentioned settings.

        <p align="center">
            <img src="file/demo_pororo_good_v1.png" width="1000">
        </p>
        
        Due to the small training dataset size for story visualization, the model has poor generalization to some unseen settings. The model struggles to generate coherent images in the following scenarios.
        * Multiple characters in a frame.
        * Non-visual actions such as *compliment*.
        * Characters that are infrequent in the training dataset e.g. Rody, Harry.
        * Background locations that are not found in the cartoon e.g. stage in a concert.
        * Color-based descriptions for object.
        * Completely new characters based on textual descriptions.
        ''')

        with gr.Row():

            with gr.Column():
                caption_1 = gr.Textbox(label="Caption 1")
                caption_2 = gr.Textbox(label="Caption 2")
                caption_3 = gr.Textbox(label="Caption 3")
                caption_4 = gr.Textbox(label="Caption 4")
                source = gr.Radio(["Pororo", "Loopy", "Crong", "Poby", "Eddy", "Petty", "Tongtong", "Rody", "Harry"], label="Source", value="Pororo")
                top_k = gr.Slider(16, 128, label="top_k", value=32)
                top_p = gr.Slider(0.01, 1.0, label="top_p", value=0.2)
                supercondition = gr.Checkbox(value=False, label='supercondition')
                n_candidates = gr.Dropdown([1, 2, 3, 4], value=4, label='n_candidates')

                with gr.Row():
                    # clear_btn = gr.Button("Clear")
                    submit_btn = gr.Button("Submit")

            with gr.Column():

                with gr.Row():
                    frame_1_label = gr.Button("Frame 1")
                    frame_2_label = gr.Button("Frame 2")
                    frame_3_label = gr.Button("Frame 2")
                    frame_4_label = gr.Button("Frame 4")
                    # frame_1_label = gr.Label("Frame 1")
                    # frame_2_label = gr.Label("Frame 2")
                    # frame_3_label = gr.Label("Frame 3")
                    # frame_4_label = gr.Label("Frame 4")
                output = gr.Image(label="", elem_id='output')

        submit_btn.click(fn=predict, inputs=[caption_1, caption_2, caption_3, caption_4, source, top_k, top_p, n_candidates, supercondition], outputs=output)

        gr.Markdown('''
        ### References

        \[1\] Li, Yitong, et al. "Storygan: A sequential conditional gan for story visualization." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019.

        \[2\] Kim, Kyung-Min, et al. "DeepStory: video story QA by deep embedded memory networks." Proceedings of the 26th International Joint Conference on Artificial Intelligence. 2017.

        \[3\] Sharma, Piyush, et al. "Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning." Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2018.
        ''')

    demo.launch(share=True)


if __name__ == "__main__":

    args_list = ['--model_name_or_path', '/nas-hdd/tarbucket/adyasha/models/megaDALLE/pororo/Model/ckpt_20/',
                 '--prefix_model_name_or_path', './1.3B/',
                 '--dataset_name', 'pororo',
                 '--tuning_mode', 'story',
                 '--preseqlen', '32',
                 '--condition',
                 '--story_len', '4',
                 '--sent_embed', '512',
                 '--prefix_dropout', '0.2',
                 '--data_dir', '/playpen-ssd/adyasha/projects/StoryGAN/pororo_png/',
                 '--dataloader_num_workers', '1',
                 '--do_eval',
                 '--per_gpu_eval_batch_size', '16',
                 '--mode', 'story', '--is_mega']

    parser = argparse.ArgumentParser(description='arguments for training/evaluating prefix-tuning DALLE')

    # Model Arguments
    parser.add_argument('--model_name_or_path', type=str, default=None,
                        help='The model checkpoint for weights initialization.')
    parser.add_argument('--prefix_model_name_or_path', type=str, default=None,
                        help='The prefix model checkpoint for weights initialization.')
    parser.add_argument('--prefix_mode', type=str, default='activation', help='activation or embedding')
    parser.add_argument('--preseqlen', type=int, default=0, help='how many tokens of prefix should we include.')
    parser.add_argument('--optim_prefix', action="store_true",
                        help='set to True if optimizing prefix directly; no if through amortized function')
    parser.add_argument('--tuning_mode', type=str, default='prefixtune', help='prefixtune or finetune')
    parser.add_argument('--top_k_layers', type=int, default=2,
                        help='In finetuning setting, if we only tune the top k layers.')
    parser.add_argument('--parameterize_mode', type=str, default='mlp',
                        help="mlp or emb to parametrize when we optimize for the embeddings.")
    parser.add_argument('--prefix_dropout', type=float, default=0.0, help='dropout rate for the prefix tuning model.')
    parser.add_argument('--teacher_dropout', type=float, default=0.0, help='dropout rate for the teacher model.')
    parser.add_argument('--init_random', action="store_true", help="set True if initializing random embeddings")
    parser.add_argument('--init_shallow', action="store_true", help="set True if not using reparameterization")
    parser.add_argument('--init_shallow_word', type=bool, default=False,
                        help="set True if init_shallow and specify words")
    parser.add_argument('--replay_buffer', action="store_true", help="set True if using replay buffer in training")
    parser.add_argument('--gumbel', action="store_true", help="set True if using the gumbel softmax in training")
    parser.add_argument('--hidden_dim_prefix', type=float, default=512, help="hidden dim of MLP for generating prefix?")

    parser.add_argument("--is_mega", action="store_true", help="Whether to run training.")


    # Data Arguments
    parser.add_argument('--dataset_name', type=str, default='pororo', help="dataset name")
    parser.add_argument('--data_dir', type=str, default=None, help="Path to data directory")
    parser.add_argument('--lowdata_token', type=str, default='story',
                        help="The token to be prepended at initialization time.")
    parser.add_argument('--use_lowdata_token', type=bool, default=True,
                        help="Whether we should use the lowdata token for prefix-tuning")
    parser.add_argument('--train_embeddings', action="store_true", help="Whether to train word embeddings")
    parser.add_argument('--train_max_target_length', type=int, default=100,
                        help='the max target length for training data.')
    parser.add_argument('--val_max_target_length', type=int, default=100, help='the max target length for dev data.')
    parser.add_argument('--dataloader_num_workers', type=int, default=8, help='number of workers when loading data')

    # new arguments for story
    parser.add_argument('--prompt', action="store_true", help="set True if using prompts in StoryDALLE")
    parser.add_argument('--story_len', type=int, default=4, help='the max target length for dev data.')
    parser.add_argument('--sent_embed', type=int, default=384, help='the max target length for dev data.')
    parser.add_argument('--condition', action="store_true", help="set True if using prompts in StoryDALLE")
    parser.add_argument('--clip_embed', action="store_true", help="set True if using prompts in StoryDALLE")

    # Training Arguments
    parser.add_argument('--output_dir', type=str, default=None, help="Path to data directory")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run evaluation.")
    parser.add_argument("--do_test", action="store_true", help="Whether to run test.")
    parser.add_argument('--seed', type=int, default=42, help='seed for reproducibility')
    parser.add_argument("--overwrite_output_dir", action="store_true", help="Whether to overwrite output dir.")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    parser.add_argument('--mode', type=str, default='val', help="mval or test.")

    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3, type=int, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )

    parser.add_argument("--debug", action="store_true", help="Whether to debug the demo.")

    args = parser.parse_args(args_list)

    main(args)





