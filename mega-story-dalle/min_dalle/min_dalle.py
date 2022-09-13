import os
from PIL import Image
import numpy
from torch import LongTensor, FloatTensor
import torch
import torch.nn as nn
import torch.backends.cudnn, torch.backends.cuda
import json
import requests
from typing import Iterator
from .text_tokenizer import TextTokenizer
from .models import DalleBartEncoder, DalleBartDecoder, VQGanDetokenizer, VQGanTokenizer
from torch.cuda.amp import autocast

torch.set_grad_enabled(True)
torch.set_num_threads(os.cpu_count())
torch.backends.cudnn.enabled = True
torch.backends.cudnn.allow_tf32 = True

MIN_DALLE_REPO = 'https://huggingface.co/kuprel/min-dalle/resolve/main/'
IMAGE_TOKEN_COUNT = 256


class MinDalle(nn.Module):

    def __init__(
        self,
        models_root: str = 'pretrained',
        dtype: torch.dtype = torch.float32,
        device: str = None,
        is_mega: bool = True, 
        is_reusable: bool = True,
        is_verbose = True,
        is_train = False,
        condition = False
    ):
        super().__init__()
        if device == None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if is_verbose: print("using device", device)
        self.device = device
        self.is_mega = is_mega
        self.is_reusable = is_reusable
        self.dtype = dtype
        self.is_verbose = is_verbose
        self.text_token_count = 64
        self.layer_count = 24 if is_mega else 12
        self.attention_head_count = 32 if is_mega else 16
        self.embed_count = 2048 if is_mega else 1024
        self.glu_embed_count = 4096 if is_mega else 2730
        self.text_vocab_count = 50272 if is_mega else 50264
        self.image_vocab_count = 16415 if is_mega else 16384
        self.condition = condition

        model_name = 'dalle_bart_{}'.format('mega' if is_mega else 'mini')
        dalle_path = os.path.join(models_root, model_name)
        vqgan_path = os.path.join(models_root, 'vqgan')
        if not os.path.exists(dalle_path): os.makedirs(dalle_path)
        if not os.path.exists(vqgan_path): os.makedirs(vqgan_path)
        self.vocab_path = os.path.join(dalle_path, 'vocab.json')
        self.merges_path = os.path.join(dalle_path, 'merges.txt')
        self.encoder_params_path = os.path.join(dalle_path, 'encoder.pt')
        self.decoder_params_path = os.path.join(dalle_path, 'decoder.pt')
        self.detoker_params_path = os.path.join(vqgan_path, 'detoker.pt')
        self.toker_params_path = os.path.join(vqgan_path, 'last.ckpt')

        if 'pororo' in models_root:
            self.text_vocab_count = 50283 if is_mega else 50264

        self.init_tokenizer()
        if is_reusable:
            self.init_encoder(is_train)
            self.init_decoder(is_train, condition)
            self.init_detokenizer()
            self.init_vqgan_tokenizer()

        # if is_train:
        #     self.register_buffer("mask", torch.ones(IMAGE_TOKEN_COUNT, IMAGE_TOKEN_COUNT), persistent=False)
        #     self.mask = torch.tril(self.mask).view(1, IMAGE_TOKEN_COUNT, IMAGE_TOKEN_COUNT)

    def download_tokenizer(self):
        if self.is_verbose: print("downloading tokenizer params")
        suffix = '' if self.is_mega else '_mini'
        _ = requests.get(MIN_DALLE_REPO + 'config.json') # trigger HF download
        vocab = requests.get(MIN_DALLE_REPO + 'vocab{}.json'.format(suffix))
        merges = requests.get(MIN_DALLE_REPO + 'merges{}.txt'.format(suffix))
        with open(self.vocab_path, 'wb') as f: f.write(vocab.content)
        with open(self.merges_path, 'wb') as f: f.write(merges.content)


    def download_encoder(self):
        if self.is_verbose: print("downloading encoder params")
        suffix = '' if self.is_mega else '_mini'
        params = requests.get(MIN_DALLE_REPO + 'encoder{}.pt'.format(suffix))
        with open(self.encoder_params_path, 'wb') as f: f.write(params.content)


    def download_decoder(self):
        if self.is_verbose: print("downloading decoder params")
        suffix = '' if self.is_mega else '_mini'
        params = requests.get(MIN_DALLE_REPO + 'decoder{}.pt'.format(suffix))
        with open(self.decoder_params_path, 'wb') as f: f.write(params.content)
    

    def download_detokenizer(self):
        if self.is_verbose: print("downloading detokenizer params")
        params = requests.get(MIN_DALLE_REPO + 'detoker.pt')
        with open(self.detoker_params_path, 'wb') as f: f.write(params.content)


    def init_tokenizer(self):
        _ = requests.get(MIN_DALLE_REPO + 'config.json') # trigger HF download
        is_downloaded = os.path.exists(self.vocab_path)
        is_downloaded &= os.path.exists(self.merges_path)
        if not is_downloaded: self.download_tokenizer()
        if self.is_verbose: print("intializing TextTokenizer")
        with open(self.vocab_path, 'r', encoding='utf8') as f:
            vocab = json.load(f)
        with open(self.merges_path, 'r', encoding='utf8') as f:
            merges = f.read().split("\n")[1:-1]
        self.tokenizer = TextTokenizer(vocab, merges)


    def init_encoder(self, is_train=False):
        is_downloaded = os.path.exists(self.encoder_params_path)
        if not is_downloaded: self.download_encoder()
        if self.is_verbose: print("initializing DalleBartEncoder")
        self.encoder = DalleBartEncoder(
            attention_head_count = self.attention_head_count,
            embed_count = self.embed_count,
            glu_embed_count = self.glu_embed_count,
            text_token_count = self.text_token_count,
            text_vocab_count = self.text_vocab_count,
            layer_count = self.layer_count,
            device=self.device
        ).to(self.dtype)

        if not is_train:
            self.encoder.eval()

        params = torch.load(self.encoder_params_path)
        self.encoder.load_state_dict(params, strict=False)
        del params
        self.encoder = self.encoder.to(device=self.device)


    def init_decoder(self, is_train=False, condition=False):
        is_downloaded = os.path.exists(self.decoder_params_path)
        if not is_downloaded: self.download_decoder()
        if self.is_verbose: print("initializing DalleBartDecoder")
        self.decoder = DalleBartDecoder(
            image_vocab_count = self.image_vocab_count,
            attention_head_count = self.attention_head_count,
            embed_count = self.embed_count,
            glu_embed_count = self.glu_embed_count,
            layer_count = self.layer_count,
            device=self.device,
            condition=condition
        ).to(self.dtype)

        if not is_train:
            self.decoder.eval()

        params = torch.load(self.decoder_params_path)
        self.decoder.load_state_dict(params, strict=False)
        del params
        self.decoder = self.decoder.to(device=self.device)


    def init_detokenizer(self):
        is_downloaded = os.path.exists(self.detoker_params_path)
        if not is_downloaded: self.download_detokenizer()
        if self.is_verbose: print("initializing VQGanDetokenizer")
        self.detokenizer = VQGanDetokenizer().eval()
        params = torch.load(self.detoker_params_path)
        self.detokenizer.load_state_dict(params)
        del params
        self.detokenizer = self.detokenizer.to(device=self.device)

    def init_vqgan_tokenizer(self):

        self.vqgan_tokenizer = VQGanTokenizer().eval()
        params = torch.load(self.toker_params_path)['state_dict']
        self.vqgan_tokenizer.load_state_dict(params, strict=False)
        del params
        self.vqgan_tokenizer = self.vqgan_tokenizer.to(device=self.device)
        print("Successfully loaded VQGAN Tokenizer")


    def image_grid_from_tokens(
        self,
        image_tokens: LongTensor,
        is_seamless: bool,
        is_verbose: bool = False
    ) -> FloatTensor:
        if not self.is_reusable: del self.decoder
        torch.cuda.empty_cache()
        if not self.is_reusable: self.init_detokenizer()
        if is_verbose: print("detokenizing image")
        images = self.detokenizer.forward(is_seamless, image_tokens, grid=True)
        if not self.is_reusable: del self.detokenizer
        return images


    def generate_raw_image_stream(
        self, 
        text: str, 
        seed: int,
        grid_size: int,
        progressive_outputs: bool = False,
        is_seamless: bool = False,
        temperature: float = 1,
        top_k: int = 256,
        supercondition_factor: int = 16,
        is_verbose: bool = False
    ) -> Iterator[FloatTensor]:
        image_count = grid_size ** 2
        if is_verbose: print("tokenizing text")
        tokens = self.tokenizer.tokenize(text, is_verbose=is_verbose)
        if len(tokens) > self.text_token_count: 
            tokens = tokens[:self.text_token_count]
        if is_verbose: print("{} text tokens".format(len(tokens)), tokens)
        text_tokens = numpy.ones((2, 64), dtype=numpy.int32)
        text_tokens[0, :2] = [tokens[0], tokens[-1]]
        text_tokens[1, :len(tokens)] = tokens
        text_tokens = torch.tensor(
            text_tokens, 
            dtype=torch.long, 
            device=self.device
        )

        if not self.is_reusable: self.init_encoder()
        if is_verbose: print("encoding text tokens")
        with torch.cuda.amp.autocast(dtype=self.dtype):
            encoder_state = self.encoder.forward(text_tokens)
        if not self.is_reusable: del self.encoder
        torch.cuda.empty_cache()

        if not self.is_reusable: self.init_decoder()

        with torch.cuda.amp.autocast(dtype=self.dtype):
            expanded_indices = [0] * image_count + [1] * image_count
            text_tokens = text_tokens[expanded_indices]
            encoder_state = encoder_state[expanded_indices]
            attention_mask = text_tokens.not_equal(1)[:, None, None, :]
            attention_state = torch.zeros(
                size=(
                    self.layer_count,
                    image_count * 4,
                    IMAGE_TOKEN_COUNT,
                    self.embed_count
                ), 
                device=self.device
            )
            image_tokens = torch.full(
                (image_count, IMAGE_TOKEN_COUNT + 1), 
                2 ** 14 - 1,
                dtype=torch.long,
                device=self.device
            )
            
            if seed > 0: torch.manual_seed(seed)

        token_indices = torch.arange(IMAGE_TOKEN_COUNT, device=self.device)
        settings = torch.tensor(
            [temperature, top_k, supercondition_factor], 
            dtype=torch.float32,
            device=self.device
        )
        for i in range(IMAGE_TOKEN_COUNT):
            torch.cuda.empty_cache()                
            with torch.cuda.amp.autocast(dtype=self.dtype):
                image_tokens[:, i + 1], attention_state = self.decoder.sample_tokens(
                    settings=settings,
                    attention_mask=attention_mask,
                    encoder_state=encoder_state,
                    attention_state=attention_state,
                    # prev_tokens=image_tokens[:, :i+1],
                    # token_index=token_indices[:i+1]
                    prev_tokens=image_tokens[:, [i]],
                    token_index=token_indices[[i]]
                )

            print(image_tokens.shape, attention_state.shape)

            with torch.cuda.amp.autocast(dtype=torch.float32):
                if ((i + 1) % 32 == 0 and progressive_outputs) or i + 1 == 256:
                    yield self.image_grid_from_tokens(
                        image_tokens=image_tokens[:, 1:],
                        is_seamless=is_seamless,
                        is_verbose=is_verbose
                    )

    def generate_image_stream(self, *args, **kwargs) -> Iterator[Image.Image]:
        image_stream = self.generate_raw_image_stream(*args, **kwargs)
        for image in image_stream:
            image = image.to(torch.uint8).to('cpu').numpy()
            yield Image.fromarray(image)


    def generate_images_stream(self, *args, **kwargs) -> Iterator[FloatTensor]:
        image_stream = self.generate_raw_image_stream(*args, **kwargs)
        for image in image_stream:
            grid_size = kwargs['grid_size']
            image = image.view([grid_size * 256, grid_size, 256, 3])
            image = image.transpose(1, 0)
            image = image.reshape([grid_size ** 2, 2 ** 8, 2 ** 8, 3])
            yield image


    def generate_image(self, *args, **kwargs) -> Image.Image:
        image_stream = self.generate_image_stream(
            *args, **kwargs, 
            progressive_outputs=False
        )
        return next(image_stream)


    def generate_images(self, *args, **kwargs) -> Image.Image:
        images_stream = self.generate_images_stream(
            *args, **kwargs, 
            progressive_outputs=False
        )
        return next(images_stream)

    def sample_images(self, text_tokens, condition_state,
                      top_k=32, top_p=0.2, supercondition=False):

        # TODO: Add supercondition

        image_count = text_tokens.shape[0]
        encoder_state = self.encoder.forward(text_tokens)
        # print('Encoder state: ', encoder_state.shape)
        with torch.cuda.amp.autocast(dtype=self.dtype):
            attention_mask = text_tokens.not_equal(1)[:, None, None, :]
            # print(text_tokens, attention_mask)
            attention_state = torch.zeros(
                size=(
                    self.layer_count,
                    image_count * 2,
                    IMAGE_TOKEN_COUNT,
                    self.embed_count
                ),
                device=self.device
            )
            image_tokens = torch.full(
                (image_count, IMAGE_TOKEN_COUNT + 1),
                2 ** 14 - 1,
                dtype=torch.long,
                device=self.device
            )
        token_indices = torch.arange(IMAGE_TOKEN_COUNT, device=self.device)
        temperature = 1,
        top_k = top_k,
        supercondition_factor = 1,
        settings = torch.tensor(
            [temperature, top_k, supercondition_factor],
            dtype=torch.float32,
            device=self.device
        )

        if self.condition:
            # print("Image shape:", images.shape)
            with torch.no_grad():
                with autocast(enabled=False):
                    assert condition_state is not None
                    _, _, [_, _, condition_state] = self.vqgan_tokenizer(condition_state)
                    condition_state = condition_state.detach()
                    if condition_state is not None:
                        condition_state = condition_state.view(condition_state.shape[0], -1)
                    condition_state = self.decoder.embed_tokens.forward(condition_state)
                    B_c, N_c = condition_state.shape[:2]
                    pos_enc_tokens = torch.arange(N_c, device=condition_state.device).repeat((B_c, 1))
                    # print(condition_state.shape, pos_enc_tokens.shape)
                    condition_state += self.decoder.embed_positions.forward(pos_enc_tokens)
                    # print(condition_state.shape)
                    condition_state = condition_state.repeat_interleave(int(image_count / B_c), dim=0)
                    # print(condition_state.shape)
        else:
            condition_state = None

        for i in range(IMAGE_TOKEN_COUNT):
            torch.cuda.empty_cache()
            with torch.cuda.amp.autocast(dtype=self.dtype):
                image_tokens[:, i + 1], attention_state = self.decoder.sample_tokens(
                    settings=settings,
                    attention_mask=attention_mask,
                    encoder_state=encoder_state,
                    attention_state=attention_state,
                    # prev_tokens=image_tokens[:, :i+1],
                    # token_index=token_indices[:i+1]
                    prev_tokens=image_tokens[:, [i]],
                    token_index=token_indices[[i]],
                    condition_state=condition_state
                )

        # print(image_tokens.shape, attention_state.shape)

        images = self.detokenizer.forward(False, image_tokens[:, 1:])
        return images

    def forward(self, images, text_tokens, condition_state=None):

        image_count = text_tokens.shape[0]
        encoder_state = self.encoder.forward(text_tokens)
        # print('encoder', encoder_state.requires_grad, encoder_state.grad_fn)
        # print(encoder_state.shape)

        with torch.cuda.amp.autocast(dtype=self.dtype):
            attention_mask = text_tokens.not_equal(1)[:, None, None, :]
            # print(text_tokens, attention_mask)
            attention_state = torch.zeros(
                size=(
                    self.layer_count,
                    image_count * 4,
                    IMAGE_TOKEN_COUNT,
                    self.embed_count
                ),
                device=self.device
            )

        # print(attention_mask.shape, attention_state.shape)

        # print("Image shape:", images.shape)
        with torch.no_grad():
            with autocast(enabled=False):
                _, _, [_, _, image_tokens] = self.vqgan_tokenizer(images)
                image_tokens = image_tokens.detach()

                if self.condition:
                    assert condition_state is not None
                    _, _, [_, _, condition_tokens] = self.vqgan_tokenizer(condition_state)
                    condition_tokens = condition_tokens.detach()
                else:
                    condition_tokens = None

        image_tokens = image_tokens.view(image_tokens.shape[0], -1)
        if condition_tokens is not None:
            condition_tokens = condition_tokens.view(condition_tokens.shape[0], -1)

        # add <bos> token to beginning of input
        image_bos_tokens = torch.full(
            (image_count, 1),
            2 ** 14 - 1,
            dtype=torch.long,
            device=self.device
        )
        image_input_tokens = torch.cat([image_bos_tokens, image_tokens[:, :-1]], dim=-1)
        # print("Image tokens shape:", image_input_tokens.shape)

        with torch.cuda.amp.autocast(dtype=self.dtype):
            image_logits, attention_state = self.decoder.forward(
                attention_mask=attention_mask,
                encoder_state=encoder_state,
                attention_state=attention_state,
                prev_tokens=image_input_tokens,
                condition_state=condition_tokens
            )

        # print('Image logits shape: ', image_logits.shape)

        return image_logits, image_tokens


class PipelineParallelMinDalle(MinDalle):

    def __init__(
        self, *args, **kwargs):
        super(PipelineParallelMinDalle, self).__init__(*args, **kwargs)
        self.device1 = torch.device("cuda:0")
        self.device2 = torch.device("cuda:1")
        self.encoder = self.encoder.to(self.device1)
        self.vqgan_tokenizer = self.vqgan_tokenizer.to(self.device1)
        self.detokenizer = self.detokenizer.to(self.device1)
        self.decoder = self.decoder.to(self.device2)
        self.decoder.token_indices = self.decoder.token_indices.to(self.device2)
        for i in range(self.decoder.layer_count):
            self.decoder.layers[i].token_indices = self.decoder.layers[i].token_indices.to(self.device2)

    def forward(self, images, text_tokens, condition_state=None):
        image_count = text_tokens.shape[0]
        encoder_state = self.encoder.forward(text_tokens).to(self.device2)
        # print('encoder', encoder_state.requires_grad, encoder_state.grad_fn)
        # print(encoder_state.shape)

        with torch.cuda.amp.autocast(dtype=self.dtype):
            attention_mask = text_tokens.not_equal(1)[:, None, None, :].to(self.device2)
            # print(text_tokens, attention_mask)
            attention_state = torch.zeros(
                size=(
                    self.layer_count,
                    image_count * 4,
                    IMAGE_TOKEN_COUNT,
                    self.embed_count
                ),
                device=self.device2
            )

        # print(attention_mask.shape, attention_state.shape)

        # print("Image shape:", images.shape)
        with torch.no_grad():
            with autocast(enabled=False):
                _, _, [_, _, image_tokens] = self.vqgan_tokenizer(images)
                image_tokens = image_tokens.detach()

                if self.condition:
                    assert condition_state is not None
                    _, _, [_, _, condition_tokens] = self.vqgan_tokenizer(condition_state)
                    condition_tokens = condition_tokens.detach()
                else:
                    condition_tokens = None

        image_tokens = image_tokens.view(image_tokens.shape[0], -1).to(self.device2)
        if condition_tokens is not None:
            condition_tokens = condition_tokens.view(condition_tokens.shape[0], -1).to(self.device2)

        # add <bos> token to beginning of input
        image_bos_tokens = torch.full(
            (image_count, 1),
            2 ** 14 - 1,
            dtype=torch.long,
            device=self.device2
        )
        image_input_tokens = torch.cat([image_bos_tokens, image_tokens[:, :-1]], dim=-1)
        # print("Image tokens shape:", image_input_tokens.shape)

        with torch.cuda.amp.autocast(dtype=self.dtype):
            image_logits, attention_state = self.decoder.forward(
                attention_mask=attention_mask,
                encoder_state=encoder_state,
                attention_state=attention_state,
                prev_tokens=image_input_tokens,
                condition_state=condition_tokens
            )

        # print('Image logits shape: ', image_logits.shape)

        return image_logits, image_tokens

    def sample_images(self, text_tokens, condition_state=None, supercondition=False):

        # TODO: Add supercondition
        image_count = text_tokens.shape[0]
        encoder_state = self.encoder.forward(text_tokens).to(self.device2)
        # print('Encoder state: ', encoder_state.shape)
        with torch.cuda.amp.autocast(dtype=self.dtype):
            attention_mask = text_tokens.not_equal(1)[:, None, None, :].to(self.device2)
            # print(text_tokens, attention_mask)
            attention_state = torch.zeros(
                size=(
                    self.layer_count,
                    image_count * 2,
                    IMAGE_TOKEN_COUNT,
                    self.embed_count
                ),
                device=self.device2
            )
            image_tokens = torch.full(
                (image_count, IMAGE_TOKEN_COUNT + 1),
                2 ** 14 - 1,
                dtype=torch.long,
                device=self.device2
            )
        token_indices = torch.arange(IMAGE_TOKEN_COUNT, device=self.device2)
        temperature = 1,
        top_k = 256,
        supercondition_factor = 1,
        settings = torch.tensor(
            [temperature, top_k, supercondition_factor],
            dtype=torch.float32,
            device=self.device2
        )

        if self.condition:
            # print("Image shape:", images.shape)
            with torch.no_grad():
                with autocast(enabled=False):
                    assert condition_state is not None
                    _, _, [_, _, condition_state] = self.vqgan_tokenizer(condition_state)
                    condition_state = condition_state.detach().to(self.device2)
                    if condition_state is not None:
                        condition_state = condition_state.view(condition_state.shape[0], -1)
                    condition_state = self.decoder.embed_tokens.forward(condition_state)
                    B_c, N_c = condition_state.shape[:2]
                    pos_enc_tokens = torch.arange(N_c, device=condition_state.device).repeat((B_c, 1))
                    condition_state += self.decoder.embed_positions.forward(pos_enc_tokens)
                    condition_state = condition_state.repeat_interleave(int(image_count / B_c), dim=0)
        else:
            condition_state = None

        # print(settings.device, attention_mask.device,
        #       encoder_state.device, image_tokens.device,
        #       token_indices.device)
        for i in range(IMAGE_TOKEN_COUNT):
            torch.cuda.empty_cache()
            with torch.cuda.amp.autocast(dtype=self.dtype):
                image_tokens[:, i + 1], attention_state = self.decoder.sample_tokens(
                    settings=settings,
                    attention_mask=attention_mask,
                    encoder_state=encoder_state,
                    attention_state=attention_state,
                    # prev_tokens=image_tokens[:, :i+1],
                    # token_index=token_indices[:i+1]
                    prev_tokens=image_tokens[:, [i]],
                    token_index=token_indices[[i]],
                    condition_state=condition_state
                )

        # print(image_tokens.shape, attention_state.shape)

        images = self.detokenizer.forward(False, image_tokens[:, 1:].to(self.device1))
        return images