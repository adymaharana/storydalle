from typing import Tuple, List
import torch
from torch import nn, LongTensor, FloatTensor, BoolTensor
from .dalle_bart_encoder import GLU, AttentionBase

IMAGE_TOKEN_COUNT = 256


class DecoderCrossAttention(AttentionBase):
    def forward(
        self,
        decoder_state: FloatTensor,
        encoder_state: FloatTensor,
        attention_mask: BoolTensor
    ) -> FloatTensor:
        keys = self.k_proj.forward(encoder_state)
        values = self.v_proj.forward(encoder_state)
        queries = self.q_proj.forward(decoder_state)
        return super().forward(keys, values, queries, attention_mask)


class DecoderSelfAttention(AttentionBase):
    def __init__(self, head_count: int, embed_count: int):
        super().__init__(head_count, embed_count)

    def forward(
        self, 
        decoder_state: FloatTensor,
        attention_state: FloatTensor,
        attention_mask: BoolTensor,
        token_index: LongTensor = None
    ) -> Tuple[FloatTensor, FloatTensor]:
        keys = self.k_proj.forward(decoder_state)
        values = self.v_proj.forward(decoder_state)
        queries = self.q_proj.forward(decoder_state)

        # TODO: refer to the cache process in minDALLE to fix this
        if token_index is not None:
            token_count = token_index.shape[1]
            if token_count == 1:
                batch_count = decoder_state.shape[0]
                attn_state_new = torch.cat([keys, values]).to(attention_state.dtype)
                attention_state[:, token_index[0]] = attn_state_new
                keys = attention_state[:batch_count]
                values = attention_state[batch_count:]

        decoder_state = super().forward(keys, values, queries, attention_mask)
        return decoder_state, attention_state


class DecoderLayer(nn.Module):
    def __init__(
        self, 
        head_count: int, 
        embed_count: int,
        glu_embed_count: int,
        device: str,
        condition: bool = False
    ):
        super().__init__()
        self.pre_self_attn_layer_norm = nn.LayerNorm(embed_count)
        self.self_attn = DecoderSelfAttention(head_count, embed_count)
        self.self_attn_layer_norm = nn.LayerNorm(embed_count)
        self.pre_encoder_attn_layer_norm = nn.LayerNorm(embed_count)
        self.encoder_attn = DecoderCrossAttention(head_count, embed_count)
        self.encoder_attn_layer_norm = nn.LayerNorm(embed_count)
        self.glu = GLU(embed_count, glu_embed_count)
        self.token_indices = torch.arange(IMAGE_TOKEN_COUNT, device=device)
        self.head_count = head_count

        self.condition = condition
        if condition:
            self.pre_condition_attn_layer_norm = nn.LayerNorm(embed_count)
            self.condition_attn = DecoderCrossAttention(head_count, embed_count)
            self.condition_attn_layer_norm = nn.LayerNorm(embed_count)

    def sample(
            self,
            decoder_state: FloatTensor,
            encoder_state: FloatTensor,
            attention_state: FloatTensor,
            attention_mask: BoolTensor,
            token_index: LongTensor,
            condition_state: FloatTensor = None
    ) -> Tuple[FloatTensor, FloatTensor]:
        # Self Attention
        token_count = token_index.shape[1]
        if token_count == 1:
            # print(self.token_indices.device, token_index.device)
            self_attn_mask = self.token_indices <= token_index
            self_attn_mask = self_attn_mask[:, None, None, :]
        else:
            self_attn_mask = (
                    self.token_indices[None, None, :token_count] <=
                    token_index[:, :, None]
            )
            self_attn_mask = self_attn_mask[:, None, :, :]

        # TODO: Fix self-attention mask

        residual = decoder_state
        decoder_state = self.pre_self_attn_layer_norm.forward(decoder_state)
        decoder_state, attention_state = self.self_attn.forward(
            decoder_state=decoder_state,
            attention_state=attention_state,
            attention_mask=self_attn_mask,
            token_index=token_index
        )
        decoder_state = self.self_attn_layer_norm.forward(decoder_state)
        decoder_state = residual + decoder_state

        # Cross Attention
        residual = decoder_state
        decoder_state = self.pre_encoder_attn_layer_norm.forward(decoder_state)
        decoder_state = self.encoder_attn.forward(
            decoder_state=decoder_state,
            encoder_state=encoder_state,
            attention_mask=attention_mask
        )
        decoder_state = self.encoder_attn_layer_norm.forward(decoder_state)
        decoder_state = residual + decoder_state

        # Cross-Attention Over Image Condition
        if self.condition:
            assert condition_state is not None
            residual = decoder_state
            decoder_state = self.pre_condition_attn_layer_norm.forward(decoder_state)
            decoder_state = self.condition_attn.forward(
                decoder_state=decoder_state,
                encoder_state=encoder_state,
                attention_mask=attention_mask
            )
            decoder_state = self.condition_attn_layer_norm.forward(decoder_state)
            decoder_state = residual + decoder_state

        # Feed forward
        residual = decoder_state
        decoder_state = self.glu.forward(decoder_state)
        decoder_state = residual + decoder_state

        return decoder_state, attention_state


    def forward(
        self,
        decoder_state: FloatTensor,
        encoder_state: FloatTensor,
        attention_state: FloatTensor,
        attention_mask: BoolTensor,
        condition_state: FloatTensor = None
    ) -> Tuple[FloatTensor, FloatTensor]:
        # Self Attention
        # token_count = token_index.shape[1]
        # if token_count == 1:
        #     self_attn_mask = self.token_indices <= token_index
        #     self_attn_mask = self_attn_mask[:, None, None, :]
        # else:
        #     self_attn_mask = (
        #         self.token_indices[None, None, :token_count] <=
        #         token_index[:, :, None]
        #     )
        #     self_attn_mask = self_attn_mask[:, None, :, :]

        # TODO: Fix self-attention mask
        B, N = decoder_state.shape[:2]
        self_attn_mask = torch.tril(torch.ones(size=(N, N), device=decoder_state.device)).view(1, 1, N, N).repeat(B, self.head_count, 1, 1)
        # print("Self-attention mask shape: ", self_attn_mask.shape)
        
        residual = decoder_state
        decoder_state = self.pre_self_attn_layer_norm.forward(decoder_state)
        decoder_state, attention_state = self.self_attn.forward(
            decoder_state=decoder_state,
            attention_state=attention_state,
            attention_mask=self_attn_mask,
            # token_index=token_index
        )
        decoder_state = self.self_attn_layer_norm.forward(decoder_state)
        decoder_state = residual + decoder_state

        # Cross Attention
        residual = decoder_state
        decoder_state = self.pre_encoder_attn_layer_norm.forward(decoder_state)
        decoder_state = self.encoder_attn.forward(
            decoder_state=decoder_state,
            encoder_state=encoder_state,
            attention_mask=attention_mask
        )
        decoder_state = self.encoder_attn_layer_norm.forward(decoder_state)
        decoder_state = residual + decoder_state

        # Cross-Attention Over Image Condition
        if self.condition:
            assert condition_state is not None
            residual = decoder_state
            decoder_state = self.pre_condition_attn_layer_norm.forward(decoder_state)
            decoder_state = self.condition_attn.forward(
                decoder_state=decoder_state,
                encoder_state=encoder_state,
                attention_mask=attention_mask
            )
            decoder_state = self.condition_attn_layer_norm.forward(decoder_state)
            decoder_state = residual + decoder_state

        # Feed forward
        residual = decoder_state
        decoder_state = self.glu.forward(decoder_state)
        decoder_state = residual + decoder_state

        return decoder_state, attention_state


class DalleBartDecoder(nn.Module):
    def __init__(
        self,
        image_vocab_count: int,
        embed_count: int,
        attention_head_count: int,
        glu_embed_count: int,
        layer_count: int,
        device: str,
        condition: bool = False
    ):
        super().__init__()
        self.layer_count = layer_count
        self.embed_count = embed_count
        self.image_vocab_count = image_vocab_count
        self.embed_tokens = nn.Embedding(image_vocab_count + 1, embed_count)
        self.embed_positions = nn.Embedding(IMAGE_TOKEN_COUNT, embed_count)
        self.layers: List[DecoderLayer] = nn.ModuleList([
            DecoderLayer(
                head_count=attention_head_count,
                embed_count=embed_count,
                glu_embed_count=glu_embed_count,
                device=device,
                condition = (i+1)%3 == 0 if condition else False
            )
            for i in range(layer_count)
        ])
        self.condition = condition
        self.layernorm_embedding = nn.LayerNorm(embed_count)
        self.final_ln = nn.LayerNorm(embed_count)
        self.lm_head = nn.Linear(embed_count, image_vocab_count + 1, bias=False)
        self.token_indices = torch.arange(IMAGE_TOKEN_COUNT, device=device)

        if self.condition:
            print("Initialized %s condition attention layers" % sum([(i+1)%3 == 0 for i in range(layer_count)]))


    def forward(
        self,
        attention_mask: BoolTensor,
        encoder_state: FloatTensor,
        attention_state: FloatTensor,
        prev_tokens: LongTensor,
        condition_state: FloatTensor = None
    ) -> Tuple[FloatTensor, FloatTensor]:
        decoder_state = self.embed_tokens.forward(prev_tokens)
        B, N = prev_tokens.shape
        pos_enc_tokens = torch.arange(N, device=prev_tokens.device).repeat((B, 1))
        decoder_state += self.embed_positions.forward(pos_enc_tokens)
        decoder_state = self.layernorm_embedding.forward(decoder_state)

        if condition_state is not None:
            condition_state = self.embed_tokens.forward(condition_state)
            B_c, N_c = condition_state.shape[:2]
            pos_enc_tokens = torch.arange(N_c, device=condition_state.device).repeat((B_c, 1))
            # print(condition_state.shape, pos_enc_tokens.shape)
            condition_state += self.embed_positions.forward(pos_enc_tokens)
            # print(condition_state.shape)
            condition_state = condition_state.repeat_interleave(int(B/B_c), dim=0)
            # print(condition_state.shape)

        for i in range(self.layer_count):
            decoder_state, attention_state[i] = self.layers[i].forward(
                decoder_state,
                encoder_state,
                attention_state[i],
                attention_mask,
                condition_state=condition_state if self.condition and (i+1)%3 == 0 else None
            )
        decoder_state = self.final_ln(decoder_state)
        logits = self.lm_head(decoder_state)
        return logits, attention_state

    def sample(
        self,
        attention_mask: BoolTensor,
        encoder_state: FloatTensor,
        attention_state: FloatTensor,
        prev_tokens: LongTensor,
        token_index: LongTensor,
        condition_state: FloatTensor = None,
        supercondition: bool = False
    ) -> Tuple[FloatTensor, FloatTensor]:
        image_count = encoder_state.shape[0] // 2
        token_index = token_index.unsqueeze(0).repeat(image_count * 2, 1)
        if supercondition:
            prev_tokens = prev_tokens.repeat(2, 1)
        decoder_state = self.embed_tokens.forward(prev_tokens)
        decoder_state += self.embed_positions.forward(token_index)
        decoder_state = self.layernorm_embedding.forward(decoder_state)
        for i in range(self.layer_count):
            decoder_state, attention_state[i] = self.layers[i].sample(
                decoder_state,
                encoder_state,
                attention_state[i],
                attention_mask,
                token_index,
                condition_state=condition_state if self.condition and (i + 1) % 3 == 0 else None
            )
        decoder_state = self.final_ln(decoder_state)
        logits = self.lm_head(decoder_state)
        return logits, attention_state


    def sample_tokens(self, settings, **kwargs) -> Tuple[LongTensor, FloatTensor]:
        logits, attention_state = self.sample(supercondition=settings[2] != 1, **kwargs)
        image_count = logits.shape[0] // 2
        temperature = settings[[0]]
        top_k = settings[[1]].to(torch.long)
        supercondition_factor = settings[[2]]

        logits = logits[:, -1, : 2 ** 14]
        if supercondition_factor != 1:
            logits: FloatTensor = (
                logits[:image_count] * (1 - supercondition_factor) +
                logits[image_count:] * supercondition_factor
            )
        else:
            # logits: FloatTensor = (
            #     logits[:image_count] * 0 +
            #     logits[image_count:] * 1
            # )
            # print(logits.shape)
            pass

        # print(logits.shape)
        logits_sorted, _ = logits.sort(descending=True)
        # print(logits_sorted.shape)
        is_kept = logits >= logits_sorted[:, top_k - 1]
        if len(is_kept.shape) == 3:
            is_kept = logits >= logits_sorted[:, [top_k - 1]]
            assert len(is_kept.shape) == 2
        # print(logits_sorted[:, [0]])
        logits -= logits_sorted[:, [0]]
        # print(logits.shape)
        logits /= temperature
        # print(logits.shape, temperature)
        logits.exp_()
        logits *= is_kept.to(torch.float32)
        image_tokens = torch.multinomial(logits, 1)[:, 0]
        return image_tokens, attention_state