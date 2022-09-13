import torch
from torch import nn
from torch import FloatTensor, LongTensor
from math import sqrt
from einops import rearrange

class ResnetBlock(nn.Module):
    def __init__(self, log2_count_in: int, log2_count_out: int):
        super().__init__()
        m, n = 2 ** log2_count_in, 2 ** log2_count_out
        self.is_middle = m == n
        self.norm1 = nn.GroupNorm(2 ** 5, m)
        self.conv1 = nn.Conv2d(m, n, 3, padding=1)
        self.norm2 = nn.GroupNorm(2 ** 5, n)
        self.conv2 = nn.Conv2d(n, n, 3, padding=1)
        if not self.is_middle:
            self.nin_shortcut = nn.Conv2d(m, n, 1)

    def forward(self, x: FloatTensor) -> FloatTensor:
        h = x
        h = self.norm1.forward(h)
        h *= torch.sigmoid(h)
        h = self.conv1.forward(h)
        h = self.norm2.forward(h)
        h *= torch.sigmoid(h)
        h = self.conv2(h)
        if not self.is_middle:
            x = self.nin_shortcut.forward(x)
        return x + h


class AttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        n = 2 ** 9
        self.norm = nn.GroupNorm(2 ** 5, n)
        self.q = nn.Conv2d(n, n, 1)
        self.k = nn.Conv2d(n, n, 1)
        self.v = nn.Conv2d(n, n, 1)
        self.proj_out = nn.Conv2d(n, n, 1)

    def forward(self, x: FloatTensor) -> FloatTensor:
        n, m = 2 ** 9, x.shape[0]
        h = x
        h = self.norm(h)
        k = self.k.forward(h)
        v = self.v.forward(h)
        q = self.q.forward(h)
        k = k.reshape(m, n, -1)
        v = v.reshape(m, n, -1)
        q = q.reshape(m, n, -1)
        q = q.permute(0, 2, 1)
        w = torch.bmm(q, k)
        w /= n ** 0.5
        w = torch.softmax(w, dim=2)
        w = w.permute(0, 2, 1)
        h = torch.bmm(v, w)
        token_count = int(sqrt(h.shape[-1]))
        h = h.reshape(m, n, token_count, token_count)
        h = self.proj_out.forward(h)
        return x + h


class MiddleLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.block_1 = ResnetBlock(9, 9)
        self.attn_1 = AttentionBlock()
        self.block_2 = ResnetBlock(9, 9)
    
    def forward(self, h: FloatTensor) -> FloatTensor:
        h = self.block_1.forward(h)
        h = self.attn_1.forward(h)
        h = self.block_2.forward(h)
        return h


class Upsample(nn.Module):
    def __init__(self, log2_count):
        super().__init__()
        n = 2 ** log2_count
        self.upsample = torch.nn.UpsamplingNearest2d(scale_factor=2)
        self.conv = nn.Conv2d(n, n, 3, padding=1)

    def forward(self, x: FloatTensor) -> FloatTensor:
        x = self.upsample.forward(x.to(torch.float32))
        x = self.conv.forward(x)
        return x


class Downsample(nn.Module):
    def __init__(self, log2_count):
        super().__init__()
        n = 2 ** log2_count
        self.conv = torch.nn.Conv2d(n, n, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        pad = (0,1,0,1)
        x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class DownsampleBlock(nn.Module):
    def __init__(
            self,
            log2_count_in: int,
            log2_count_out: int,
            has_attention: bool,
            has_downsample: bool
    ):
        super().__init__()
        self.has_attention = has_attention
        self.has_downsample = has_downsample

        self.block = nn.ModuleList([
            ResnetBlock(log2_count_in, log2_count_out),
            ResnetBlock(log2_count_out, log2_count_out),
        ])

        if has_attention:
            self.attn = nn.ModuleList([
                AttentionBlock(),
                AttentionBlock(),
            ])

        if has_downsample:
            self.downsample = Downsample(log2_count_out)

    def forward(self, h: FloatTensor) -> FloatTensor:
        for j in range(2):
            h = self.block[j].forward(h)
            if self.has_attention:
                h = self.attn[j].forward(h)
        if self.has_downsample:
            h = self.downsample.forward(h)
        return h



class UpsampleBlock(nn.Module):
    def __init__(
        self, 
        log2_count_in: int, 
        log2_count_out: int, 
        has_attention: bool, 
        has_upsample: bool
    ):
        super().__init__()
        self.has_attention = has_attention
        self.has_upsample = has_upsample
        
        self.block = nn.ModuleList([
            ResnetBlock(log2_count_in, log2_count_out),
            ResnetBlock(log2_count_out, log2_count_out),
            ResnetBlock(log2_count_out, log2_count_out),
        ])

        if has_attention:
            self.attn = nn.ModuleList([
                AttentionBlock(),
                AttentionBlock(),
                AttentionBlock()
            ])

        if has_upsample:
            self.upsample = Upsample(log2_count_out)


    def forward(self, h: FloatTensor) -> FloatTensor:
        for j in range(3):
            h = self.block[j].forward(h)
            if self.has_attention:
                h = self.attn[j].forward(h)
        if self.has_upsample:
            h = self.upsample.forward(h)
        return h


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        # downsampling
        self.conv_in = torch.nn.Conv2d(3, 2 ** 7, 3, stride=1, padding=1)

        self.down = nn.ModuleList([
            DownsampleBlock(7, 7, False, True),
            DownsampleBlock(7, 7, False, True),
            DownsampleBlock(7, 8, False, True),
            DownsampleBlock(8, 8, False, True),
            DownsampleBlock(8, 9, True, False),
        ])

        # middle
        self.mid = MiddleLayer()

        # end
        self.norm_out = nn.GroupNorm(2 ** 5, 2 ** 9)
        self.conv_out = nn.Conv2d(2 ** 9, 2 ** 8, 3, padding=1)

    def forward(self, z: FloatTensor) -> FloatTensor:
        z = self.conv_in.forward(z)
        for i in range(5):
            z = self.down[i].forward(z)
        z = self.mid.forward(z)

        z = self.norm_out.forward(z)
        z *= torch.sigmoid(z)
        z = self.conv_out.forward(z)
        return z


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_in = nn.Conv2d(2 ** 8, 2 ** 9, 3, padding=1)
        self.mid = MiddleLayer()

        self.up = nn.ModuleList([
            UpsampleBlock(7, 7, False, False),
            UpsampleBlock(8, 7, False, True),
            UpsampleBlock(8, 8, False, True),
            UpsampleBlock(9, 8, False, True),
            UpsampleBlock(9, 9, True, True)
        ])

        self.norm_out = nn.GroupNorm(2 ** 5, 2 ** 7)
        self.conv_out = nn.Conv2d(2 ** 7, 3, 3, padding=1)

    def forward(self, z: FloatTensor) -> FloatTensor:
        z = self.conv_in.forward(z)
        z = self.mid.forward(z)

        for i in reversed(range(5)):
            z = self.up[i].forward(z)

        z = self.norm_out.forward(z)
        z *= torch.sigmoid(z)
        z = self.conv_out.forward(z)
        return z

class VectorQuantizer2(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """
    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(self, n_e=16384, e_dim=256, beta=0.25,
                 sane_index_shape=False, legacy=True):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.re_embed = n_e
        self.remap = None

        self.sane_index_shape = sane_index_shape

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        match = (inds[:,:,None]==used[None,None,...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2)<1
        if self.unknown_index == "random":
            new[unknown]=torch.randint(0,self.re_embed,size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]: # extra token
            inds[inds>=self.used.shape[0]] = 0 # simply set to zero
        back=torch.gather(used[None,:][inds.shape[0]*[0],:], 1, inds)
        return back.reshape(ishape)

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach()-z)**2) + \
                   torch.mean((z_q - z.detach()) ** 2)
        else:
            loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
                   torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0],-1) # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1,1) # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_q.shape[0], z_q.shape[2], z_q.shape[3])

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            indices = indices.reshape(shape[0],-1) # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1) # flatten again

        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q


class VQGanDetokenizer(nn.Module):
    def __init__(self):
        super().__init__()
        vocab_count, embed_count = 2 ** 14, 2 ** 8
        self.vocab_count = vocab_count
        self.embedding = nn.Embedding(vocab_count, embed_count)
        self.post_quant_conv = nn.Conv2d(embed_count, embed_count, 1)
        self.decoder = Decoder()

    def forward(self, is_seamless: bool, z: LongTensor, grid: bool = False) -> FloatTensor:
        image_count = z.shape[0]
        grid_size = int(sqrt(z.shape[0]))
        token_count = grid_size * 2 ** 4
        
        if is_seamless:
            z = z.view([grid_size, grid_size, 2 ** 4, 2 ** 4])
            z = z.flatten(1, 2).transpose(1, 0).flatten(1, 2)
            z = z.flatten().unsqueeze(1)
            z = self.embedding.forward(z)
            z = z.view((1, token_count, token_count, 2 ** 8))
        else:
            z = self.embedding.forward(z)
            z = z.view((z.shape[0], 2 ** 4, 2 ** 4, 2 ** 8))

        z = z.permute(0, 3, 1, 2).contiguous()
        z = self.post_quant_conv.forward(z)
        z = self.decoder.forward(z)
        z = z.permute(0, 2, 3, 1)
        z = z.clip(0.0, 1.0) * 255

        if is_seamless:
            z = z[0]
        else:
            if grid:
                z = z.view([grid_size, grid_size, 2 ** 8, 2 ** 8, 3])
                z = z.flatten(1, 2).transpose(1, 0).flatten(1, 2)
            else:
                z = z.view([image_count, 2 ** 8, 2 ** 8, 3])

        return z


class VQGanTokenizer(nn.Module):
    def __init__(self):
        super().__init__()
        vocab_count, embed_count = 2 ** 14, 2 ** 8
        self.vocab_count = vocab_count
        self.quant_conv = nn.Conv2d(embed_count, embed_count, 1)
        self.encoder = Encoder()
        self.quantize = VectorQuantizer2(sane_index_shape=True)

    def forward(self, x: LongTensor):

        h = self.encoder(x)
        # print('VQGan encoder output shape', h.shape)
        h = self.quant_conv(h)
        # print("Post-quant shape", h.shape)
        quant, emb_loss, info = self.quantize(h)
        # print(quant.shape, info[-1].shape)
        return quant, emb_loss, info