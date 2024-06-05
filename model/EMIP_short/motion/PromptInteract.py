import torch
import torch.nn as nn
import torch.nn.functional as F
from model.EPFlow_1_feature.motion.common import LayerNorm2d
from typing import List, Tuple, Type, Optional, Any
from model.EPFlow_1_feature.motion.transformer import TwoWayTransformer
from einops import rearrange
import numpy as np
from timm.models.layers import to_2tuple
import numbers

class PromptInteract(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.num_mask_tokens = args['update']['num_mask_tokens']
        # self.num_motion_tokens = args['update']['num_motion_tokens']
        self.transformer_dim = args['update']['transformer_dim']
        self.motion_embed_dim = args['update']['motion_embed_dim']
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, self.transformer_dim)
        self.motion_tokens = nn.Parameter(torch.zeros(self.transformer_dim))

        self.prompt_embed_dim = args['update']['prompt_embed_dim']
        self.transformer = TwoWayTransformer(depth=2,
                embedding_dim=self.prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            )

        self.pe_layer = PositionEmbeddingRandom(self.prompt_embed_dim // 2)
        self.inp_size = args['inp_size']
        self.image_embedding_size = self.inp_size // args['update']['patch_size']

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(self.transformer_dim, self.transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(self.transformer_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(self.transformer_dim // 4, self.transformer_dim // 8, kernel_size=2, stride=2),
            nn.GELU(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(self.transformer_dim, self.transformer_dim, self.transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )
        self.flow_head = MLP(
            self.transformer_dim, args['update']['flow_head_hidden_dim'], self.num_mask_tokens, args['update']['flow_head_depth']
        )
        self.mask_in_chans = args['update']['mask_in_chans']
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(4, self.mask_in_chans // 4, kernel_size=2, stride=2),  # 1-2
            LayerNorm2d(self.mask_in_chans // 4),
            nn.GELU(),
            nn.Conv2d(self.mask_in_chans // 4, self.mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(self.mask_in_chans),
            nn.GELU(),
            nn.Conv2d(self.mask_in_chans, self.prompt_embed_dim, kernel_size=2, stride=2),  # 修改kernel/stride 1-2
        )
        self.PatchEmbed = PatchEmbed(img_size=44, patch_size=8, in_chans=128, embed_dim=128)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)


    def forward(self, image_embeddings, flow):  #image_embeddings和flow都是(bs,128,44,44)
        output_tokens = self.mask_tokens.weight  # (4,128)
        flow_embeddings = self.PatchEmbed(flow)  # bs,25,128
        # motion_tokens = self.motion_tokens.unsqueeze(0).expand(flow.size(0), -1, -1) # bs,1,128
        # motion_tokens = torch.cat([motion_tokens, flow_embeddings], dim=1)

        # motion_embeddings = rearrange(motion_embeddings, 'b c h w -> b (h w) c')  # 2 3872 128
        # output_tokens = output_tokens.unsqueeze(0).expand(motion_tokens.size(0), -1, -1)  # bs 4 128
        # tokens = torch.cat((output_tokens, motion_tokens), dim=1)  # bs 30 128
        output_tokens = output_tokens.unsqueeze(0).expand(flow_embeddings.size(0), -1, -1)  # bs 4 128
        tokens = torch.cat((output_tokens, flow_embeddings), dim=1)  # 10-24
        # tokens = flow_embeddings
        image_pe = self.pe_layer(self.image_embedding_size).unsqueeze(0)  # 1 128 44 44

        src = image_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)  # bs 128 44 44
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)  # return processed point_embedding, processed image_embedding 2 3878 128/ 2 1936 128 , return query and keys
        # hs, src = self.transformer(src, pos_src, flow_embeddings)
        mask_tokens_out = hs[:, :self.num_mask_tokens, :]  # 2 4 128
        # flow_src = hs[:, self.num_mask_tokens:, :]  # 2 3872 128
        # flow_src = flow_src.transpose(1, 2).view(b, c, h, w)  # 2 128 44 44


        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)  #(bs,128,44,44)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w) #(bs,4,176,176)
        masks = self.mask_downscaling(masks) # (bs,128,22,22)
        masks = self.upsample(masks)
        return masks

        #
        # src = src.transpose(1, 2).view(b, c, h, w)  # bs 128 44 44
        # return src


class Interact(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.num_mask_tokens = args['update']['num_mask_tokens']
        # self.num_motion_tokens = args['update']['num_motion_tokens']
        self.transformer_dim = args['update']['transformer_dim']
        self.motion_embed_dim = args['update']['motion_embed_dim']
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, self.transformer_dim)
        self.motion_tokens = nn.Parameter(torch.zeros(self.transformer_dim))
        self.flow_tokens = nn.Embedding(2, self.transformer_dim)

        self.prompt_embed_dim = args['update']['prompt_embed_dim']
        self.transformer = TwoWayTransformer(depth=1,
                embedding_dim=self.prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            )

        self.pe_layer = PositionEmbeddingRandom(self.prompt_embed_dim // 2)
        self.inp_size = args['inp_size']
        self.image_embedding_size = self.inp_size // args['update']['patch_size']

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(self.transformer_dim, self.transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(self.transformer_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(self.transformer_dim // 4, self.transformer_dim // 8, kernel_size=2, stride=2),
            nn.GELU(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(self.transformer_dim, self.transformer_dim, self.transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )
        self.flow_head = MLP(
            self.transformer_dim, args['update']['flow_head_hidden_dim'], self.num_mask_tokens, args['update']['flow_head_depth']
        )
        self.mask_in_chans = args['update']['mask_in_chans']
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(2, self.mask_in_chans // 4, kernel_size=2, stride=2),  # 1-2
            LayerNorm2d(self.mask_in_chans // 4),
            nn.GELU(),
            nn.Conv2d(self.mask_in_chans // 4, self.mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(self.mask_in_chans),
            nn.GELU(),
            nn.Conv2d(self.mask_in_chans, self.prompt_embed_dim, kernel_size=2, stride=2),  # 修改kernel/stride 1-2
        )
        self.PatchEmbed = PatchEmbed(img_size=44, patch_size=8, in_chans=128, embed_dim=128)


    def forward(self, image_embeddings, flow):  #image_embeddings和flow都是(bs,128,44,44)
        flow_embeddings = self.PatchEmbed(flow)  # bs,25,128
        image_pe = self.pe_layer(self.image_embedding_size).unsqueeze(0)  # 1 128 44 44

        src = image_embeddings # (bs,128,44,44)
        pos_src = torch.repeat_interleave(image_pe, flow_embeddings.shape[0], dim=0)  # bs 128 44 44
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, flow_embeddings)  # return processed point_embedding, processed image_embedding 2 3878 128/ 2 1936 128
        mask_tokens_out = hs[:, :self.num_mask_tokens, :]  # 2 4 128
        # flow_src = hs[:, self.num_mask_tokens:, :]  # 2 3872 128
        # flow_src = flow_src.transpose(1, 2).view(b, c, h, w)  # 2 128 44 44

        src = src.transpose(1, 2).view(b, c, h, w)  # bs 128 44 44
        return src

# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: int) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size, size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # allow different input size
        # B, C, H, W = x.shape
        # _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        # _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


# zhangxin 2023-9-30
##########################################################################
##---------- Prompt Gen Module -----------------------
class PromptGenBlock(nn.Module):
    def __init__(self, prompt_dim=128, prompt_len=5, prompt_size=96, lin_dim=192):
        super(PromptGenBlock, self).__init__()
        self.prompt_param = nn.Parameter(torch.rand(1, prompt_len, prompt_dim, prompt_size, prompt_size)) # (1,5,320,16,16)
        self.linear_layer = nn.Linear(lin_dim, prompt_len) # 384->5
        self.conv3x3 = nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, stride=1, padding=1, bias=False) # 320->320

    def forward(self, x): # (bs,384,16,16)
        B, C, H, W = x.shape
        emb = x.mean(dim=(-2, -1)) # (bs,384)
        prompt_weights = F.softmax(self.linear_layer(emb), dim=1) # (bs,5)
        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.prompt_param.unsqueeze(0).repeat(B, 1,
                                                                                                                  1, 1,
                                                                                                                  1,
                                                                                                                  1).squeeze(
            1) # (bs,5,320,16,16)
        prompt = torch.sum(prompt, dim=1) # (bs,320,16,16)
        prompt = F.interpolate(prompt, (H, W), mode="bilinear") # (bs,320,16,16)
        prompt = self.conv3x3(prompt) # (bs,320,16,16)

        return prompt

##########################################################################
# 2023-10-31
##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention_MDTA(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention_MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # 原始代码
        # self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        # self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, x1):
        b, c, h, w = x.shape

        # qkv = self.qkv_dwconv(self.qkv(x))
        # q, k, v = qkv.chunk(3, dim=1)

        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(x1))
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


## Transformer Block
class TransformerBlock_MDTA(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock_MDTA, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention_MDTA(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.norm3 = LayerNorm(dim, LayerNorm_type)

    def forward(self, x, x1):
        x = x + self.attn(self.norm1(x), self.norm2(x1))
        x = x + self.ffn(self.norm3(x))

        return x

class Injector(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        # self.transformer = nn.Sequential(*[
        #     TransformerBlock_MDTA(dim=int(128), num_heads=2, ffn_expansion_factor=2.66,
        #                      bias=False, LayerNorm_type='WithBias') for i in range(2)])
        self.transformer = TransformerBlock_MDTA(dim=int(128), num_heads=2, ffn_expansion_factor=2.66,
                             bias=False, LayerNorm_type='WithBias')


    def forward(self, image_embeddings, flow):  #image_embeddings和flow都是(bs,128,44,44)
        src = self.transformer(image_embeddings, flow)
        return src