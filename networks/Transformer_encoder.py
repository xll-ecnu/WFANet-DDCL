import math

import pywt
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat
from .utils import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, head_num):
        super().__init__()

        self.head_num = head_num
        self.dk = (embedding_dim // head_num) ** (1 / 2) #（768//12）**(1/2) = 8

        self.qkv_layer = nn.Linear(embedding_dim, embedding_dim * 3, bias=False)
        self.out_attention = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, x, mask=None):
        #x = torch.tensor(x)
        print(x.shape)
        qkv = self.qkv_layer(x)
        #print('x.shape af qkv_layer', qkv.shape)
        query, key, value = tuple(rearrange(qkv, 'b t (d k h ) -> k b h t d ', k=3, h=self.head_num))
        #print('query.shape ', query.shape,'key.shape',key.shape,'value.shape',value.shape)
        q_multiply_k = torch.einsum("... i d , ... j d -> ... i j", query, key)
        energy = q_multiply_k * self.dk
        #print('q_multiply_k.shape', q_multiply_k.shape)
        if mask is not None:
            energy = energy.masked_fill(mask, -np.inf)

        attention = torch.softmax(energy, dim=-1)
        #print('attention.shape', attention.shape)
        at_multiply_v = torch.einsum("... i j , ... j d -> ... i d", attention, value)
        #print('at_multiply_v.shape', at_multiply_v.shape)
        x = rearrange(at_multiply_v, "b h t d -> b t (h d)")
        #print('x.shape af rearrange', x.shape)
        x = self.out_attention(x)

        return x


class MLP(nn.Module):
    def __init__(self, embedding_dim, mlp_dim, is_wave = False):
        super().__init__()

        self.mlp_layers = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_dim, embedding_dim),
            nn.Dropout(0.1)
        )
        self.is_wave = is_wave

    def forward(self, x):
        #print('x.shape bf mlp_layers',x.shape)
        x = self.mlp_layers(x)
        if self.is_wave:
            x = rearrange(x, 'b x y d -> b d x y'
                                    )
        #print('x.shape af mlp_layers', x.shape)
        return x
class TransformerEncoderBlock(nn.Module):
    def __init__(self, embedding_dim, head_num, mlp_dim):
        super().__init__()

        self.multi_head_attention = MultiHeadAttention(embedding_dim, head_num)
        self.mlp = MLP(embedding_dim, mlp_dim)

        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        _x = self.multi_head_attention(x)
        _x = self.dropout(_x)
        x = x + _x
        x = self.layer_norm1(x)

        _x = self.mlp(x)
        x = x + _x
        x = self.layer_norm2(x)

        return x



class TransformerEncoder(nn.Module):
    def __init__(self, embedding_dim, head_num, mlp_dim, block_num, z_idx_list):
        super().__init__()
        self.z_idx_list = z_idx_list
        layers = []
        for _ in range(block_num):
            layers.append(WFAE(embedding_dim, mlp_dim))
            layers.append(TransformerEncoderBlock(embedding_dim, head_num, mlp_dim))


        self.layer_blocks = nn.ModuleList(layers)
        # self.layer_blocks = nn.ModuleList(
        #     [TransformerEncoderBlock(embedding_dim, head_num, mlp_dim) for _ in range(block_num)])

    def forward(self, x):

        z_outputs = []
        for idx, layer_block in enumerate(self.layer_blocks, start=1):
            x = layer_block(x)
            if idx in self.z_idx_list:
                #print('x.shape in {}'.format(idx),x.shape)
                z_outputs.append(x)

        return z_outputs


class AbsPositionalEncoding1D(nn.Module):
    def __init__(self, tokens, dim):
        super().__init__()
        self.abs_pos_enc = nn.Parameter(torch.randn(1, tokens, dim))

    def forward(self, x):
        #print('self.abs_pos_enc.shape',self.abs_pos_enc.shape)
        batch = x.size()[0]
        tile = batch // self.abs_pos_enc.shape[0]
        #print('tile', tile)
        repeat1 = repeat(self.abs_pos_enc, 'b ... -> (b tile) ...', tile=batch // self.abs_pos_enc.shape[0])
        #print('repeat1.shape', repeat1.shape)
        out = x + repeat1
        return out


class Transformer2D(nn.Module):
    def __init__(self, img_dim, in_channels, embedding_dim, head_num, mlp_dim,
                 block_num, patch_size, z_idx_list):
        super().__init__()

        self.patch_size = patch_size
        self.n_patches = int((img_dim[0] * img_dim[1]) / (patch_size ** 2))

        self.patch_embeddings = nn.Conv2d(in_channels, embedding_dim,
                                          kernel_size=patch_size, stride=patch_size, bias=False)

        self.position_embeddings = AbsPositionalEncoding1D(self.n_patches, embedding_dim)
        self.dropout = nn.Dropout(0.1)

        self.transformer = TransformerEncoder(embedding_dim, head_num, mlp_dim, block_num, z_idx_list)

    def forward(self, x):
        patch_embeded = self.patch_embeddings(x)
        #print('patch_embeded.shape',patch_embeded.shape)
        #patch_embeded1 = rearrange(patch_embeded, 'b d x y -> b x y d')
        embeddings1 = rearrange(patch_embeded, 'b d x y -> b (x y) d')
        #print('embeddings1.shape', embeddings1.shape)
        embeddings2 = self.position_embeddings(embeddings1)
        #print('embeddings2.shape', embeddings2.shape)
        embeddings3 = self.dropout(embeddings2)
        #print('embeddings3.shape', embeddings3.shape)
        z_outputs = self.transformer(embeddings3)
        #print('z_outputs.size', z_outputs.size)
        return z_outputs





if __name__ == '__main__':
    trans = Transformer2D(img_dim=(224, 224),
                          in_channels=1,
                          patch_size=16,
                          embedding_dim=768,
                          block_num=12,
                          head_num=12,
                          mlp_dim=3072,
                          z_idx_list=[3, 6, 9, 12])

