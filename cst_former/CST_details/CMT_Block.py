import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from .CST_encoder import CST_attention
from .layers import LocalPerceptionUint, InvertedResidualFeedForward, Temp_attention, Spec_attention

class CMT_Layers(torch.nn.Module):
    def __init__(self, params, temp_embed_dim, ffn_ratio=4., drop_path_rate=0.):
        super().__init__()
        self.cmt_split = params['CMT_split']
        self.ch_attn_dca = params['ChAtten_DCA']
        self.ch_attn_ule = params['ChAtten_ULE']
        self.temp_embed_dim = temp_embed_dim
        self.ffn_ratio = ffn_ratio
        self.dim = params['nb_cnn2d_filt']

        self.channels = params['nb_channels']

        #if params['use_ngcc']:
        #    if params['use_mel']:
        #        self.channels = int(params['ngcc_out_channels'] * params['n_mics'] * (params['n_mics'] - 1) / 2 +  params['n_mics'])
        #    else:
        #        self.channels = int(params['ngcc_out_channels'] * params['n_mics'] * ( 1 + (params['n_mics'] - 1) / 2))
        #else:
        #    self.channels = channels

        self.norm1 = nn.LayerNorm(self.dim)
        self.LPU = LocalPerceptionUint(self.dim)
        self.IRFFN = InvertedResidualFeedForward(self.dim, self.ffn_ratio)

        if not self.cmt_split:
            self.cst_attention = CST_attention(temp_embed_dim=self.temp_embed_dim,params=params)
        elif self.cmt_split:
            self.spectral_atten = Spec_attention(temp_embed_dim=self.temp_embed_dim, params=params)
            self.temporal_atten = Temp_attention(temp_embed_dim=self.temp_embed_dim, params=params)
            self.norm2 = nn.LayerNorm(self.dim)
            self.LPU2 = LocalPerceptionUint(self.dim)
            self.IRFFN2 = InvertedResidualFeedForward(self.dim, self.ffn_ratio)

        self.drop_path = nn.Dropout(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        if not self.cmt_split:
            if not self.ch_attn_dca: # ch_attn (ULE) or dst_attn
                lpu = self.LPU(x)
                x = x + lpu

                B, C, T, F = x.size()
                x = rearrange(x, 'b c t f -> b t (f c)')
                if not self.ch_attn_dca: # channel attention with unfolding
                    x = self.cst_attention(x, self.channels, C, T, F)
                else:   # dst attention
                    x = self.cst_attention(x,C,T,F)

                x_2 = rearrange(x, 'b t (f c) -> b (t f) c', f=F).contiguous()
                x_res = rearrange(x, 'b t (f c) -> b c t f', f=F).contiguous()
                norm1 = self.norm1(x_2)
                norm1 = rearrange(norm1, 'b (t f) c -> b c t f', f=F).contiguous()
                ffn = self.IRFFN(norm1)
                x = x_res + self.drop_path(ffn)

            if self.ch_attn_dca: # ch_attn (DCA)
                B, C, T, F = x.size()
                M = self.channels

                lpu = self.LPU(x)
                x = x + lpu

                x = rearrange(x, '(b m) c t f -> b t (m f c)', m=M).contiguous()
                x = self.cst_attention(x, M, C, T, F)

                x_2 = rearrange(x, 'b t (m f c) -> b (t m f) c', m=M, f=F).contiguous()
                x_res = rearrange(x, 'b t (m f c) -> (b m) c t f', m=M, f=F).contiguous()
                norm1 = self.norm1(x_2)
                norm1 = rearrange(norm1, 'b (t m f) c -> (b m) c t f', f=F, c=C, t=T).contiguous()
                ffn = self.IRFFN(norm1)
                x = x_res + self.drop_path(ffn)

        else: # CMT Split
            if not self.ch_attn_dca and not self.ch_attn_ule:
                # Spectral Conformer
                lpu = self.LPU(x)
                x = x + lpu

                B, C, T, F = x.size()
                x = rearrange(x, 'b c t f -> b t (f c)').contiguous()
                x = self.spectral_atten(x, C, T, F)

                x_s = rearrange(x, 'b t (f c) -> b (t f) c', f=F).contiguous()
                x_res = rearrange(x, 'b t (f c) -> b c t f', f=F).contiguous()
                norm1 = self.norm1(x_s)
                norm1 = rearrange(norm1, 'b (t f) c -> b c t f', f=F).contiguous()
                ffn_s = self.IRFFN(norm1)
                xs = x_res + self.drop_path(ffn_s)

                # Temporal Conformer
                lpu2 = self.LPU2(xs)
                xs = xs + lpu2

                B, C, T, F = xs.size()
                x2 = rearrange(xs, 'b c t f -> b t (f c)').contiguous()
                x2 = self.temporal_atten(x2, C, T, F)

                x_t = rearrange(x2, 'b t (f c) -> b (t f) c', f=F).contiguous()
                x_res_t = rearrange(x2, 'b t (f c) -> b c t f', f=F).contiguous()
                norm2 = self.norm2(x_t)
                norm2 = rearrange(norm2, 'b (t f) c -> b c t f', f=F).contiguous()
                ffn_t = self.IRFFN2(norm2)
                x = x_res_t + self.drop_path(ffn_t)
            else:
                print("CST attention with split cmt block is not implemented yet.")
                raise()
        return x

class CMT_block(torch.nn.Module):
    def __init__(self, params, temp_embed_dim, ffn_ratio=4., drop_path_rate=0.1):
        super().__init__()
        self.temp_embed_dim = temp_embed_dim
        self.num_layers = params['nb_self_attn_layers']
        self.ch_atten_dca = params['ChAtten_DCA']
        self.ffn_ratio = ffn_ratio
        self.nb_ch = params['nb_channels']
        
        #if params['use_ngcc']:
        #    if params['use_mel']:
        #        self.nb_ch = int(params['ngcc_out_channels'] * params['n_mics'] * (params['n_mics'] - 1) / 2 +  params['n_mics'])
        #    else:
        #        self.nb_ch = int(params['ngcc_out_channels'] * params['n_mics'] * ( 1 + (params['n_mics'] - 1) / 2))
        #else:
        #    self.nb_ch = channels

        self.block_list = nn.ModuleList([CMT_Layers(
            params=params,
            temp_embed_dim=self.temp_embed_dim,
            ffn_ratio=self.ffn_ratio,
            drop_path_rate=drop_path_rate
        ) for i in range(self.num_layers)]
        )

    def forward(self, x):
        B, C, T, F = x.size()
        M = self.nb_ch

        for block in self.block_list:
            x = block(x)

        if self.ch_atten_dca: # CST (DCA)
            B = B // M
            x = rearrange(x, '(b m) c t f -> b t (m f c)', b=B,m=M).contiguous()
        else: # CST (ULE) & DST
            x = rearrange(x, 'b c t f -> b t (f c)', c=C, t=T, f=F).contiguous()

        return x
