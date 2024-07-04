import torch
import torch.nn as nn
from einops import rearrange

class CST_attention(torch.nn.Module):
    def __init__(self, temp_embed_dim, params):
        super().__init__()
        self.ChAtten_dca = params['ChAtten_DCA']
        self.ChAtten_ule = params['ChAtten_ULE']
        self.FreqAtten = params['FreqAtten']
        self.linear_layer = params['LinearLayer']
        self.dropout_rate = params['dropout_rate']
        self.temp_embed_dim = temp_embed_dim
        print(temp_embed_dim)
        #self.nb_ch = 7

        # Channel attention w. Divided Channel Attention (DCA) ---------------------------------------------#
        if self.ChAtten_dca:
            self.ch_attn_embed_dim = params['nb_cnn2d_filt']  # 64
            self.ch_mhsa = nn.MultiheadAttention(embed_dim=self.ch_attn_embed_dim, num_heads=params['nb_heads'],
                                      dropout=self.dropout_rate, batch_first=True)
            self.ch_layer_norm = nn.LayerNorm(self.temp_embed_dim)
            if self.linear_layer:
                self.ch_linear = nn.Linear(self.temp_embed_dim, self.temp_embed_dim)

        # Channel attention w. Unfolded Local Embedding (ULE) ----------------------------------------------#
        if self.ChAtten_ule:
            self.patch_size_t = 25 if params['t_pooling_loc']=='end' else 10
            self.patch_size_f = 4
            self.patch_size = (self.patch_size_t, self.patch_size_f)
            f_dim = 382 if params['use_salsalite'] else 64
            self.freq_dim = int(f_dim / torch.prod(torch.Tensor(params['f_pool_size'])))
            self.temp_dim = 250 if params['t_pooling_loc']=='end' else 50
            self.unfold = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)
            self.fold = nn.Fold(output_size=(self.temp_dim, self.freq_dim), kernel_size=self.patch_size, stride=self.patch_size)
            self.ch_attn_embed_dim = int(self.patch_size_t * self.patch_size_f)
            self.ch_mhsa = nn.MultiheadAttention(embed_dim=self.ch_attn_embed_dim, num_heads=10 if params['t_pooling_loc']=='end' else params['nb_heads'],
                                                 dropout=self.dropout_rate, batch_first=True)
            self.ch_layer_norm = nn.LayerNorm(self.temp_embed_dim)
            if self.linear_layer:
                self.ch_linear = nn.Linear(self.temp_embed_dim, self.temp_embed_dim)

        # Spectral attention -------------------------------------------------------------------------------#
        if self.FreqAtten:
            self.sp_attn_embed_dim = params['nb_cnn2d_filt']  # 64
            self.embed_dim_4_freq_attn = params['nb_cnn2d_filt'] # Update the temp embedding if freq attention is applied
            self.sp_mhsa = nn.MultiheadAttention(embed_dim=self.sp_attn_embed_dim, num_heads=params['nb_heads'],
                                      dropout=self.dropout_rate, batch_first=True)
            self.sp_layer_norm = nn.LayerNorm(self.temp_embed_dim)
            if self.linear_layer:
                self.sp_linear = nn.Linear(self.temp_embed_dim, self.temp_embed_dim)

        # temporal attention -----------------------------------------------------------------------------------#
        self.temp_mhsa = nn.MultiheadAttention(embed_dim=self.embed_dim_4_freq_attn if params['FreqAtten'] else self.embed_dim,
                                  num_heads=params['nb_heads'],
                                  dropout=self.dropout_rate, batch_first=True)
        self.temp_layer_norm = nn.LayerNorm(self.temp_embed_dim)
        if self.linear_layer:
            self.temp_linear = nn.Linear(self.temp_embed_dim, self.temp_embed_dim)

        self.activation = nn.GELU()
        self.drop_out = nn.Dropout(self.dropout_rate if self.dropout_rate > 0. else nn.Identity())

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self,x, M, C, T, F):
        if self.ChAtten_ule: # CST-attention(ULE)
            # channel attention unfold
            B = x.size(0)
            x_init = x.clone()

            x_unfold_in = rearrange(x_init, ' b t (f c) -> b c t f', c=C, t=T, f=F).contiguous()
            x_unfold = self.unfold(x_unfold_in) # unfold for additional embedding for channel attention
            x_unfold = rearrange(x_unfold, 'b (c u) tf -> (b tf) c u', c=C).contiguous()

            xc, _ = self.ch_mhsa(x_unfold, x_unfold, x_unfold)

            xc = rearrange(xc, '(b tf) c u -> b (c u) tf', b=B).contiguous()
            xc = self.fold(xc)  # fold to rearrange
            xc = rearrange(xc, 'b c t f -> b t (f c)').contiguous()

            if self.linear_layer:
                xc = self.activation(self.ch_linear(xc))
            xc = xc + x_init
            if self.dropout_rate:
                xc = self.drop_out(xc)
            xc = self.ch_layer_norm(xc)

            # spectral attention
            xs = rearrange(xc, ' b t (f c) -> (b t) f c', f=F).contiguous()
            xs, _ = self.sp_mhsa(xs, xs, xs)
            xs = rearrange(xs, ' (b t) f c -> b t (f c)', t=T).contiguous()
            if self.linear_layer:
                xs = self.activation(self.sp_linear(xs))
            xs = xs + xc
            if self.dropout_rate:
                xs = self.drop_out(xs)
            xs = self.sp_layer_norm(xs)

            # temporal attention
            xt = rearrange(xs, ' b t (f c) -> (b f) t c', f=F).contiguous()
            xt, _ = self.temp_mhsa(xt, xt, xt)
            xt = rearrange(xt, ' (b f) t c -> b t (f c)', f=F).contiguous()
            if self.linear_layer:
                xt = self.activation(self.temp_linear(xt))
            xt = xt + xs
            if self.dropout_rate:
                xt = self.drop_out(xt)
            x = self.temp_layer_norm(xt)

        elif self.ChAtten_dca: #CST-attention (DCA)
            # channel attention
            x_init = x.clone()
            xc = rearrange(x_init, 'b t (m f c)-> (b t f) m c', c=C, f=F).contiguous()

            xc, _ = self.ch_mhsa(xc, xc, xc)
            xc = rearrange(xc, ' (b t f) m c -> b t (f m c)', t=T, f=F).contiguous()
            if self.linear_layer:
                xc = self.activation(self.ch_linear(xc))
            xc = xc + x_init
            if self.dropout_rate:
                xc = self.drop_out(xc)
            xc = self.ch_layer_norm(xc)

            # spectral attention
            xs = rearrange(xc, ' b t (f m c) -> (b t m) f c', c=C, t=T, f=F).contiguous()
            xs, _ = self.sp_mhsa(xs, xs, xs)
            xs = rearrange(xs, ' (b t m) f c -> b t (f m c)', m=M, t=T).contiguous()
            if self.linear_layer:
                xs = self.activation(self.sp_linear(xs))
            xs = xs + xc
            if self.dropout_rate:
                xs = self.drop_out(xs)
            xs = self.sp_layer_norm(xs)

            # temporal attention
            xt = rearrange(xs, ' b t (f m c) -> (b f m) t c', m=M, f=F).contiguous()
            xt, _ = self.temp_mhsa(xt, xt, xt)
            xt = rearrange(xt, ' (b f m) t c -> b t (f m c)', m=M, f=F).contiguous()
            if self.linear_layer:
                xt = self.activation(self.temp_linear(xt))
            xt = xt + xs
            if self.dropout_rate:
                xt = self.drop_out(xt)
            x = self.temp_layer_norm(xt)

        elif self.FreqAtten: # DST-attention
            x_init = x.clone()
            # spectral attention
            x_attn_in = rearrange(x_init, ' b t (f c) -> (b t) f c', f=F).contiguous()
            xs, _ = self.sp_mhsa(x_attn_in, x_attn_in, x_attn_in)
            xs = rearrange(xs, ' (b t) f c -> b t (f c)', t=T).contiguous()
            if self.linear_layer:
                xs = self.activation(self.sp_linear(xs))
            xs = xs + x_init
            if self.dropout_rate:
                xs = self.drop_out(xs)
            xs = self.sp_layer_norm(xs)

            # temporal attention
            xt = rearrange(xs, ' b t (f c) -> (b f) t c', c=C).contiguous()
            xt, _ = self.temp_mhsa(xt, xt, xt)
            xt = rearrange(xt, ' (b f) t c -> b t (f c)', f=F).contiguous()
            if self.linear_layer:
                xt = self.activation(self.temp_linear(xt))
            xt = xt + xs
            if self.dropout_rate:
                xt = self.drop_out(xt)
            x = self.temp_layer_norm(xt)

        else: # Basic Temporal Attention
            x_attn_in = x
            x, _ = self.temp_mhsa(x_attn_in, x_attn_in, x_attn_in)
            x = x + x_attn_in
            x = self.temp_layer_norm(x)

        return x

class CST_encoder(torch.nn.Module):
    def __init__(self, temp_embed_dim, params):
        super().__init__()
        self.freq_atten = params['FreqAtten']
        self.ch_atten_dca = params['ChAtten_DCA']
        self.ch_atten_ule = params['ChAtten_ULE']
        self.nb_ch = params['nb_channels']
        n_layers = params['nb_self_attn_layers']

        self.block_list = nn.ModuleList([CST_attention(
            temp_embed_dim = temp_embed_dim,
            params=params
        ) for _ in range(n_layers)]
        )

    def forward(self, x):
        B, C, T, F = x.size()
        M = self.nb_ch # Number of Microphone Channels

        # CST-attention
        if self.ch_atten_dca:
            B = B // M  # Real Batch
            x = rearrange(x, '(b m) c t f -> b t (m f c)', b=B, m=M).contiguous()

        # DST-attention
        if self.ch_atten_ule or self.freq_atten:
            x = rearrange(x, 'b c t f -> b t (f c)').contiguous()

        for block in self.block_list:
            x = block(x, M, C, T, F)

        return x
