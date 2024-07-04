import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def make_pairs(x):
    """make the int -> tuple
    """
    return x if isinstance(x, tuple) else (x, x)

def generate_relative_distance(number_size):
    """return relative distance, (number_size**2, number_size**2, 2)
    """
    indices = torch.tensor(np.array([[x, y] for x in range(number_size) for y in range(number_size)]))
    distances = indices[None, :, :] - indices[:, None, :]
    distances = distances + number_size - 1   # shift the zeros postion
    return distances


#####################################################################################################################
### Layers for CST-former
#####################################################################################################################
class GRU_layer(torch.nn.Module):
    """
    GRU layer for baseline
    """
    def __init__(self, in_feat_shape, params):
        super().__init__()
        if params["baseline"]:
            self.gru_input_dim = params['nb_cnn2d_filt'] * int(np.floor(in_feat_shape[-1] / 4))
            self.gru = torch.nn.GRU(input_size=self.gru_input_dim, hidden_size=params['rnn_size'],
                                    num_layers=params['nb_rnn_layers'], batch_first=True,
                                    dropout=params['dropout_rate'], bidirectional=True)

    def forward(self,x):
        x = x.transpose(1, 2).contiguous()
        x = x.view(x.shape[0], x.shape[1], -1).contiguous()
        (x, _) = self.gru(x)
        x = torch.tanh(x)
        x = x[:, :, x.shape[-1] // 2:] * x[:, :, :x.shape[-1] // 2]
        return x

class FC_layer(torch.nn.Module):
    """
    Fully Connected layer for baseline

    Args:
        out_shape (int): output shape for SLED
                         ex. 39 for single-ACCDOA, 117 for multi-ACCDOA
        temp_embed_dim (int): the input size
        params : parameters from parameter.py
    """
    def __init__(self, out_shape,temp_embed_dim, params):
        super().__init__()

        self.fnn_list = torch.nn.ModuleList()
        if params['nb_fnn_layers']:
            for fc_cnt in range(params['nb_fnn_layers']):
                self.fnn_list.append(
                    nn.Linear(params['fnn_size'] if fc_cnt else temp_embed_dim, params['fnn_size'], bias=True))
        self.fnn_list.append(
            nn.Linear(params['fnn_size'] if params['nb_fnn_layers'] else temp_embed_dim, out_shape[-1],
                      bias=True))

        self.doa_act = nn.Tanh()
        self.dist_act = nn.ReLU()

    def forward(self, x:torch.Tensor):
        for fnn_cnt in range(len(self.fnn_list) - 1):
            x = self.fnn_list[fnn_cnt](x)

        doa = self.fnn_list[-1](x)

       	doa = doa.reshape(doa.size(0), doa.size(1), 3, 4, 13)
        #doa1 = doa[:, :, :, :3, :]
        #dist = doa[:, :, :, 3:, :]

        #doa1 = self.doa_act(doa1)
        #dist = self.dist_act(dist)
        #doa = torch.cat((doa1, dist), dim=3)

        doa = doa.reshape((doa.size(0), doa.size(1), -1))
        return doa


#####################################################################################################################
### Convolution meets transformer (CMT)
#####################################################################################################################

class LocalPerceptionUint(torch.nn.Module):
    def __init__(self, dim, act=False):
        super(LocalPerceptionUint, self).__init__()
        self.act = act
        self.conv_3x3_dw = ConvDW3x3(dim)
        if self.act:
            self.actation = nn.Sequential(
                nn.GELU(),
                nn.BatchNorm2d(dim)
            )

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        if self.act:
            out = self.actation(self.conv_3x3_dw(x))
            return out
        else:
            out = self.conv_3x3_dw(x)
            return out

class InvertedResidualFeedForward(torch.nn.Module):
    def __init__(self, dim, dim_ratio=4.):
        super(InvertedResidualFeedForward, self).__init__()
        output_dim = int(dim_ratio * dim)
        self.conv1x1_gelu_bn = ConvGeluBN(
            in_channel=dim,
            out_channel=output_dim,
            kernel_size=1,
            stride_size=1,
            padding=0
        )
        self.conv3x3_dw = ConvDW3x3(dim=output_dim)
        self.act = nn.Sequential(
            nn.GELU(),
            nn.BatchNorm2d(output_dim)
        )
        self.conv1x1_pw = nn.Sequential(
            nn.Conv2d(output_dim, dim, 1, 1, 0),
            nn.BatchNorm2d(dim)
        )

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.conv1x1_gelu_bn(x)
        out = x + self.act(self.conv3x3_dw(x))
        out = self.conv1x1_pw(out)
        return out


class ConvDW3x3(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super(ConvDW3x3, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=make_pairs(kernel_size),
            padding=make_pairs(1),
            groups=dim)

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')

    def forward(self, x):
        x = self.conv(x)
        return x

class ConvGeluBN(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride_size, padding=1):
        """build the conv3x3 + gelu + bn module
        """
        super(ConvGeluBN, self).__init__()
        self.kernel_size = make_pairs(kernel_size)
        self.stride_size = make_pairs(stride_size)
        self.padding_size = make_pairs(padding)
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.conv3x3_gelu_bn = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channel,
                      out_channels=self.out_channel,
                      kernel_size=self.kernel_size,
                      stride=self.stride_size,
                      padding=self.padding_size),
            nn.GELU(),
            nn.BatchNorm2d(self.out_channel)
        )

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.conv3x3_gelu_bn(x)
        return x


#####################################################################################################################
### Convolutional Blocks
#####################################################################################################################

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize nn.Linear and nn.LayerNorm
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

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        return x


class ConvBlockTwo(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize nn.Linear and nn.LayerNorm
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

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


#####################################################################################################################
### ResNet
#####################################################################################################################

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
        super().__init__()
        # First Layer
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize nn.Linear and nn.LayerNorm
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

    def forward(self, x_init):
        identity = x_init.clone()
        x = F.relu(self.bn1(self.conv1(x_init)))
        x = F.relu(self.bn2(self.conv2(x)) + identity)
        return x


#####################################################################################################################
### Squeeze and Excitation
#####################################################################################################################

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SE_MSCAM(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SE_MSCAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.se1 = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.BatchNorm2d(channel)
        )
        self.se2 = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.BatchNorm2d(channel)
        )
        self.activation = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y1 = self.avg_pool(x).view(b, c, 1, 1)
        y1 = self.se1(y1).view(b, c, 1, 1)

        y2 = self.se2(x)

        y = y1.expand_as(y2) + y2
        y = self.activation(y)
        return x * y


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None,
                 *, reduction=16, MSCAM=False):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if not MSCAM:
            self.se = SELayer(out_channels, reduction)
        else:
            self.se = SE_MSCAM(out_channels, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

#####################################################################################################################
### Attention Layers for CMT Split (To apply LPU&IRFFN on each attention layers)
#####################################################################################################################
class Spec_attention(torch.nn.Module):
    def __init__(self, temp_embed_dim, params):
        super().__init__()
        # self.params = params
        self.dropout_rate = params['dropout_rate']
        self.linear_layer = params['linear_layer']
        self.temp_embed_dim = temp_embed_dim

        # Spectral attention -------------------------------------------------------------------------------#
        self.sp_attn_embed_dim = params['nb_cnn2d_filt']  # 64
        self.sp_mhsa = nn.MultiheadAttention(embed_dim=self.sp_attn_embed_dim, num_heads=params['nb_heads'],
                                  dropout=params['dropout_rate'], batch_first=True)
        self.sp_layer_norm = nn.LayerNorm(self.temp_embed_dim)
        if self.params['LinearLayer']:
            self.sp_linear = nn.Linear(self.sp_attn_embed_dim, self.sp_attn_embed_dim)

        self.activation = nn.GELU()
        self.drop_out = nn.Dropout(self.dropout_rate) if self.dropout_rate > 0. else nn.Identity()

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
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

    def forward(self,x, C, T, F):
        # spectral attention
        x_init = x
        x_attn_in = rearrange(x_init, ' b t (f c) -> (b t) f c', c=C,f=F).contiguous()
        xs, _ = self.sp_mhsa(x_attn_in, x_attn_in, x_attn_in)
        xs = rearrange(xs, ' (b t) f c -> b t (f c)', t=T).contiguous()
        if self.linear_layer:
            xs = self.activation(self.sp_linear(xs))
        xs = xs + x_init
        if self.dropout_rate:
            xs = self.drop_out(xs)
        x_out = self.sp_layer_norm(xs)
        return x_out

class Temp_attention(torch.nn.Module):
    def __init__(self, temp_embed_dim, params):
        super().__init__()
        # self.params = params
        self.dropout_rate = params['dropout_rate']
        self.linear_layer = params['linear_layer']
        self.temp_embed_dim = temp_embed_dim
        self.embed_dim_4_freq_attn = params['nb_cnn2d_filt']  # Update the temp embedding if freq attention is applied
        # temporal attention -----------------------------------------------------------------------------------#
        self.temp_mhsa = nn.MultiheadAttention(embed_dim=self.embed_dim_4_freq_attn if params['FreqAtten'] else self.temp_embed_dim,
                                  num_heads=params['nb_heads'],
                                  dropout=params['dropout_rate'], batch_first=True)
        self.temp_layer_norm = nn.LayerNorm(self.temp_embed_dim)
        if self.params['LinearLayer']:
            self.temp_linear = nn.Linear(self.temp_embed_dim, self.temp_embed_dim)

        self.activation = nn.GELU()
        self.drop_out = nn.Dropout(self.dropout_rate) if self.dropout_rate > 0. else nn.Identity()

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
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

    def forward(self,x, C, T, F):
        # temporal attention
        x_init = x
        xt = rearrange(x_init, ' b t (f c) -> (b f) t c', c=C).contiguous()
        xt, _ = self.temp_mhsa(xt, xt, xt)
        xt = rearrange(xt, ' (b f) t c -> b t (f c)', f=F).contiguous()
        if self.linear_layer:
            xt = self.activation(self.temp_linear(xt))
        xt = xt + x_init
        if self.dropout_rate:
            xt = self.drop_out(xt)
        x_out = self.temp_layer_norm(xt)
        return x_out
