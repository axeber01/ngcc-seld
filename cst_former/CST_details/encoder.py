import torch
import torch.nn as nn
from .layers import ConvBlock, ConvBlockTwo, ResidualBlock, SEBasicBlock

class conv_encoder(torch.nn.Module):
    def __init__(self, in_feat_shape, params):
        super().__init__()
        self.params = params
        self.t_pooling_loc = params["t_pooling_loc"]
        assert(len(params['f_pool_size']))

        self.conv_block_list = nn.ModuleList()
        
        if self.params['ChAtten_DCA']: in_channels = 1
        else: in_channels = params['nb_channels']
            #if params['use_ngcc']:
            #    if params['use_mel']:
            #        in_channels = int(params['ngcc_out_channels'] * params['n_mics'] * (params['n_mics'] - 1) / 2 +  params['n_mics'])
            #    else:
            #        in_channels = int(params['ngcc_out_channels'] * params['n_mics'] * ( 1 + (params['n_mics'] - 1) / 2))
            #else:
            #    in_channels = in_feat_shape[1]

        for conv_cnt in range(len(params['f_pool_size'])):
            self.conv_block_list.append(nn.Sequential(
                ConvBlock(in_channels=params['nb_cnn2d_filt'] if conv_cnt else in_channels,
                          out_channels=params['nb_cnn2d_filt']),
                nn.MaxPool2d((params['t_pool_size'][conv_cnt] if self.t_pooling_loc == 'front' else 1,
                              params['f_pool_size'][conv_cnt])),
                nn.Dropout2d(p=params['dropout_rate']),
            ))

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
        for conv_cnt in range(len(self.conv_block_list)):
            x = self.conv_block_list[conv_cnt](x)  # out: B,C,T,F
        return x

class resnet_encoder(torch.nn.Module):
    def __init__(self, in_feat_shape, params):
        super().__init__()
        self.params = params
        self.t_pooling_loc = params["t_pooling_loc"]
        self.ratio = [1, 2, 4, 8]
        assert(len(params['f_pool_size']))

        self.res_block_list = nn.ModuleList()
        self.first_conv = ConvBlock(in_channels=in_feat_shape[1],
                  out_channels=params['nb_resnet_filt'])

        self.last_conv = ConvBlock(in_channels=params['nb_resnet_filt'] * self.ratio[-1],
                  out_channels=params['nb_cnn2d_filt'])


        for conv_cnt in range(len(params['f_pool_size'])):
            self.res_block_list.append(nn.Sequential(
                # First Layer
                ResidualBlock(in_channels=params['nb_resnet_filt'] ,
                              out_channels=params['nb_resnet_filt']*self.ratio[conv_cnt]) if not conv_cnt else
                ConvBlockTwo(in_channels=params['nb_resnet_filt']*self.ratio[conv_cnt-1],
                          out_channels=params['nb_resnet_filt']*self.ratio[conv_cnt]),
                # Second Layer
                ResidualBlock(in_channels=params['nb_resnet_filt'] * self.ratio[conv_cnt],
                              out_channels=params['nb_resnet_filt'] * self.ratio[conv_cnt]),
                # T-F pooling
                nn.MaxPool2d((params['t_pool_size'][conv_cnt] if self.t_pooling_loc == 'front' else 1,
                              params['f_pool_size'][conv_cnt])) # T-F pooling
            ))

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
        x = self.first_conv(x)
        for conv_cnt in range(len(self.res_block_list)):
            x = self.res_block_list[conv_cnt](x)  # out: B,C,T,F
        x = self.last_conv(x)
        return x

class senet_encoder(torch.nn.Module):
    def __init__(self, in_feat_shape, params):
        super().__init__()
        self.params = params
        self.baseline = params['baseline']
        self.t_pooling_loc = params["t_pooling_loc"]
        self.ratio = [1, 2, 4, 8]
        assert(len(params['f_pool_size']))

        self.res_block_list = nn.ModuleList()
        self.first_conv = ConvBlock(in_channels=in_feat_shape[1],
                  out_channels=params['nb_resnet_filt'])

        if not self.baseline:
            self.last_conv = ConvBlock(in_channels=params['nb_resnet_filt'] * self.ratio[-1],
                      out_channels=params['nb_cnn2d_filt'])

        for conv_cnt in range(len(params['f_pool_size'])):
            self.res_block_list.append(nn.Sequential(
                # First Layer
                SEBasicBlock(in_channels=params['nb_resnet_filt'] ,
                              out_channels=params['nb_resnet_filt']*self.ratio[conv_cnt], MSCAM=params['MSCAM']) if not conv_cnt else
                ConvBlockTwo(in_channels=params['nb_resnet_filt']*self.ratio[conv_cnt-1],
                          out_channels=params['nb_resnet_filt']*self.ratio[conv_cnt]),
                # Second Layer
                SEBasicBlock(in_channels=params['nb_resnet_filt'] * self.ratio[conv_cnt],
                              out_channels=params['nb_resnet_filt'] * self.ratio[conv_cnt], MSCAM=params['MSCAM']),
                # T-F pooling
                nn.MaxPool2d((params['t_pool_size'][conv_cnt] if self.t_pooling_loc == 'front' else 1,
                              params['f_pool_size'][conv_cnt])) # T-F pooling
            ))

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
        x = self.first_conv(x)

        for conv_cnt in range(len(self.res_block_list)):
            x = self.res_block_list[conv_cnt](x)  # out: B,C,T,F

        if not self.baseline:
            x = self.last_conv(x)
        return x


class Encoder(torch.nn.Module):
    def __init__(self, in_feat_shape, params):
        super().__init__()

        if params['encoder'] == 'ResNet':
            self.encoder = resnet_encoder(in_feat_shape, params)
        elif params['encoder'] == 'conv':
            self.encoder = conv_encoder(in_feat_shape, params)
        elif params['encoder'] == 'SENet':
            self.encoder = senet_encoder(in_feat_shape, params)

    def forward(self, x):
        x = self.encoder(x)
        return x
