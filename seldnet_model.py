# The SELDnet architecture

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from IPython import embed


class MSELoss_ADPIT(object):
    def __init__(self, relative_dist=False, no_dist=False, visual_loss=False):
        super().__init__()
        self._each_loss = nn.MSELoss(reduction='none')
        self.relative_dist = relative_dist
        self.no_dist = no_dist
        self.visual_loss = visual_loss
        self.eps = 0.001

        # relevant classes are [0 -female speech, 1-male speech, 2-clapping, 4-laugh, 6-footsteps, 9-music instrument]
        self.visual_classes = [0, 1, 2, 4, 6, 9]

    def _each_calc(self, output, target):
        loss = self._each_loss(output, target)

        if self.no_dist:
            # don't train on distances at all
            loss[:, :, 3] = 0.
            loss[:, :, 7] = 0.
            loss[:, :, 11] = 0.

        elif self.relative_dist:
            # scale loss with 1 / d
            # distance indices are 3, 7, 11
            # only scale loss for distances > 0
            loss[:, :, 3] = torch.where(target[:, :, 3] > 0., loss[:, :, 3] / (target[:, :, 3] + self.eps), loss[:, :, 3])
            loss[:, :, 7] = torch.where(target[:, :, 7] > 0., loss[:, :, 7] / (target[:, :, 7] + self.eps), loss[:, :, 7])
            loss[:, :, 11] = torch.where(target[:, :, 11] > 0., loss[:, :, 11] / (target[:, :, 11] + self.eps), loss[:, :, 11])

        if self.visual_loss:
            # distance indices are 3, 7, 11. Class is active if distance > 0
            # if class is not active, set loss=0 (we do not want to train on non-active classes)
            loss[:, :, 3] = torch.where(target[:, :, 3] == 0., 0., loss[:, :, 3])
            loss[:, :, 7] = torch.where(target[:, :, 7] == 0., 0., loss[:, :, 7])
            loss[:, :, 11] = torch.where(target[:, :, 11] == 0., 0., loss[:, :, 11])
            # select only visual classes
            loss = loss[:, :, :, self.visual_classes]

        return loss.mean(dim=(2))  # class-wise frame-level

    def __call__(self, output, target):
        """
        Auxiliary Duplicating Permutation Invariant Training (ADPIT) for 13 (=1+6+6) possible combinations
        Args:
            output: [batch_size, frames, num_track*num_axis*num_class=3*3*12]
            target: [batch_size, frames, num_track_dummy=6, num_axis=4, num_class=12]
        Return:
            loss: scalar
        """

        target_A0 = target[:, :, 0, 0:1, :] * target[:, :, 0, 1:, :]  # A0, no ov from the same class, [batch_size, frames, num_axis(act)=1, num_class=12] * [batch_size, frames, num_axis(XYZD)=4, num_class=12]
        target_B0 = target[:, :, 1, 0:1, :] * target[:, :, 1, 1:, :]  # B0, ov with 2 sources from the same class
        target_B1 = target[:, :, 2, 0:1, :] * target[:, :, 2, 1:, :]  # B1
        target_C0 = target[:, :, 3, 0:1, :] * target[:, :, 3, 1:, :]  # C0, ov with 3 sources from the same class
        target_C1 = target[:, :, 4, 0:1, :] * target[:, :, 4, 1:, :]  # C1
        target_C2 = target[:, :, 5, 0:1, :] * target[:, :, 5, 1:, :]  # C2

        target_A0A0A0 = torch.cat((target_A0, target_A0, target_A0), 2)  # 1 permutation of A (no ov from the same class), [batch_size, frames, num_track*num_axis=3*4, num_class=12]
        target_B0B0B1 = torch.cat((target_B0, target_B0, target_B1), 2)  # 6 permutations of B (ov with 2 sources from the same class)
        target_B0B1B0 = torch.cat((target_B0, target_B1, target_B0), 2)
        target_B0B1B1 = torch.cat((target_B0, target_B1, target_B1), 2)
        target_B1B0B0 = torch.cat((target_B1, target_B0, target_B0), 2)
        target_B1B0B1 = torch.cat((target_B1, target_B0, target_B1), 2)
        target_B1B1B0 = torch.cat((target_B1, target_B1, target_B0), 2)
        target_C0C1C2 = torch.cat((target_C0, target_C1, target_C2), 2)  # 6 permutations of C (ov with 3 sources from the same class)
        target_C0C2C1 = torch.cat((target_C0, target_C2, target_C1), 2)
        target_C1C0C2 = torch.cat((target_C1, target_C0, target_C2), 2)
        target_C1C2C0 = torch.cat((target_C1, target_C2, target_C0), 2)
        target_C2C0C1 = torch.cat((target_C2, target_C0, target_C1), 2)
        target_C2C1C0 = torch.cat((target_C2, target_C1, target_C0), 2)

        output = output.reshape(output.shape[0], output.shape[1], target_A0A0A0.shape[2], target_A0A0A0.shape[3])  # output is set the same shape of target, [batch_size, frames, num_track*num_axis=3*4, num_class=12]
        
        pad4A = target_B0B0B1 + target_C0C1C2
        pad4B = target_A0A0A0 + target_C0C1C2
        pad4C = target_A0A0A0 + target_B0B0B1
        loss_0 = self._each_calc(output, target_A0A0A0 + pad4A)  # padded with target_B0B0B1 and target_C0C1C2 in order to avoid to set zero as target
        loss_1 = self._each_calc(output, target_B0B0B1 + pad4B)  # padded with target_A0A0A0 and target_C0C1C2
        loss_2 = self._each_calc(output, target_B0B1B0 + pad4B)
        loss_3 = self._each_calc(output, target_B0B1B1 + pad4B)
        loss_4 = self._each_calc(output, target_B1B0B0 + pad4B)
        loss_5 = self._each_calc(output, target_B1B0B1 + pad4B)
        loss_6 = self._each_calc(output, target_B1B1B0 + pad4B)
        loss_7 = self._each_calc(output, target_C0C1C2 + pad4C)  # padded with target_A0A0A0 and target_B0B0B1
        loss_8 = self._each_calc(output, target_C0C2C1 + pad4C)
        loss_9 = self._each_calc(output, target_C1C0C2 + pad4C)
        loss_10 = self._each_calc(output, target_C1C2C0 + pad4C)
        loss_11 = self._each_calc(output, target_C2C0C1 + pad4C)
        loss_12 = self._each_calc(output, target_C2C1C0 + pad4C)

        loss_min = torch.min(
            torch.stack((loss_0,
                         loss_1,
                         loss_2,
                         loss_3,
                         loss_4,
                         loss_5,
                         loss_6,
                         loss_7,
                         loss_8,
                         loss_9,
                         loss_10,
                         loss_11,
                         loss_12), dim=0),
            dim=0).indices

        loss = (loss_0 * (loss_min == 0) +
                loss_1 * (loss_min == 1) +
                loss_2 * (loss_min == 2) +
                loss_3 * (loss_min == 3) +
                loss_4 * (loss_min == 4) +
                loss_5 * (loss_min == 5) +
                loss_6 * (loss_min == 6) +
                loss_7 * (loss_min == 7) +
                loss_8 * (loss_min == 8) +
                loss_9 * (loss_min == 9) +
                loss_10 * (loss_min == 10) +
                loss_11 * (loss_min == 11) +
                loss_12 * (loss_min == 12)).mean()

        return loss


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        return x


class SeldModel(torch.nn.Module):
    def __init__(self, in_feat_shape, out_shape, params, in_vid_feat_shape=None):
        super().__init__()
        self.nb_classes = params['unique_classes']
        self.params=params
        self.conv_block_list = nn.ModuleList()
        if len(params['f_pool_size']):
            for conv_cnt in range(len(params['f_pool_size'])):
                self.conv_block_list.append(ConvBlock(in_channels=params['nb_cnn2d_filt'] if conv_cnt else in_feat_shape[1], out_channels=params['nb_cnn2d_filt']))
                self.conv_block_list.append(nn.MaxPool2d((params['t_pool_size'][conv_cnt], params['f_pool_size'][conv_cnt])))
                self.conv_block_list.append(nn.Dropout2d(p=params['dropout_rate']))

        self.gru_input_dim = params['nb_cnn2d_filt'] * int(np.floor(in_feat_shape[-1] / np.prod(params['f_pool_size'])))
        self.gru = torch.nn.GRU(input_size=self.gru_input_dim, hidden_size=params['rnn_size'],
                                num_layers=params['nb_rnn_layers'], batch_first=True,
                                dropout=params['dropout_rate'], bidirectional=True)

        self.mhsa_block_list = nn.ModuleList()
        self.layer_norm_list = nn.ModuleList()
        for mhsa_cnt in range(params['nb_self_attn_layers']):
            self.mhsa_block_list.append(nn.MultiheadAttention(embed_dim=self.params['rnn_size'], num_heads=self.params['nb_heads'], dropout=self.params['dropout_rate'], batch_first=True))
            self.layer_norm_list.append(nn.LayerNorm(self.params['rnn_size']))

        # fusion layers
        if in_vid_feat_shape is not None:
            self.visual_embed_to_d_model = nn.Linear(in_features = int(in_vid_feat_shape[2]*in_vid_feat_shape[3]), out_features = self.params['rnn_size'] )
            self.transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=self.params['rnn_size'], nhead=self.params['nb_heads'], batch_first=True)
            self.transformer_decoder = nn.TransformerDecoder(self.transformer_decoder_layer, num_layers=self.params['nb_transformer_layers'])

        self.fnn_list = torch.nn.ModuleList()
        if params['nb_fnn_layers']:
            for fc_cnt in range(params['nb_fnn_layers']):
                self.fnn_list.append(nn.Linear(params['fnn_size'] if fc_cnt else self.params['rnn_size'], params['fnn_size'], bias=True))
        self.fnn_list.append(nn.Linear(params['fnn_size'] if params['nb_fnn_layers'] else self.params['rnn_size'], out_shape[-1], bias=True))

        self.doa_act = nn.Tanh()
        self.dist_act = nn.ReLU()

    def forward(self, x, vid_feat=None):
        """input: (batch_size, mic_channels, time_steps, mel_bins)"""
        for conv_cnt in range(len(self.conv_block_list)):
            x = self.conv_block_list[conv_cnt](x)

        x = x.transpose(1, 2).contiguous()
        x = x.view(x.shape[0], x.shape[1], -1).contiguous()
        (x, _) = self.gru(x)
        x = torch.tanh(x)
        x = x[:, :, x.shape[-1]//2:] * x[:, :, :x.shape[-1]//2]

        for mhsa_cnt in range(len(self.mhsa_block_list)):
            x_attn_in = x
            x, _ = self.mhsa_block_list[mhsa_cnt](x_attn_in, x_attn_in, x_attn_in)
            x = x + x_attn_in
            x = self.layer_norm_list[mhsa_cnt](x)

        if vid_feat is not None:
            vid_feat = vid_feat.view(vid_feat.shape[0], vid_feat.shape[1], -1)  # b x 50 x 49
            vid_feat = self.visual_embed_to_d_model(vid_feat)
            x = self.transformer_decoder(x, vid_feat)

        for fnn_cnt in range(len(self.fnn_list) - 1):
            x = self.fnn_list[fnn_cnt](x)
        doa = self.fnn_list[-1](x)

        return doa
    
class MySeldModel(torch.nn.Module):
    def __init__(self, in_feat_shape, out_shape, params, in_vid_feat_shape=None, n_channels=4, n_delays=None):
        super().__init__()
        self.n_channels = n_channels
        self.nb_classes = params['unique_classes']
        self.params=params
        self.conv_block_list = nn.ModuleList()
        if len(params['f_pool_size']):
            for conv_cnt in range(len(params['f_pool_size'])):
                self.conv_block_list.append(ConvBlock(in_channels=params['nb_cnn2d_filt'] if conv_cnt else n_channels, out_channels=params['nb_cnn2d_filt']))
                self.conv_block_list.append(nn.MaxPool2d((params['t_pool_size'][conv_cnt], params['f_pool_size'][conv_cnt])))
                self.conv_block_list.append(nn.Dropout2d(p=params['dropout_rate']))

        self.gru_input_dim = params['nb_cnn2d_filt'] * int(np.floor(in_feat_shape[-1] / np.prod(params['f_pool_size'])))
        self.gru = torch.nn.GRU(input_size=self.gru_input_dim, hidden_size=params['rnn_size'],
                                num_layers=params['nb_rnn_layers'], batch_first=True,
                                dropout=params['dropout_rate'], bidirectional=True)

        self.mhsa_block_list = nn.ModuleList()
        self.layer_norm_list = nn.ModuleList()
        for mhsa_cnt in range(params['nb_self_attn_layers']):
            self.mhsa_block_list.append(nn.MultiheadAttention(embed_dim=self.params['rnn_size'], num_heads=self.params['nb_heads'], dropout=self.params['dropout_rate'], batch_first=True))
            self.layer_norm_list.append(nn.LayerNorm(self.params['rnn_size']))

        # fusion layers
        if in_vid_feat_shape is not None:
            self.visual_embed_to_d_model = nn.Linear(in_features = int(in_vid_feat_shape[2]*in_vid_feat_shape[3]), out_features = self.params['rnn_size'] )
            self.transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=self.params['rnn_size'], nhead=self.params['nb_heads'], batch_first=True)
            self.transformer_decoder = nn.TransformerDecoder(self.transformer_decoder_layer, num_layers=self.params['nb_transformer_layers'])

        # GCC-PHAT relation network
        self.mic_token_dim = 8
        self.n_gcc = int(n_channels * (n_channels - 1) / 2)
        self.mic_tokens = nn.Parameter(torch.randn(1, self.n_gcc, 1, self.mic_token_dim))

        self.n_gcc = n_channels * (n_channels - 1) / 2
        if n_delays is None:
            n_delays = in_feat_shape[-1]  # this is currently = 64. TODO: 2*6=12 delays per correlation
        self.rel1 = nn.Sequential(
            nn.Linear(n_delays, n_delays * 8),
            nn.LayerNorm(n_delays * 8),
            nn.GELU(),
            nn.Linear(n_delays * 8, n_delays * 16),
            nn.LayerNorm(n_delays * 16),
            nn.GELU(),
        )

        self.avg_pool = nn.AvgPool2d((params['t_pool_size'][0], 1)) # pool over 5 time samples, (5,1)

        self.rel2 = nn.Sequential(
            nn.Linear(n_delays * 16 +  self.mic_token_dim, n_delays * 32),
            nn.LayerNorm(n_delays * 32),
            nn.GELU(),
            nn.Linear(n_delays * 32, n_delays * 64),
            nn.LayerNorm(n_delays * 64),
            nn.GELU(),
        )

        self.rel3 = nn.Sequential(
            nn.Linear(n_delays * 64, n_delays * 64),
            nn.LayerNorm(n_delays * 64),
            nn.GELU(),
        )

        #fully connected for predictions
        self.ff = nn.Sequential(
            nn.Linear(self.params['rnn_size']+n_delays*64, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, out_shape[-1])
        )

        self.doa_act = nn.Tanh()
        self.dist_act = nn.ReLU()


    def forward(self, x, vid_feat=None):
        """input: (batch_size, mic_channels, time_steps, mel_bins)"""

        # separate gcc and mel features
        gcc = x[:, self.n_channels:] # gcc correlations
        x = x[:, :self.n_channels] # mel spectrograms

        for conv_cnt in range(len(self.conv_block_list)):
            x = self.conv_block_list[conv_cnt](x)

        x = x.transpose(1, 2).contiguous()
        x = x.view(x.shape[0], x.shape[1], -1).contiguous()
        (x, _) = self.gru(x)
        x = torch.tanh(x)
        x = x[:, :, x.shape[-1]//2:] * x[:, :, :x.shape[-1]//2]

        for mhsa_cnt in range(len(self.mhsa_block_list)):
            x_attn_in = x
            x, _ = self.mhsa_block_list[mhsa_cnt](x_attn_in, x_attn_in, x_attn_in)
            x = x + x_attn_in
            x = self.layer_norm_list[mhsa_cnt](x)

        if vid_feat is not None:
            vid_feat = vid_feat.view(vid_feat.shape[0], vid_feat.shape[1], -1)  # b x 50 x 49
            vid_feat = self.visual_embed_to_d_model(vid_feat)
            x = self.transformer_decoder(x, vid_feat)

        # relation network
        gcc = self.rel1(gcc)
        gcc = self.avg_pool(gcc)
        
        # append mic tokens to gcc features
        bs, _, n_time, _ = gcc.shape
        mic_tokens = self.mic_tokens.repeat(bs, 1, n_time, 1) 
        gcc = torch.cat((gcc, mic_tokens), dim=-1)

        gcc = self.rel2(gcc)
        gcc = torch.max(gcc, dim=1)[0]
        gcc = self.rel3(gcc)

        x = torch.cat((x, gcc), dim=-1)

        doa = self.ff(x)

        return doa
        # the below-commented code applies tanh for doa and relu for distance estimates respectively in multi-accdoa scenarios.
        # they can be uncommented and used, but there is no significant changes in the results.
        #doa = doa.reshape(doa.size(0), doa.size(1), 3, 4, 13)
        #doa1 = doa[:, :, :, :3, :]
        #dist = doa[:, :, :, 3:, :]

        #doa1 = self.doa_act(doa1)
        #dist = self.dist_act(dist)
        #doa2 = torch.cat((doa1, dist), dim=3)

        #doa2 = doa2.reshape((doa.size(0), doa.size(1), -1))
        #return doa2