"""
this module implements TDNN transformer
for transducer encoder
"""

import torch.nn as nn
import torch.nn.functional as F

from trainer.model.modules.transformer import TransformerEncoderLayer

class Net(nn.Module):
    """
    Class implements TDNN transformer used for transducer
    encoder in "Minimum Bayes Risk Training of RNN-Transducer
    for End-to-End Speech Recognition", InterSpeech2020

    Args:
        input_dim (int): input dimension
        input_ctx (int): deprecated
        output_dim (int): output dimension
        tdnn_nhid (int): hidden dimension for TDNN
        tdnn_layers (int): number of TDNN layers, note that we insert
                          1 transformer layer after each 3 TDNN layers
        bn_dim (int): deprecated

    """
    def __init__(self, input_dim, input_ctx, output_dim,
                 tdnn_nhid, tdnn_layers, bn_dim=0):
        super(Net, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.tdnn_nhid = tdnn_nhid

        #the number of each tdnn layer params:
        #filter_size * tdnn_nhid * tdnn_nhid
        filter_size = 3
        self.tdnn_nhid = tdnn_nhid
        self.filter_size = filter_size
        self.fc_in = nn.Linear(input_dim, tdnn_nhid)
        self.bn_in = nn.BatchNorm1d(tdnn_nhid)

        #hidden layers
        assert tdnn_layers > (3 + 1)
        self.hidden_conv = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=tdnn_nhid,
                       kernel_size=(filter_size, tdnn_nhid),
                       dilation=(1, 1)) for i in range(3)])
        self.hidden_conv.extend([nn.Conv2d(in_channels=1,
                                           out_channels=tdnn_nhid,
                                           kernel_size=(filter_size, tdnn_nhid),
                                           dilation=(3, 1))
                                 for i in range(tdnn_layers-3-1)])
        self.hidden_conv.append(nn.Conv2d(in_channels=1,
                                          out_channels=tdnn_nhid,
                                          kernel_size=(filter_size, tdnn_nhid),
                                          stride=(4, 1),#stride=(3, 1),
                                          dilation=(3, 1)))
        self.hidden_bn = nn.ModuleList(
            [nn.BatchNorm1d(tdnn_nhid) for i in range(tdnn_layers)])

        #self.fc_final   = nn.Linear(tdnn_nhid, tdnn_nhid)
        num_heads = [16, 16, 8]
        self.transformer = nn.ModuleList(
            [TransformerEncoderLayer(tdnn_nhid, num_heads[i],
                                     tdnn_nhid*4, 0.2,
                                     max_relative_positions=0)
             for i in range(3)])

        self.bn_final = nn.BatchNorm1d(tdnn_nhid)
        self.fc_out = nn.Linear(tdnn_nhid, output_dim)


    def forward(self, x, frame_offset=0):
        #x: batch, frame, dim
        bsz = x.size()[0]
        x = self.bn_in(F.relu(self.fc_in(x)).contiguous()
                       .view(-1, self.tdnn_nhid))
        x = x.contiguous().view(bsz, -1, self.tdnn_nhid)

        for l, (conv, bn) in enumerate(zip(self.hidden_conv, self.hidden_bn)):
            x = conv(x.unsqueeze(1))
            x = bn(F.relu(x).squeeze(-1)).transpose(1, 2).contiguous()
            if (l + 1) % 3 == 0:
                x = self.transformer[l//3](x, mask=None)
        x = self.bn_final(x.contiguous().view(-1, self.tdnn_nhid))
        x = self.fc_out(x)

        x = x.contiguous().view(bsz, -1, self.output_dim)
        return x[:, frame_offset:, :]
