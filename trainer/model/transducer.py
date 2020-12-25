"""
this module implements generic transducer
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from trainer.model.rnnt_tdnn_transformer import Net as encoder_tdnn
from trainer.model.rnnt_conv_transformer_lm import Net as decoder_transformer

class Net(nn.Module):
    """
    Class implements a generic transducer

    Args:
        opt (dict): model options
        input_dim (int): model input dimension
        output_dim (int): model output dimension

    """
    def __init__(self, opt, input_dim, output_dim):
        super(Net, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hid_dim = opt.rnn_size
        self.local_rank = opt.local_rank
        self.pack_seq = True
        self.decoder_type = opt.decoder_type
        hid_size_factor = 1
        if opt.brnn:
            hid_size_factor = 2
        if opt.encoder_type == 'rnn':
            self.encoder = nn.LSTM(input_size=input_dim,
                                   hidden_size=self.hid_dim//hid_size_factor,
                                   dropout=opt.dropout,
                                   num_layers=opt.enc_layers,
                                   bidirectional=opt.brnn,
                                   batch_first=True)
        else:
            self.encoder = encoder_tdnn(input_dim=input_dim,
                                        input_ctx=0, #dummy
                                        output_dim=self.hid_dim,
                                        tdnn_nhid=1024,
                                        tdnn_layers=9)
            self.pack_seq = False
        self.embed = nn.Embedding(output_dim+1, opt.embd_dim,
                                  padding_idx=opt.padding_idx)

        if opt.decoder_type == 'rnn':
            self.decoder = nn.LSTM(input_size=opt.embd_dim,
                                   hidden_size=self.hid_dim,
                                   dropout=opt.dropout,
                                   num_layers=opt.dec_layers,
                                   bidirectional=False,
                                   batch_first=True)
        else: #transformer
            self.decoder = decoder_transformer(embeddings=self.embed,
                                               output_dim=self.hid_dim,
                                               d_model=512,
                                               num_layers=opt.dec_layers,
                                               heads=8,
                                               d_ff=2048, dropout=opt.dropout)

        self.fc1 = nn.Linear(2*self.hid_dim, self.hid_dim)
        self.fc_gate = nn.Linear(2*self.hid_dim, self.hid_dim)
        self.fc2 = nn.Linear(self.hid_dim, output_dim)

    def forward(self, x, y, x_len=None, softmax=True):
        """
        Args:
            x(tensor): batch, frame, dim
            y(tensor): batch, frame
            x_len(tensor): batch
        """

        if self.pack_seq and x_len is not None:
            packed_x = pack(x, x_len, batch_first=True,
                            enforce_sorted=True)
            packed_x, _ = self.encoder(packed_x)
            x = unpack(packed_x, batch_first=True)[0]
        else:
            x = self.encoder(x)
        #prepend SOS, blk=0
        SOS = Variable(torch.zeros(y.shape[0], 1).long())
        SOS = SOS.cuda()
        y = torch.cat((SOS, y), dim=1)
        if self.decoder_type == 'rnn':
            y = self.embed(y)
            y, _ = self.decoder(y)
        else:
            y = self.decoder(y)
        T = x.size()[1]
        U = y.size()[1]
        #x: batch, T, U, dim
        #y: batch, T, U, dim
        x = x.unsqueeze(2).expand(-1, -1, U, -1)
        y = y.unsqueeze(1).expand(-1, T, -1, -1)
        #x_gate = F.glu(torch.cat((x, y), dim=-1), dim=-1)
        #y_gate = F.glu(torch.cat((y, x), dim=-1), dim=-1)
        #out = torch.cat((x_gate, y_gate), dim=-1)
        out = torch.cat((x, y), dim=-1)
        out = self.fc2(F.tanh(self.fc1(out)) * F.sigmoid(self.fc_gate(out)))
        #out = self.fc2(F.selu(self.fc1(out)))
        if softmax:
            out = F.log_softmax(out, dim=-1)
        return out

    def clean_hidden(self):
        """
        clean hidden interface
        """

    def reset_hidden(self, h, reset_idx):
        """
        reset hidden interface
        """
