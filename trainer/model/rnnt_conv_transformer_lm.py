"""
this module implements convolutional transformer
for transducer decoder
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from trainer.model.modules.transformer import TransformerEncoderLayer

class Net(nn.Module):
    """
    Class implements convolutonal transformer used for transducer
    decoder in "Minimum Bayes Risk Training of RNN-Transducer for
    End-to-End Speech Recognition", InterSpeech2020

    Args:
        embeddings (nn.module): embedding layer
        output_dim (int): output dimension
        d_model (int): d_model dimension used in transformer
        num_layers (int): number of convlution and transformer layers
        heads (int): number of heads in multi-head self-attention
        d_ff (int): size of the inner FF layer in transformer
        dropout (float): dropout probilities
        max_relative_positions (int): relative position encoding
        max_size (int): maximum input sequence length

    """
    def __init__(self, embeddings, output_dim, d_model,
                 num_layers, heads=8, d_ff=2048, dropout=0.1,
                 max_relative_positions=0, max_size=5000):
        super(Net, self).__init__()
        self.embeddings = embeddings
        self.output_dim = output_dim
        self.conv = nn.ModuleList(
            [nn.Conv1d(in_channels=embeddings.embedding_dim,
                       out_channels=d_model,
                       kernel_size=5,
                       padding=(5-1)*1)] + \
            [nn.Conv1d(in_channels=d_model,
                       out_channels=d_model,
                       kernel_size=5,
                       padding=(5-1)*1) for i in range(num_layers-1)]
        )
        self.transformer = nn.ModuleList(
            [TransformerEncoderLayer(
                d_model, heads, d_ff, dropout,
                max_relative_positions=max_relative_positions)
             for i in range(num_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.linear_out = nn.Linear(d_model, output_dim)

        mask = self._get_attn_subsequent_mask(max_size)
        # Register self.mask as a buffer .
        self.register_buffer('mask', mask)

    def forward(self, src, softmax=False):
        #src: batch, frame
        out = self.embeddings(src)
        # getting mask for current batch
        # Run the forward pass of every layer of the tranformer.
        src_batch, src_len = src.size()
        padding_idx = self.embeddings.padding_idx
        pad_mask = src.data.eq(padding_idx).unsqueeze(1)\
                   .expand(src_batch, src_len, src_len)
        mask = torch.gt(pad_mask + self.mask[:, :pad_mask.size(1), :pad_mask.size(1)]\
                                            .expand_as(pad_mask), 0)

        for conv, trans in zip(self.conv, self.transformer):
            #causal convolution
            out = F.relu(conv(out.transpose(1, 2))[:, :, :-conv.padding[0]])
            out = out.transpose(1, 2)
            out = trans(out, mask=mask)
        out = self.layer_norm(out)
        out = self.linear_out(out)
        if softmax:
            out = F.log_softmax(out, dim=-1)
        return out

    def _get_attn_subsequent_mask(self, size):
        ''' Get an attention mask to avoid using the subsequent info.'''
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        subsequent_mask = torch.from_numpy(subsequent_mask)
        return subsequent_mask
