"""
this module implements generic LAS seq2seq model

"""

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from trainer.model.modules.stacked_rnn import StackedLSTM, StackedGRU
from trainer.model.modules.global_attention import GlobalAttention
from trainer.model.modules.sru import SRU
from trainer.model.modules.context_gate import context_gate_factory
########### LAS definition, ################
###Modified from ONMT NMTModel definition###
class Net(nn.Module):
    """
    Class implements a generic LAS seq2seq model

    Args:
        opt (dict): model options
        input_dim (int): model input dimension
        output_dim (int): model output dimension
        pad_idx (int): padding index
    """
    def __init__(self, opt, input_dim, output_dim, pad_idx):
        super(Net, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hid_dim = opt.rnn_size

        self.encoder = make_las_encoder(opt, input_dim)
        self.enc_proj = nn.Linear(opt.rnn_size, output_dim)

        if opt.use_downsampler:
            self.downsampler = make_las_downsampler(opt)
        else:
            self.downsampler = None

        self.tgt_embeddings = LASEmbeddings(opt, output_dim, pad_idx)

        self.dec_proj = nn.Linear(opt.rnn_size, output_dim)
        self.decoder = make_las_decoder(opt, self.tgt_embeddings, self.dec_proj, pad_idx)



    def forward(self, src, tgt, lengths, dec_state=None, enable_dec=True, enable_enc=True):
        """
        Args:
            src(FloatTensor): a sequence of source tensors with
                    (len x batch x featdim).
            tgt(LongTensor): a sequence of target tensors with
                    (len x batch).
            lengths([int]): an array of the src length.
            dec_state: A decoder state object
        Returns:
            outputs (FloatTensor): [(len x batch), hidden_size]: decoder outputs
            attns (FloatTensor): Dictionary of (src_len x batch)
            dec_hidden (FloatTensor): tuple (1 x batch x hidden_size)
                                      Init hidden state
            enc_out (FloatTensor): encoder outputs
        """
        # exclude EOS for inputs
        tgt = tgt[:-1]
        if not enable_enc:
            #we are now doing decoder pretraining,
            outputs = self._pretrain_inputfeed_decoder(tgt)
            return outputs, None, None, None

        # encoder forward
        enc_hidden, enc_out = self.encoder(src, lengths)
        # decoder forward
        if enable_dec:
            ds_out = enc_out
            ds_hidden = enc_hidden
            if self.downsampler is not None:
                ds_hidden, ds_out, _ = self.downsampler(ds_out, lengths)
            enc_state = self.decoder.init_decoder_state(src, ds_out, ds_hidden)
            #for memory efficiency, delay projection forward
            #when computing loss function see LASLossCompute
            out, dec_state, attns = self.decoder(tgt, ds_out,
                                                 enc_state if dec_state is None
                                                 else dec_state)
            return out, attns, dec_state, enc_out
        #encoder pretraining
        return None, None, None, enc_out

    def _pretrain_inputfeed_decoder(self, tgt):
        """
        pretraining input-feeding decoder with LM task

        Args:
            tgt (FloatTensor): target labels

        """
        emb = self.tgt_embeddings(tgt)
        _, batch_, _ = emb.size()
        #pad = Variable(emb.data.new(len_, batch_, self.hid_dim).zero_(),
        #               requires_grad=False)
        #emb = torch.cat([emb, pad], 2)
        h = Variable(emb.data.new(self.decoder.num_layers, batch_, self.hid_dim).zero_())
        c = Variable(emb.data.new(self.decoder.num_layers, batch_, self.hid_dim).zero_())
        hidden = (h, c)
        output = Variable(emb.data.new(batch_, self.hid_dim).zero_(), requires_grad=False)
        outputs = []
        for _, emb_t in enumerate(emb.split(1)):
            emb_t = emb_t.squeeze(0)
            emb_t = torch.cat([emb_t, output], 1)
            output, hidden = self.decoder.rnn(emb_t, hidden)
            outputs += [output]
        outputs = torch.stack(outputs)
        return outputs

def make_las_encoder(opt, input_dim):
    """
    LAS Encoder Creator Factory function

    Args:
        opt (dict): model options
        input_dim (dim): model input dimension

    """
    if opt.encoder_type == "cnn":
        raise NotImplementedError
    elif opt.encoder_type == "pyramid_rnn":
        raise NotImplementedError
    elif opt.encoder_type == "rnn":
        return LASRNNEncoder(opt.rnn_type, opt.brnn, opt.enc_layers,
                             opt.rnn_size, opt.dropout, input_dim)
    else:
        raise NotImplementedError

def make_las_decoder(opt, embeddings, proj, pad_idx):
    """
    LAS Decoder Creator Factory function

    Args:
        opt (dict): model options
        embeddings (nn.module): embedding layer.
        proj (nn.module): projection layer.
        pad_idx (int): padding index
    """
    if opt.num_heads > 1:
        opt.input_feed_multihead = 1
        opt.input_feed = 0

    if opt.sampling_decoder:
        return InputFeedSamplingRNNDecoder(opt.rnn_type, opt.brnn,
                                           opt.dec_layers, opt.rnn_size,
                                           proj, pad_idx,
                                           opt.global_attention,
                                           opt.coverage_attn,
                                           opt.context_gate,
                                           opt.copy_attn,
                                           opt.dropout,
                                           embeddings, opt.sampling_prob)

    if opt.input_feed:
        return InputFeedRNNDecoder(opt.rnn_type, opt.brnn,
                                   opt.dec_layers, opt.rnn_size,
                                   opt.global_attention,
                                   opt.coverage_attn,
                                   opt.context_gate,
                                   opt.copy_attn,
                                   opt.dropout,
                                   embeddings)

def make_las_downsampler(opt):
    """
    LAS downsampler Creator Factory function

    Args:
        opt (dict): model options
    """
    if opt.downsampler_type == "rnn":
        return PyramidRNN(opt.rnn_size, opt.rnn_size, opt.brnn,
                          opt.downsampler_layers, opt.downsampler_rate,
                          opt.dropout)
    elif opt.downsampler_type == "dnn":
        raise NotImplementedError
    elif opt.downsampler_type == "cnn":
        raise NotImplementedError
    else:
        raise NotImplementedError

class PyramidRNN(nn.Module):
    """
    Class implements a pyramidRNN, i.e, downsampling
    along temporal axis at upper layers

    Args:
        input_dim (int): input dimension
        hid_dim (int): hidden dimension
        bidirectional (bool): uni- or bi-directional
        num_layers (int): number of layers
        subsample_rate (int): subsample rate along temporal axis
        dropout (float): dropout probability

    """
    def __init__(self, input_dim, hid_dim, bidirectional,
                 num_layers, subsample_rate, dropout):
        super(PyramidRNN, self).__init__()
        self.input_dim = input_dim * subsample_rate
        if bidirectional:
            self.hid_dim = hid_dim // 2
        else:
            self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.subsample_rate = subsample_rate
        self.rnn = nn.LSTM(self.input_dim, self.hid_dim, num_layers,
                           batch_first=True, dropout=dropout,
                           bidirectional=bidirectional)

    def forward(self, input, lengths=None, hidden=None):
        """
        Args:
            input (tensor): T, B, C
            lengths (LongTensor): B
        """
        #input: [seq_len, batch, dim]
        l, b, d = input.size()
        out_len = (l - 1) // self.subsample_rate + 1
        pad_len = out_len * self.subsample_rate - l
        if pad_len > 0:
            padding = Variable(input.data.new(pad_len, b, d).zero_())
            input = torch.cat([input, padding], 0)
        input = input.transpose(0, 1).contiguous().view(b, out_len, -1)
        packed_input = input
        if lengths is not None:
            lengths = [(len - 1) // self.subsample_rate + 1 for len in lengths]
            packed_input = pack(input, lengths, batch_first=True,
                                enforce_sorted=False)

        outputs, hidden_t = self.rnn(packed_input, hidden)

        if lengths is not None:
            outputs = unpack(outputs, batch_first=True)[0]

        outputs = outputs.transpose(0, 1)

        return hidden_t, outputs, lengths



class EncoderBase(nn.Module):
    """
    Base encoder class. Specifies the interface used by different encoder types
    and required by :obj:`onmt.Models.NMTModel`.

    .. mermaid::

       graph BT
          A[Input]
          subgraph RNN
            C[Pos 1]
            D[Pos 2]
            E[Pos N]
          end
          F[Context]
          G[Final]
          A-->C
          A-->D
          A-->E
          C-->F
          D-->F
          E-->F
          E-->G
    """
    def _check_args(self, input, lengths=None, hidden=None):
        _, n_batch, _ = input.size()
        if lengths is not None:
            n_batch_, = lengths.size()
            assert n_batch == n_batch_

    def forward(self, input, lengths=None, hidden=None):
        """
        Args:
            input (:obj:`LongTensor`):
               padded sequences of sparse indices `[src_len x batch x nfeat]`
            lengths (:obj:`LongTensor`): length of each sequence `[batch]`
            hidden (class specific):
               initial hidden state.

        Returns:k
            (tuple of :obj:`FloatTensor`, :obj:`FloatTensor`):
                * final encoder state, used to initialize decoder
                   `[layers x batch x hidden]`
                * contexts for attention, `[src_len x batch x hidden]`
        """
        raise NotImplementedError


class DecoderState():
    """Interface for grouping together the current state of a recurrent
    decoder. In the simplest case just represents the hidden state of
    the model.  But can also be used for implementing various forms of
    input_feeding and non-recurrent models.

    Modules need to implement this to utilize beam search decoding.
    """
    def detach(self):
        """
        detach variables from computation graph
        """
        for h in self._all:
            if h is not None:
                h.detach_()

    def beam_update(self, idx, positions, beam_size):
        """
        update beam according to positions
        """
        for e in self._all:
            a, br, d = e.size()
            sent_states = e.view(a, beam_size, br // beam_size, d)[:, :, idx]
            sent_states.data.copy_(
                sent_states.data.index_select(1, positions))


class RNNDecoderState(DecoderState):
    """Interface for grouping together the current stat of
       a RNN decoder,
    """
    def __init__(self, context, hidden_size, rnnstate):
        """
        Args:
            context (FloatTensor): output from the encoder of size
                                   len x batch x rnn_size.
            hidden_size (int): the size of hidden layer of the decoder.
            rnnstate (Variable): final hidden state from the encoder.
                transformed to shape: layers x batch x (directions*dim).
            input_feed (FloatTensor): output from last layer of the decoder.
            coverage (FloatTensor): coverage output from the decoder.
        """
        if not isinstance(rnnstate, tuple):
            self.hidden = (rnnstate,)
        else:
            self.hidden = rnnstate
        self.coverage = None

        # Init the input feed.
        batch_size = context.size(1)
        h_size = (batch_size, hidden_size)
        self.input_feed = Variable(context.data.new(*h_size).zero_(),
                                   requires_grad=False).unsqueeze(0)

    @property
    def _all(self):
        return self.hidden + (self.input_feed,)

    def update_state(self, rnnstate, input_feed, coverage):
        """update state"""
        if not isinstance(rnnstate, tuple):
            self.hidden = (rnnstate,)
        else:
            self.hidden = rnnstate
        self.input_feed = input_feed
        self.coverage = coverage

    def repeat_beam_size_times(self, beam_size):
        """ Repeat beam_size times along batch dimension. """
        vars = [Variable(e.data.repeat(1, beam_size, 1), volatile=True)
                for e in self._all]
        self.hidden = tuple(vars[:-1])
        self.input_feed = vars[-1]


class RNNDecoderBase(nn.Module):
    """
    Base recurrent attention-based decoder class.
    Specifies the interface used by different decoder types
    and required by :obj:`onmt.Models.NMTModel`.


    .. mermaid::

       graph BT
          A[Input]
          subgraph RNN
             C[Pos 1]
             D[Pos 2]
             E[Pos N]
          end
          G[Decoder State]
          H[Decoder State]
          I[Outputs]
          F[Context]
          A--emb-->C
          A--emb-->D
          A--emb-->E
          H-->C
          C-- attn --- F
          D-- attn --- F
          E-- attn --- F
          C-->I
          D-->I
          E-->I
          E-->G
          F---I

    Args:
       rnn_type (:obj:`str`):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional_encoder (bool) : use with a bidirectional encoder
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       attn_type (str) : see :obj:`onmt.modules.GlobalAttention`
       coverage_attn (str): see :obj:`onmt.modules.GlobalAttention`
       context_gate (str): see :obj:`onmt.modules.ContextGate`
       copy_attn (bool): setup a separate copy attention mechanism
       dropout (float) : dropout value for :obj:`nn.Dropout`
       embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
    """
    def __init__(self, rnn_type, bidirectional_encoder, num_layers,
                 hidden_size, attn_type="general",
                 coverage_attn=False, context_gate=None,
                 copy_attn=False, dropout=0.0, embeddings=None):
        super(RNNDecoderBase, self).__init__()

        # Basic attributes.
        self.decoder_type = 'rnn'
        self.bidirectional_encoder = bidirectional_encoder
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embeddings = embeddings
        self.dropout = nn.Dropout(dropout)

        # Build the RNN.
        self.rnn = self._build_rnn(rnn_type, self._input_size, hidden_size,
                                   num_layers, dropout)

        # Set up the context gate.
        self.context_gate = None
        if context_gate is not None:
            self.context_gate = context_gate_factory(
                context_gate, self._input_size,
                hidden_size, hidden_size, hidden_size
            )

        # Set up the standard attention.
        self._coverage = coverage_attn
        self.attn = GlobalAttention(
            hidden_size, coverage=coverage_attn,
            attn_type=attn_type
        )

        # Set up a separated copy attention layer, if needed.
        self._copy = False
        if copy_attn:
            self.copy_attn = GlobalAttention(
                hidden_size, attn_type=attn_type
            )
            self._copy = True

    def forward(self, input, context, state, context_lengths=None):
        """
        Args:
            input (`LongTensor`): sequences of padded tokens
                                `[tgt_len x batch x nfeats]`.
            context (`FloatTensor`): vectors from the encoder
                 `[src_len x batch x hidden]`.
            state (:obj:`onmt.Models.DecoderState`):
                 decoder state object to initialize the decoder
            context_lengths (`LongTensor`): the padded source lengths
                `[batch]`.
        Returns:
            (`FloatTensor`,:obj:`onmt.Models.DecoderState`,`FloatTensor`):
                * outputs: output from the decoder
                         `[tgt_len x batch x hidden]`.
                * state: final hidden state from the decoder
                * attns: distribution over src at each tgt
                        `[tgt_len x batch x src_len]`.
        """
        # Args Check
        assert isinstance(state, RNNDecoderState)
        _, input_batch, _ = input.size()
        _, contxt_batch, _ = context.size()
        assert input_batch == contxt_batch
        # END Args Check

        # Run the forward pass of the RNN.
        hidden, outputs, attns, coverage = self._run_forward_pass(
            input, context, state, context_lengths=context_lengths)

        # Update the state with the result.
        final_output = outputs[-1]
        state.update_state(hidden, final_output.unsqueeze(0),
                           coverage.unsqueeze(0)
                           if coverage is not None else None)

        # Concatenates sequence of tensors along a new dimension.
        outputs = torch.stack(outputs)
        for k in attns:
            attns[k] = torch.stack(attns[k])

        return outputs, state, attns

    def _fix_enc_hidden(self, h):
        """
        The encoder hidden is  (layers*directions) x batch x dim.
        We need to convert it to layers x batch x (directions*dim).
        """
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    def init_decoder_state(self, src, context, enc_hidden):
        """initialize decoder state with encoder last status"""
        if isinstance(enc_hidden, tuple):  # LSTM
            return RNNDecoderState(context, self.hidden_size,
                                   tuple([self._fix_enc_hidden(enc_hidden[i])
                                          for i in range(len(enc_hidden))]))
        else:  # GRU
            return RNNDecoderState(context, self.hidden_size,
                                   self._fix_enc_hidden(enc_hidden))


class LASRNNEncoder(EncoderBase):
    """ The standard RNN encoder. """
    def __init__(self, rnn_type, bidirectional, num_layers,
                 hidden_size, dropout, input_dim):
        super(LASRNNEncoder, self).__init__()

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions
        self.no_pack_padded_seq = False

        # Use pytorch version when available.
        if rnn_type == "SRU":
            # SRU doesn't support PackedSequence.
            self.no_pack_padded_seq = True
            self.rnn = SRU(input_size=input_dim,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           dropout=dropout,
                           bidirectional=bidirectional)
        else:
            self.rnn = getattr(nn, rnn_type)(input_size=input_dim,
                                             hidden_size=hidden_size,
                                             num_layers=num_layers,
                                             dropout=dropout,
                                             bidirectional=bidirectional)

    def forward(self, input, lengths=None, hidden=None):
        """ See EncoderBase.forward() for description of args and returns."""
        self._check_args(input, lengths, hidden)
        #s_len, batch, input_dim = input.size()

        packed_input = input
        if lengths is not None and not self.no_pack_padded_seq:
            # Lengths data is wrapped inside a Variable.
            lengths = lengths.view(-1).tolist()
            packed_input = pack(input, lengths, enforce_sorted=False)

        outputs, hidden_t = self.rnn(packed_input, hidden)

        if lengths is not None and not self.no_pack_padded_seq:
            outputs = unpack(outputs)[0]

        return hidden_t, outputs


class LASEmbeddings(nn.Module):
    """
    a wrapper for embeddings of targets
    mainly add .embedding_size property
    """
    def __init__(self, opt, output_dim, pad_idx):

        super(LASEmbeddings, self).__init__()
        self.padding_idx = pad_idx
        self.num_embeddings = output_dim
        self.embedding_size = opt.embd_dim
        #plus one for padded embedding
        self.embeddings = nn.Embedding(output_dim + 1,
                                       opt.embd_dim,
                                       padding_idx=pad_idx)

    def forward(self, input):
        input = torch.squeeze(input, 2)
        return self.embeddings(input)


#



class InputFeedRNNDecoder(RNNDecoderBase):
    """
    Input feeding based decoder. See :obj:`RNNDecoderBase` for options.

    Based around the input feeding approach from
    "Effective Approaches to Attention-based Neural Machine Translation"
    :cite:`Luong2015`


    .. mermaid::

       graph BT
          A[Input n-1]
          AB[Input n]
          subgraph RNN
            E[Pos n-1]
            F[Pos n]
            E --> F
          end
          G[Encoder]
          H[Context n-1]
          A --> E
          AB --> F
          E --> H
          G --> H
    """

    def _run_forward_pass(self, input, context, state, context_lengths=None):
        """
        See StdRNNDecoder._run_forward_pass() for description
        of arguments and return values.
        """
        # Additional args check.
        output = state.input_feed.squeeze(0)
        output_batch, _ = output.size()
        input_len, input_batch, _ = input.size()
        assert input_batch == output_batch
        # END Additional args check.

        # Initialize local and return variables.
        outputs = []
        attns = {"std": []}
        if self._copy:
            attns["copy"] = []
        if self._coverage:
            attns["coverage"] = []

        emb = self.embeddings(input)
        assert emb.dim() == 3  # len x batch x embedding_dim

        hidden = state.hidden
        coverage = state.coverage.squeeze(0) \
            if state.coverage is not None else None

        # Input feed concatenates hidden state with
        # input at every time step.
        for _, emb_t in enumerate(emb.split(1)):
            emb_t = emb_t.squeeze(0)
            emb_t = torch.cat([emb_t, output], 1)

            rnn_output, hidden = self.rnn(emb_t, hidden)
            attn_output, attn = self.attn(
                rnn_output,
                context.transpose(0, 1),
                context_lengths=context_lengths)
            if self.context_gate is not None:
                # context gate should be employed
                # instead of second RNN transform.
                output = self.context_gate(
                    emb_t, rnn_output, attn_output
                )
                output = self.dropout(output)
            else:
                output = self.dropout(attn_output)
            outputs += [output]
            attns["std"] += [attn]

            # Update the coverage attention.
            if self._coverage:
                coverage = coverage + attn \
                    if coverage is not None else attn
                attns["coverage"] += [coverage]

            # Run the forward pass of the copy attention layer.
            if self._copy:
                _, copy_attn = self.copy_attn(output,
                                              context.transpose(0, 1))
                attns["copy"] += [copy_attn]

        # Return result.
        return hidden, outputs, attns, coverage

    def _build_rnn(self, rnn_type, input_size,
                   hidden_size, num_layers, dropout):
        assert not rnn_type == "SRU", "SRU doesn't support input feed! " \
                "Please set -input_feed 0!"
        if rnn_type == "LSTM":
            stacked_cell = StackedLSTM
        else:
            stacked_cell = StackedGRU
        return stacked_cell(num_layers, input_size,
                            hidden_size, dropout)

    @property
    def _input_size(self):
        """
        Using input feed by concatenating input with attention vectors.
        """
        return self.embeddings.embedding_size + self.hidden_size



class InputFeedSamplingRNNDecoder(InputFeedRNNDecoder):
    """
    Input feeding based decoder with the support of
    sheduled sampling
    """
    def __init__(self, rnn_type, bidirectional_encoder, num_layers,
                 hidden_size, proj, padding_idx, attn_type="mlp",
                 coverage_attn=False, context_gate=None,
                 copy_attn=False, dropout=0.0, embeddings=None,
                 sampling_prob=0.0):

        super(InputFeedSamplingRNNDecoder, self).\
              __init__(rnn_type, bidirectional_encoder, num_layers,
                       hidden_size, attn_type, coverage_attn, context_gate,
                       copy_attn, dropout, embeddings)

        self.proj = proj
        self.sampling_prob = sampling_prob
        self.padding_idx = padding_idx

    def set_sampling_prob(self, sampling_prob):
        self.sampling_prob = sampling_prob

    def _run_forward_pass(self, input, context, state, context_lengths=None):
        """
        See StdRNNDecoder._run_forward_pass() for description
        of arguments and return values.
        """
        # Additional args check.
        output = state.input_feed.squeeze(0)
        output_batch, _ = output.size()
        input_len, input_batch, _ = input.size()
        assert input_batch == output_batch
        # END Additional args check.

        # Initialize local and return variables.
        outputs = []
        attns = {"std": []}
        if self._copy:
            attns["copy"] = []
        if self._coverage:
            attns["coverage"] = []

        emb = self.embeddings(input)
        assert emb.dim() == 3  # len x batch x embedding_dim

        hidden = state.hidden
        coverage = state.coverage.squeeze(0) \
            if state.coverage is not None else None

        # Input feed concatenates hidden state with
        # input at every time step.
        for i, emb_t in enumerate(emb.split(1)):
            if self.sampling_prob > 0.0:
                toss = np.random.uniform(0.0, 1.0)
                if i > 0 and  toss < self.sampling_prob:
                    #sample previous symbols
                    _, symbols = torch.max(self.proj(output).data, 1)
                    #exclude eos and padding symbol
                    #input - seq_len, batch, dim
                    true_input_t = input[i].data.squeeze(1)
                    mask = torch.lt(true_input_t, self.padding_idx)
                    mask = mask * torch.gt(true_input_t, 1)
                    true_input_t[mask] = symbols[mask]
                    true_input_t = Variable(true_input_t)
                    emb_t = self.embeddings(true_input_t.unsqueeze(0).unsqueeze(2))
            emb_t = emb_t.squeeze(0)
            emb_t = torch.cat([emb_t, output], 1)

            rnn_output, hidden = self.rnn(emb_t, hidden)
            attn_output, attn = self.attn(
                rnn_output,
                context.transpose(0, 1),
                context_lengths=context_lengths)
            if self.context_gate is not None:
                # context gate should be employed
                # instead of second RNN transform.
                output = self.context_gate(
                    emb_t, rnn_output, attn_output
                )
                output = self.dropout(output)
            else:
                output = self.dropout(attn_output)
            outputs += [output]
            attns["std"] += [attn]

            # Update the coverage attention.
            if self._coverage:
                coverage = coverage + attn \
                    if coverage is not None else attn
                attns["coverage"] += [coverage]

            # Run the forward pass of the copy attention layer.
            if self._copy:
                _, copy_attn = self.copy_attn(output,
                                              context.transpose(0, 1))
                attns["copy"] += [copy_attn]

        # Return result.
        return hidden, outputs, attns, coverage
