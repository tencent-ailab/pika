"""
Main transducer decoder class
"""
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from decoder.beam_transducer import BeamMergeTransducer


class TransducerDecoder():
    """
    Uses a model to decode a batch of sentences.

    Args:
       model (:obj:`model.transducer`):
           transducer model to use for decoding
       batch_size:
       beam_size (int): size of beam to use
       n_best (int): number of best hypo produced
       blk: blank id
       global_scorer (:obj:`GlobalScorer`): object to rescore final hyps
       sm_scale: temperature used to smooth softmax
       cuda (bool): use cuda
       beam_prune (bool): prune beam that has redundant partial hyp
    """
    def __init__(self, model, batch_size,
                 beam_size, n_best=1,
                 blk=0,
                 global_scorer=None, sm_scale=1.0,
                 lm=None, lm_scale=1.0,
                 lm_scorer=None, lm_scorer_scale=1.0,
                 cuda=False, beam_prune=True, args=None):
        self.model = model
        self.batch_size = batch_size
        self.beam_size = beam_size
        self.n_best = n_best
        self.blk = blk
        self.global_scorer = global_scorer
        self.sm_scale = sm_scale
        self.cuda = cuda
        self.beam_prune = beam_prune
        #-1 to make sure first iteration will fetch 0th frame
        self.t_idx = None
        #dec_states (h, c): num_layer, batch, dim
        self.dec_states = None

        self.lm = lm
        self.lm_scale = lm_scale
        self.lm_states = None

        self.lm_scorer = lm_scorer
        self.lm_scorer_scale = lm_scorer_scale

        self.las_rescorer, self.las_rescorer_bw = None, None
        if args.las_rescorer is not None:
            self.las_rescorer = args.las_rescorer
        if args.las_rescorer_bw is not None:
            self.las_rescorer_bw = args.las_rescorer_bw
        if args.bilas_rescorer is not None:
            self.bilas_rescorer = args.bilas_rescorer

        self.args = args

    def decode_batch(self, x, x_len, max_len=None):
        """
        decode a batch of sentences.
        Args:
            x         : batch_size, seq_len, featdim
            x_len     : LongTensor()
            max_len   : List of Int if not None
        """
        # (0) Prep each of the components of the search.
        # And helper method for reducing verbosity.
        beam_size = self.beam_size
        #batch_size = self.batch_size
        batch_size = x.size()[0]
        beam = []
        beam = [BeamMergeTransducer(beam_size, self.blk, n_best=self.n_best,
                                    cuda=self.cuda,
                                    global_scorer=self.global_scorer,
                                    max_len=max_len[i] if max_len[i] else 10000,
                                    lm_scorer=self.lm_scorer,
                                    lm_scorer_scale=self.lm_scorer_scale,
                                    beam_prune=self.beam_prune,
                                    args=self.args)
                for i in range(batch_size)]

        # (1) Run the encoder on the src.
        #x: bsz, frame, dim
        if self.model.pack_seq and x_len is not None:
            packed_x = pack(x, x_len, batch_first=True,
                            enforce_sorted=True)
            packed_x, _ = self.model.encoder(packed_x)
            #saved for returning shared encoder output
            enc_out = unpack(packed_x, batch_first=True)[0]
        else:
            #saved for returning shared encoder output
            enc_out = self.model.encoder(x)
        batch_size = enc_out.size()[0]
        # (2) Repeat x, x_len `beam_size` times.
        #x: beam * bsz, frame, dim
        x = enc_out.repeat(beam_size, 1, 1)
        blk_sos = torch.LongTensor(batch_size*beam_size, 1).zero_() + self.blk
        # (3) run the decoder to generate sentences, using beam search.
        self.t_idx = torch.LongTensor(beam_size, batch_size).zero_() - 1
        if self.cuda:
            self.t_idx = self.t_idx.cuda()
            blk_sos = blk_sos.cuda()
        #self.dec_states initialization:
        #FloatTensor when decoder is Transformer based
        #which needs full input history
        #tuple of FloatTensors when decoder is RNN based
        #which only needs last step state (see torch LSTM interface)
        if self.model.decoder_type == 'rnn':
            _, self.dec_states = self.model.decoder(self.model.embed(blk_sos))
        else:
            #decoder supposed to be transformer based structure
            #self.dec_states: beam_size*batch_size, dim
            self.dec_states = self.model.decoder(blk_sos)[:, -1, :]

        while not all((b.done() for b in beam)):
            # Construct beam_size * batch last words.
            # Get all the pending current beam words and arrange for forward.
            # inp: beam_size, batch
            inp = torch.stack([b.get_current_state() for b in beam]).t()
            # t_idx increased by 1 if blk
            self.t_idx = self.t_idx + inp.eq(self.blk).long()
            inp = inp.contiguous().view(-1, 1)
            # fetch encoder hidden state
            # beam * bsz, dim
            enc_hid = x[torch.arange(batch_size*beam_size),
                        self.t_idx.contiguous().view(-1), :]
            if self.model.decoder_type == 'rnn':
                dec_hid = self.dec_states[0][-1, :, :]
            else:
                dec_hid = self.dec_states
            nonblk = inp.view(-1).gt(self.blk)
            if nonblk.sum().item() > 0:
                # Run one step for decoder.
                # only update dec_states if non-blk
                # Note: blk == 0, eos == -1
                if self.model.decoder_type == 'rnn':
                    dec_in = self.model.embed(inp[nonblk, :])
                    _, (self.dec_states[0][:, nonblk, :], self.dec_states[1][:, nonblk, :])\
                        = self.model.decoder(dec_in, (self.dec_states[0][:, nonblk, :],
                                                      self.dec_states[1][:, nonblk, :]))
                    #dec_hid: beam * bsz, dim
                    dec_hid = self.dec_states[0][-1, :, :]
                else:
                    #decoder supposed to be transformer based structure
                    partial_hyp_idx = torch.arange(batch_size*beam_size)[nonblk]
                    partial_hyp = []
                    hyps_len = []
                    for k in partial_hyp_idx:
                        cur_hyp = [self.blk] + beam[k%batch_size]\
                                      .get_current_hyp(k//batch_size)
                        partial_hyp.append(cur_hyp)
                        hyps_len.append(len(cur_hyp))
                    hyp_max_len = max(hyps_len)
                    for i in range(len(partial_hyp)):
                        partial_hyp[i].extend((hyp_max_len - hyps_len[i])*\
                                              [self.model.embed.padding_idx])
                    #partial_hyp: bb, frame
                    partial_hyp = torch.cuda.LongTensor(partial_hyp)
                    self.dec_states[nonblk, :] = self.model.decoder(partial_hyp)\
                        [torch.arange(partial_hyp.size()[0]),
                         torch.LongTensor(hyps_len) - 1, :]
                    #dec_hid: beam * bsz, dim
                    dec_hid = self.dec_states

            out = torch.cat((enc_hid, dec_hid), dim=-1)
            out = self.model.fc2(F.tanh(self.model.fc1(out)) *\
                  F.sigmoid(self.model.fc_gate(out)))
            # (b) Compute a vector of beam_size * bsz word scores.
            out = F.log_softmax(self.sm_scale*out, dim=-1)
            out = out.contiguous().view(beam_size, batch_size, -1)

            # (c) Advance each beam.
            for j, b in enumerate(beam):
                b.advance(out[:, j], self.t_idx[:, j], x_len[j])
                self._beam_update(j, b.get_current_origin(), beam_size)
        # (4) Extract sentences from beam.
        ret = self._from_beam(beam)
        return ret, enc_out

    def _beam_update(self, idx, positions, beam_size):
        if isinstance(self.dec_states, tuple):
            for e in self.dec_states:
                a, br, d = e.size()
                sent_states = e.view(a, beam_size, br // beam_size, d)[:, :, idx]
                sent_states.data.copy_(
                    sent_states.data.index_select(1, positions))
        else:
            br, d = self.dec_states.size()
            sent_states = self.dec_states.\
                          view(beam_size, br // beam_size, d)[:, idx]
            sent_states.data.copy_(
                sent_states.data.index_select(0, positions))
        cur_t_idx = self.t_idx.view(beam_size, -1)[:, idx]
        cur_t_idx.data.copy_(cur_t_idx.data.index_select(0, positions))

    def _from_beam(self, beam):
        ret = {"predictions": [],
               "scores": []}
        for b in beam:
            n_best = self.n_best
            scores, ks = b.sort_finished(minimum=n_best)
            hyps = []
            for _, (times, k) in enumerate(ks[:n_best]):
                hyp = b.get_hyp(times, k)
                #strip the ending eos(-1)
                hyps.append(hyp[:-1])
            ret["predictions"].append(hyps)
            ret["scores"].append(scores[:n_best])
        return ret

    def las_rescore(self, x, tgt, bw=False):
        """
        Args:
            x: T, B, C
            tgt: T, B, C
        assert B == 1
        """
        lens = torch.IntTensor([x.size(0)])
        if not bw:
            outputs, _, _, _ = self.las_rescorer(x, tgt, lens)
            outputs = F.log_softmax(self.las_rescorer.dec_proj(outputs), dim=-1)
        else:
            outputs, _, _, _ = self.las_rescorer_bw(x, tgt, lens)
            outputs = F.log_softmax(self.las_rescorer_bw.dec_proj(outputs), dim=-1)
        #outputs: T, C
        outputs = outputs.squeeze(1)
        tgt_idx = tgt[1:].squeeze(-1).squeeze(-1)
        return outputs[torch.arange(tgt_idx.size(0)), tgt_idx].tolist()

    def bilas_rescore(self, x, tgt):
        """
        Args:
            x: T, B, C
            tgt: T, B, C
        #assert B == 1
        """
        lens = torch.IntTensor([x.size(0)])
        ali_lens = torch.IntTensor([tgt.size(0)])
        outputs, _, _, _ = self.bilas_rescorer(x, tgt, lens, None,
                                               True, True, ali_lens)
        outputs = F.log_softmax(0.5*self.bilas_rescorer.dec_proj(outputs), dim=-1)
        #outputs: T, C
        outputs = outputs.squeeze(1)
        tgt_idx = tgt[1:].squeeze(-1).squeeze(-1)
        return outputs[torch.arange(tgt_idx.size(0)), tgt_idx].tolist()
