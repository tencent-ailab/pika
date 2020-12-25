"""
main decoding script
"""
import argparse
import sys
import importlib
import torch
import numpy as np

from kaldi.matrix import _matrix_ext, Vector, DoubleMatrix
import kaldi.fstext as fst
from kaldi.util import io

from decoder.beam_transducer import GlobalScorer
from decoder.transducer_decoder import TransducerDecoder
from decoder.sorted_matcher import SortedMatcher

def main():
    model = torch.load(args.model, map_location=lambda storage, loc: storage)
    model.eval()

    args.las_rescorer = None
    if args.las_rescorer_model is not None:
        args.las_rescorer = torch.load(args.las_rescorer_model,
                                       map_location=lambda storage, loc: storage)
        args.las_rescorer.eval()

    args.las_rescorer_bw = None
    if args.las_rescorer_bw_model is not None:
        args.las_rescorer_bw = torch.load(args.las_rescorer_bw_model,
                                          map_location=lambda storage, loc: storage)
        args.las_rescorer_bw.eval()

    args.bilas_rescorer = None
    if args.bilas_rescorer_model is not None:
        args.bilas_rescorer = torch.load(args.bilas_rescorer_model,
                                         map_location=lambda storage, loc: storage)
        args.bilas_rescorer.eval()

    if torch.cuda.is_available():
        if args.cuda:
            model.cuda()
            if args.las_rescorer is not None:
                args.las_rescorer.cuda()
            if args.las_rescorer_bw is not None:
                args.las_rescorer_bw.cuda()
            if args.bilas_rescorer is not None:
                args.bilas_rescorer.cuda()
        else:
            print('WARN: you have a cuda device, '
                  'you should probably run with --cuda', file=sys.stderr)
    #global rescorer with converage and output length

    #load cmvn states
    if args.cmvn_stats:
        with io.Input(args.cmvn_stats, binary=False) as ki:
            stats = DoubleMatrix().read_(ki.stream(), ki.binary)
            cmvn = _matrix_ext.double_matrix_to_numpy(stats)
            mean = cmvn[0][:-1] / cmvn[0][-1]
            var = cmvn[1][:-1] / cmvn[0][-1] - (mean*mean)
            floor = 1.0e-20
            if min(abs(var)) < floor:
                print('problematic cmvn_stats, variance too small')
                sys.exit()
            args.offset = torch.from_numpy(-mean).cuda(args.local_rank)
            args.offset = args.offset.repeat(args.lctx + args.rctx + 1)
            args.scale = torch.from_numpy(1.0 / np.sqrt(var))
            args.scale = args.scale.cuda(args.local_rank)
            args.scale = args.scale.repeat(args.lctx + args.rctx + 1)



    scorer = GlobalScorer()
    nnlm = None
    #if args.lm != '':
    #    lm = torch.load(args.lm, map_location=lambda storage, loc: storage)
    #    lm.eval()
    #    if torch.cuda.is_available():
    #        lm.cuda()
    lm_scorer = None
    fst_lm = None
    if args.fst_lm != '':
        fst_lm = fst.StdVectorFst.read(args.fst_lm)
        disambig_ids = [int(i) for i in args.disambig_ids.split(',')]
        lm_scorer = SortedMatcher(fst_lm, args.max_num_arcs, args.max_id,
                                  args.backoff_id, disambig_ids)

    trans_decoder = TransducerDecoder(model, batch_size=args.batch_size,
                                      beam_size=args.beam_size,
                                      n_best=args.n_best,
                                      blk=args.blk,
                                      global_scorer=scorer,
                                      sm_scale=args.sm_scale,
                                      lm=nnlm,
                                      lm_scale=args.lm_scale,
                                      lm_scorer=lm_scorer,
                                      lm_scorer_scale=args.fst_lm_scale,
                                      cuda=args.cuda, beam_prune=True,
                                      args=args)

    with open(args.symbols_map, 'r', encoding='utf-8') as f:
        sym_map = dict()
        for line in f:
            entry = line.split(" ")
            sym_map[int(entry[1])] = entry[0]

    with open(args.output_file, 'w') as f:
        for data_batch, _, len_batch, _ in \
            args.dataloader(args.input_labels, args.input_specifier, False, args):
            if args.cuda:
                len_batch = torch.from_numpy(len_batch).cuda()
            else:
                len_batch = torch.from_numpy(len_batch)
            if len_batch.max() < args.min_len:
                pad = data_batch[:, -1, :].unsqueeze(1)
                pad = pad.expand(-1, args.min_len - len_batch.max(), -1)
                #pad = torch.zeros_like(pad)

                data_batch = torch.cat((data_batch, pad), dim=1)
                len_batch[:] = args.min_len
            if args.cmvn_stats:
                #cmn#
                if args.cmn:
                    data_batch -= data_batch.mean(dim=1).unsqueeze(dim=1)

                data_batch += args.offset.unsqueeze(dim=0).unsqueeze(dim=0)
                data_batch *= args.scale.unsqueeze(dim=0).unsqueeze(dim=0)

            len_batch = len_batch - args.model_lctx - args.model_rctx
            len_batch = len_batch // args.model_stride + \
                        torch.ne((len_batch%args.model_stride), 0).int()
            ret, enc_out = trans_decoder.decode_batch(data_batch, len_batch,
                                                      len_batch + 100)
            hyps = ret["predictions"]
            scores = ret["scores"]
            for i in range(args.batch_size):
                for j in range(args.n_best):
                #for j in range(1):
                    nonblk_hyp = [e.item() for e in hyps[i][j] if e != args.blk]
                    if args.las_rescorer is not None:
                        tgt = torch.LongTensor([args.SOS] + nonblk_hyp \
                                               + [args.EOS])
                        #las_in:, T, B, C
                        las_in = enc_out[i].unsqueeze(1)
                        #tgt: T, B, C
                        tgt = tgt.cuda().unsqueeze(-1).unsqueeze(-1)
                        las_scores = trans_decoder.las_rescore(las_in, tgt)
                    if args.las_rescorer_bw is not None:
                        nonblk_hyp_bw = nonblk_hyp[::-1]
                        tgt = torch.LongTensor([args.SOS] + nonblk_hyp_bw \
                                               + [args.EOS])
                        #las_in:, T, B, C
                        las_in = enc_out[i].unsqueeze(1)
                        #tgt: T, B, C
                        tgt = tgt.cuda().unsqueeze(-1).unsqueeze(-1)
                        las_scores_bw = trans_decoder.las_rescore(las_in, tgt, bw=True)
                    if args.bilas_rescorer is not None:
                        tgt = torch.LongTensor([args.SOS] + nonblk_hyp \
                                               + [args.EOS])
                        #las_in:, T, B, C
                        las_in = enc_out[i].unsqueeze(1)
                        #tgt: T, B, C
                        tgt = tgt.cuda().unsqueeze(-1).unsqueeze(-1)
                        las_scores = trans_decoder.bilas_rescore(las_in, tgt)
                    f.write("".join([sym_map[e] for e in nonblk_hyp]))
                    if args.output_scores:
                        f.write(" {}".format(scores[i][j]))
                        if args.las_rescorer is not None:
                            f.write(' ')
                            f.write(' '.join([str(s) for s in las_scores]))
                        if args.las_rescorer_bw is not None:
                            f.write(' ')
                            f.write(' '.join([str(s) for s in las_scores_bw]))
                        if args.bilas_rescorer is not None:
                            f.write(' ')
                            f.write(' '.join([str(s) for s in las_scores + las_scores]))
                    f.write("\n")
                    f.flush()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Pytorch ASR --- decoding script for LAS model')
    parser.add_argument('model', type=str, help='model loaded for decoding')
    parser.add_argument('input_specifier', type=str, help='rspec for input feats')

    parser.add_argument('input_labels', type=str, help='rspec for dummy input labels')
    parser.add_argument('output_file', type=str, help='file to write for output hypothese')

    parser.add_argument('--lm', type=str, help="lm filename", default='')
    parser.add_argument('--lm_scale', type=float, default=1.0,
                        help="LM scale used in decoding")
    parser.add_argument('--fst_lm', type=str, help="fst lm filename", default='')
    parser.add_argument('--fst_lm_scale', type=float, default=1.0,
                        help="LM scale used in decoding")
    parser.add_argument('--nonblk_reward', type=float, default=1.5,
                        help="nonblk reward used in LM rescoring")
    parser.add_argument('--global_lm', type=str, help="fst lm filename", default='')
    parser.add_argument('--global_lm_scale', type=float, default=1.0,
                        help="LM scale used in decoding")
    parser.add_argument('--las_rescorer_model', type=str, default=None,
                        help='LAS model used to rescore RNNT N-best')
    parser.add_argument('--las_rescorer_bw_model', type=str, default=None,
                        help='backward LAS model used to rescore RNNT N-best')
    parser.add_argument('--bilas_rescorer_model', type=str, default=None,
                        help='bidirectional LAS model used to rescore RNNT N-best')
    parser.add_argument('--SOS', type=int, default=-1,
                        help='start of seq id, valid when beyond 0')
    parser.add_argument('--EOS', type=int, default=-1,
                        help='end of seq id, valid when beyond 0')
    parser.add_argument('--sm_scale', type=float, default=1.0,
                        help="softmax scale used in decoding")
    parser.add_argument('--blk', type=int, default=0,
                        help='blank ID ')
    parser.add_argument('--output_scores', action='store_true', 
                        help='output scores with hypothesis')
    parser.add_argument('--cmn', action="store_true",
                        help="apply cepstrum mean normalizaiton per utterance")
    parser.add_argument('--cmvn_stats', type=str, default=None,
                        help='cmvn_stats file')
    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    parser.add_argument('--loader', choices=['bptt', 'frame', 'utt'], default='frame',
                        help='loaders for inferencing')
    parser.add_argument('--beam_size', type=int, default=64,
                        help='num of hyps for beam search')
    parser.add_argument('--n_best', type=int, default=1,
                        help='num of best hyps output after decoding finish')
    parser.add_argument('--max_sent_length', type=int, default=500,
                        help='max length limits on decoding output')
    parser.add_argument('--padding_idx', type=int, default=-1,
                        help='padding index for targets')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='process id when using multi-GPU')
    parser.add_argument('--symbols_map', type=str,
                        help="file mapping symbol to int")
    parser.add_argument('--disambig_ids', type=str, default='',
                        help='comma separated disambig ids for LM fst')
    parser.add_argument('--max_num_arcs', type=int, default=0,
                        help='maximum number of arcs of LM fst')
    parser.add_argument('--max_id', type=int, default=0,
                        help='maximum i/o label id of LM fst')
    parser.add_argument('--backoff_id', type=int, default=0,
                        help='backoff label id of LM fst')
    parser.add_argument('--min_len', type=int, default=0,
                        help="will pad input if less than this value")

    parser.add_argument('--model_lctx', type=int, default=0,
                        help='model left context')
    parser.add_argument('--model_rctx', type=int, default=0,
                        help='model right context')
    parser.add_argument('--model_stride', type=int, default=1,
                        help='model stride, ie., subsampling in the model')


    args, unk = parser.parse_known_args()

    loader_module = importlib.import_module('loader.'+args.loader+'_loader')
    loader_module.register(parser)
    args = parser.parse_args()

    args.input_dim = loader_module.get_inputdim(args)
    args.dataloader = loader_module.dataloader

    main()
