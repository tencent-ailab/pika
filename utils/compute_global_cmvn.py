import sys
import argparse

import numpy as np
from random import randint
from kaldi.transform.cmvn import Cmvn
from kaldi.util.options import ParseOptions
from kaldi.feat.fbank import Fbank, FbankOptions
from kaldi.matrix import Matrix, Vector, _matrix_ext

from loader.audio import AudioSegment

import numpy as np


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='global CMVN estimation')
    parser.add_argument('data_lst', type=str,
                        help='input data_lst filename')
    parser.add_argument('cmvn_stats', type=str,
                        help='output cmvn states filename')
    parser.add_argument('--cmn', action="store_true",
                        help="apply cepstrum mean normalizaiton per utterance")
    parser.add_argument('--sample_rate', type=int, default=16000,
                        help='sample rate of waves')
    parser.add_argument('--feat_config', type=str, default=None,
                        help='feature extraction config file')
    parser.add_argument('--feat_dim', type=int, default=80,
                        help='feature dimension')
    args, unk = parser.parse_known_args()

    po = ParseOptions('')
    fbank_opt = FbankOptions()
    fbank_opt.register(po)
    po.read_config_file(args.feat_config)
    fbank = Fbank(fbank_opt)
    speed_rate = [0.9, 1.0 , 1.1]
    cmvn = Cmvn(args.feat_dim)

    with open(args.data_lst, 'r', encoding='utf-8') as data_lst_f:
        for line in data_lst_f:
            mrk_fn = line.split()[0]
            seq_fn = line.split()[1]
            with open(mrk_fn, 'r', encoding='utf-8') as mrk, \
                 open(seq_fn, 'rb') as seq:
                for mrk_line in mrk:
                    seq.seek(int(mrk_line.split()[1]))
                    num_bytes = int(mrk_line.split()[2])
                    #this is making sure even number of bytes
                    num_bytes -= num_bytes%2
                    audio_bytes = seq.read(num_bytes)
                    audio_np = np.frombuffer(audio_bytes, dtype='int16')
                    audio_seg = AudioSegment(audio_np, args.sample_rate)
                    spr = speed_rate[randint(0, len(speed_rate)-1)]
                    audio_seg.change_speed(spr)
                    #-55 to -10 db
                    audio_seg.normalize(np.random.uniform(-55, -10))
                    audio_np = audio_seg._convert_samples_from_float32(\
                                         audio_seg.samples, 'int16')
                    wave_1ch = Vector(audio_np)
                    feats = fbank.compute_features(wave_1ch,
                                                   args.sample_rate,
                                                   vtnl_warp=1.0)
                    if args.cmn:
                        feats = _matrix_ext.matrix_to_numpy(feats)
                        feats -= np.mean(feats, axis=0)
                        feats = Matrix(feats) 

                    cmvn.accumulate(feats)

    cmvn.write_stats(args.cmvn_stats, binary=False)

