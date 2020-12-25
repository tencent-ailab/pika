import sys
import argparse

import numpy as np
from kaldi.util.table import SequentialWaveReader
from kaldi.matrix import Matrix, _matrix_ext



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='wav.scp to seq and mrk file converter')
    parser.add_argument('--num_wav_per_seq', type=int, 
                        default=2000, help='number of wavs per seq/mrk')
    parser.add_argument('wav_rspecifier', type=str,
                        help='input wav.scp filename')
    parser.add_argument('out_mrk', type=str,
                        help='output mrk filename')
    parser.add_argument('out_seq', type=str,
                        help='output seq filename') 
    args, unk = parser.parse_known_args()
   
    wav_reader = SequentialWaveReader(args.wav_rspecifier)
    idx = 0
    num_written = 0
    offset = 0
    mrk, seq = None, None
    for uttid, wave in wav_reader:
        if num_written % args.num_wav_per_seq == 0: 
            offset = 0
            mrk = open(args.out_mrk + '.' + str(idx), 'w', encoding='utf-8')
            seq = open(args.out_seq + '.' + str(idx), 'wb')
            idx += 1
        wave_data = _matrix_ext.matrix_to_numpy(wave.data())
        assert wave_data.shape[0] == 1
        wave_data[0].astype('int16').tofile(seq)
        mrk.write('{} {} {}\n'.format(uttid, offset, 2*len(wave_data[0]))) 
        offset += 2 * len(wave_data[0]) 
        num_written+=1

        
        
