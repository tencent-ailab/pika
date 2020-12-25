import sys
import argparse

import numpy as np
from kaldi.util.table import SequentialWaveReader
from kaldi.matrix import Matrix, _matrix_ext

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='wav.scp to byte files, i.e.,'
                                                 'each line: uttid num_bytes')
    parser.add_argument('wav_rspecifier', type=str,
                        help='input wav.scp filename')
    parser.add_argument('byte_file', type=str,
                        help='input wav.scp filename')
    args, unk = parser.parse_known_args()
   
    wav_reader = SequentialWaveReader(args.wav_rspecifier)
    with open(args.byte_file, 'w') as bf:
        for uttid, wave in wav_reader:
            wave_data = _matrix_ext.matrix_to_numpy(wave.data())
            #has to be one channel 
            assert wave_data.shape[0] == 1
            bf.write('{} {}\n'.format(uttid, 2*len(wave_data[0].astype('int16'))))
