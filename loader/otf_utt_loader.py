"""
This module implements on-the-fly data augmentation loader

"""
from random import randint
import queue
from threading import Thread
from scipy.stats import truncnorm

import numpy as np


import torch
#import torch.multiprocessing as mp
#mp.set_sharing_strategy('file_system')
#import multiprocessing as mp
from kaldi.util.table import SequentialIntVectorReader

from kaldi.matrix import _matrix_ext, Vector
from kaldi.util.options import ParseOptions
from kaldi.feat.fbank import Fbank, FbankOptions
from kaldi.feat.mfcc import Mfcc, MfccOptions

#audio augmentation utlities
from loader.audio import AudioSegment


def splice(feats, lctx, rctx):
    """
    feature splicing function
    Args:
        feats (numpy array): input features
        lctx (int): left context
        rctx (int): right context
    """
    length = feats.shape[0]
    dim = feats.shape[1]
    padding = np.zeros((length + lctx + rctx, dim), dtype=np.float32)
    padding[:lctx] = feats[0]
    padding[lctx:lctx+length] = feats
    padding[lctx+length:] = feats[-1]

    spliced = np.zeros((length, dim * (lctx + 1 + rctx)), dtype=np.float32)
    for i in range(lctx + 1 + rctx):
        spliced[:, i*dim:(i+1)*dim] = padding[i:i+length, :]
    return spliced

def put_thread(queue, generator, *gen_args):
    """
    thread filling queue with generated items
    Args:
        queue: multithread queue (this could be modified to be multi-process)
        generator: data generator
        gen_args: arguments for generator
    """
    for item in generator(*gen_args):
        queue.put(item)
        if item is None:
            break

def get_inputdim(args):
    """
    calculate full input dimension
    """
    return args.feats_dim * (args.lctx + 1 + args.rctx)


def register(parser):
    """
    register loader arguments
    """
    parser.add_argument('--lctx', type=int, default=10,
                        help='left context for splice')
    parser.add_argument('--rctx', type=int, default=10,
                        help='right context for splice')
    parser.add_argument('--max_len', type=int, default=6000,
                        help='max length allowed to be loaded')
    parser.add_argument('--num_workers', type=int, default=5,
                        help='number of workers to load/process')
    parser.add_argument('--sample_rate', type=int, default=16000,
                        help='sample rate of waves')
    parser.add_argument('--buffer_size', type=int, default=128*1024,
                        help='buffer size used to shuffle data')
    parser.add_argument('--batch_first', action='store_true',
                        help='1st dim is batch or frame')
    parser.add_argument('--reverse_labels', action='store_true',
                        help='reverse labels for training, eg for LAS')
    parser.add_argument('--feat_config', type=str, default=None,
                        help='feature extraction config file')
    parser.add_argument('--stride', type=int, default=1,
                        help='strides for subsampling input')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='batch size')
    parser.add_argument('--SOS', type=int, default=-1,
                        help='start of seq id, valid when beyond 0')
    parser.add_argument('--EOS', type=int, default=-1,
                        help='end of seq id, valid when beyond 0')
    parser.add_argument('--queue_size', type=int, default=8,
                        help='queue size for threading')
    parser.add_argument('--TU_limit', type=int, default=15000,
                        help='limits on the product of T (utt length)'
                             ' and U (label length) to avoid GPU OOM')
    parser.add_argument('--padding_tgt', type=int, default=-1,
                        help='padding index for targets')
    parser.add_argument('--feats_dim', type=int, default=40,
                        help='dimension of input feature (before splicing)')
    parser.add_argument('--snr_range', type=str, default='',
                        help='comma separated SNR range in dB')
    parser.add_argument('--gain_range', type=str, default='55,10',
                        help='comma separated negative gain range in dB')
    parser.add_argument('--speed_rate', type=str, default='0.9,1.0,1.1',
                        help='comma separated rate for speed perturbation')
    parser.add_argument('--verbose', action='store_true',
                        help='printing out warnings')

def dataloader(data_lst, rir, noise, args):
    """
    Args:
        data_lst: list of mrk and seq of input audios, and label ark
        rir: list of rir, List[AudioSegment]
        noise: list of noise, List[AudioSegment]
    """
    #load mrk and seq pairs
    data_triplets = []
    with open(data_lst, 'r', encoding='utf-8') as data_lst_f:
        for line in data_lst_f:
            data_triplets.append((line.split()[0],
                                  line.split()[1],
                                  line.split()[2]))
    num_per_worker = (len(data_triplets) + \
                       args.num_workers-1)//args.num_workers
    data_triplets_lst = []
    for i in range(0, len(data_triplets), num_per_worker):
        data_triplets_lst.append(data_triplets[i:i+num_per_worker])
    assert len(data_triplets_lst) == args.num_workers
    #multi-process version:
    #q = mp.Manager().Queue()
    #q = mp.SimpleQueue()
    q = queue.Queue(args.queue_size)
    #multi-process version:
    #threads = [mp.Process(target = put_thread,
    #              args = (q, otf_utt_generator, data_triplets_lst[i],
    #              rir, noise, args))
    #            for i in range(args.num_workers)]
    threads = [Thread(target=put_thread,
                      args=(q, otf_utt_generator, data_triplets_lst[i],
                            rir, noise, args))
               for i in range(args.num_workers)]

    for thread in threads:
        thread.daemon = True
        thread.start()
    num_done = 0
    while True:
        item = q.get()
        if item is None:
            num_done += 1
            if num_done == args.num_workers:
                break
            continue
        yield item
    for thread in threads:
        thread.join()

def otf_utt_generator(data_triplets, rir, noise, args):
    """
    Args:
        data_lst: list of mrk and seq of input audios, and label ark
        rir: list of rir, List[AudioSegment]
        noise: list of noise, List[AudioSegment]
        args: argumnets for loader
    """
    max_len = args.max_len
    batch_size = args.batch_size
    data_buffer = np.zeros((batch_size, max_len, get_inputdim(args)),
                           dtype=np.float32)
    target_buffer = np.zeros((batch_size, max_len), dtype=np.int32)
    len_buffer = np.zeros(batch_size, dtype=np.int32)
    ali_len = np.zeros(batch_size, dtype=np.int32)

    batch_idx = 0
    valid_idx = 0
    target_len = 0
    batch_max_len = -1
    target_max_len = -1

    #rates for speed perturbation
    speed_rate = [float(rate) for rate in args.speed_rate.split(',')]
    #volume level perturbation
    gain_lo, gain_hi = [-float(gain) for gain in args.gain_range.split(',')]
    #snr range for noise perturbation: 0-20db with mean of 10
    #mu, sigma = 10, 10
    #lo, hi = (0 - mu) / sigma, (20 - mu) / sigma
    #Fbank config
    po = ParseOptions('')
    fbank_opt = FbankOptions()
    fbank_opt.register(po)
    #fbank_opt = MfccOptions()
    #fbank_opt.register(po)
    po.read_config_file(args.feat_config)
    fbank = Fbank(fbank_opt)
    #fbank = Mfcc(fbank_opt)

    for data_triplet in data_triplets:
        mrk_fn, seq_fn = data_triplet[0], data_triplet[1]
        ali_rspec = data_triplet[2]
        with open(mrk_fn, 'r', encoding='utf-8') as mrk,\
             open(seq_fn, 'rb') as seq:
            ali_reader = SequentialIntVectorReader(ali_rspec)
            for line, (uttid1, ali) in zip(mrk, ali_reader):
                uttid = line.split()[0]
                assert uttid == uttid1
                seq.seek(int(line.split()[1]))
                num_bytes = int(line.split()[2])
                num_bytes -= num_bytes%2
                audio_bytes = seq.read(num_bytes)
                audio_np = np.frombuffer(audio_bytes, dtype='int16')
                #data augmentation function goes here
                audio_seg = AudioSegment(audio_np, args.sample_rate)
                #speed perturbation
                spr = speed_rate[randint(0, len(speed_rate)-1)]
                audio_seg.change_speed(spr)
                audio_seg.normalize(np.random.uniform(gain_lo, gain_hi))
                #noise adding example:
                #snr = truncnorm.rvs(lo, hi, scale=sigma, loc=mu, size=1)
                #audio_seg.add_noise(noise[randint(0, len(noise)-1)], snr)
                #rir adding example:
                #audio_seg.convolve_and_normalize(rir[randint(0, len(rir)-1)])
                audio_np = audio_seg._convert_samples_from_float32(\
                                     audio_seg.samples, 'int16')
                wave_1ch = Vector(audio_np)
                feats = fbank.compute_features(wave_1ch,
                                               args.sample_rate,
                                               vtnl_warp=1.0)
                ali = np.array(ali)
                if args.reverse_labels:
                    ali = ali[::-1]
                if args.SOS >= 0:
                    ali = np.concatenate(([args.SOS], ali))
                if args.EOS >= 0:
                    ali = np.concatenate((ali, [args.EOS]))
                feats = _matrix_ext.matrix_to_numpy(feats)
                utt_len = feats.shape[0] // args.stride + \
                          int(feats.shape[0] % args.stride != 0)
                #limits on T*U products due to RNNT.
                #this is pretty hacky now
                if ali.shape[0] * utt_len // 3 <= args.TU_limit:
                    ali_len[valid_idx] = ali.shape[0]
                    data_buffer[valid_idx, :utt_len, :] = \
                        splice(feats, args.lctx, args.rctx)[::args.stride]
                    target_buffer[valid_idx, :ali_len[valid_idx]] = ali
                    len_buffer[valid_idx] = utt_len
                    if utt_len > batch_max_len:
                        batch_max_len = utt_len
                    if ali_len[valid_idx] > target_max_len:
                        target_max_len = ali_len[valid_idx]
                    valid_idx += 1

                batch_idx += 1

                if batch_idx == batch_size:
                    for b in range(valid_idx):
                        utt_len = len_buffer[b]
                        target_len = ali_len[b]
                        #data and target padding
                        if utt_len > 0:
                            data_buffer[b, utt_len:batch_max_len, :] = \
                                data_buffer[b, utt_len-1, :]
                            target_buffer[b, target_len:target_max_len] = \
                                args.padding_tgt

                    data = data_buffer[:valid_idx, :batch_max_len, :]
                    target = target_buffer[:valid_idx, :target_max_len]

                    if not args.batch_first:
                        data = np.transpose(data, (1, 0, 2))
                        target = np.transpose(target, (1, 0))

                    data = torch.from_numpy(np.copy(data))
                    target = torch.from_numpy(np.copy(target))
                    lens = torch.from_numpy(np.copy(len_buffer[:valid_idx]))
                    ali_lens = torch.from_numpy(np.copy(ali_len[:valid_idx]))

                    if valid_idx > 0:
                    #not doing cuda() here, in main process instead
                        yield data, target, lens, ali_lens
                    else:
                        yield None, None, \
                              torch.IntTensor([0]), torch.IntTensor([0])

                    batch_idx = 0
                    valid_idx = 0
                    target_len = 0
                    batch_max_len = -1
                    target_max_len = -1

            ali_reader.close()

    yield None
