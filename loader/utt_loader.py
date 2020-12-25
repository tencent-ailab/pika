"""
Offline data (kaldi feature) loader
"""
import queue
from threading import Thread
import numpy as np
import torch
from torch.autograd import Variable
from kaldi.util.table import SequentialIntVectorReader, \
                             SequentialMatrixReader
from kaldi.matrix import _matrix_ext
from loader.otf_utt_loader import splice, put_thread, get_inputdim

###Batch utterance loader###

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
    parser.add_argument('--buffer_size', type=int, default=128*1024,
                        help='buffer size used to shuffle data')
    parser.add_argument('--ctc_target', action='store_true',
                        help='whether the reader is used for CTC training or not')
    parser.add_argument('--batch_first', action='store_true',
                        help='whether 1st dim of tensor if batch or frame')
    parser.add_argument('--stride', type=int, default=1,
                        help='strides for subsampling input (after splicing)')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='batch size')
    parser.add_argument('--queue_size', type=int, default=8,
                        help='queue size for threading')
    parser.add_argument('--padding_tgt', type=int, default=-1,
                        help='padding index for targets')
    parser.add_argument('--feats_dim', type=int, default=40,
                        help='dimension of input feature (before splicing)')
    parser.add_argument('--verbose', action='store_true',
                        help='printing out warnings')

def dataloader(align_rspec, feats_rspec, dummy_args, args):
    """
    Args:
        align_rspec: kaldi style read rspecifier for alignment
        feats_rspec: kaldi stule read rspecifier for feature
    """
    q = queue.Queue(args.queue_size)
    if args.ctc_target:
        thread = Thread(target=put_thread,
                        args=(q, ctc_utt_generator, align_rspec,
                              feats_rspec, False, args))
    else:
        thread = Thread(target=put_thread,
                        args=(q, utt_generator, align_rspec,
                              feats_rspec, False, args))
    thread.daemon = True
    thread.start()

    while True:
        item = q.get()
        q.task_done()
        if item is None:
            break
        yield item
    thread.join()

def ctc_utt_generator(align_rspec, feats_rspec, shuffle, args):
    """
    we do not really need 'target' generated
    in MMI/sMBR training from this generator
    so the interface is adjusted to fullfill
    warp_ctc for CTC training, target is now
    a tuple of (label, label_size).
    """
    ali_reader = SequentialIntVectorReader(align_rspec)
    feats_reader = SequentialMatrixReader(feats_rspec)
    max_len = args.max_len
    batch_size = args.batch_size

    data_buffer = np.zeros((batch_size, max_len, get_inputdim(args)),
                           dtype=np.float32)
    target_buffer = np.zeros((batch_size * max_len), dtype=np.int32)
    len_buffer = np.zeros(batch_size, dtype=np.int32)
    ali_len = np.zeros(batch_size, dtype=np.int32)
    start_flag = torch.IntTensor([1] * batch_size)

    if args.cuda:
        start_flag = start_flag.cuda(args.local_rank)

    batch_idx = 0
    target_len = 0
    batch_max_len = -1

    #!!!make sure feature and ali
    #!!!has exact the same  order
    for (uttid, ali), (uttid2, feats) in zip(ali_reader, feats_reader):
        assert uttid2 == uttid
        ali = np.array(ali)
        feats = _matrix_ext.matrix_to_numpy(feats)
        #in CTC training, the ali is shorter
        utt_len = feats.shape[0] // args.stride + \
                  int(feats.shape[0] % args.stride != 0)
        assert ali.shape[0] <= utt_len

        ali_len[batch_idx] = ali.shape[0]
        data_buffer[batch_idx, :utt_len, :] = splice(feats, args.lctx,
                                                     args.rctx)[::args.stride]
        target_buffer[target_len: target_len + ali_len[batch_idx]] = ali
        target_len += ali_len[batch_idx]
        len_buffer[batch_idx] = utt_len

        if utt_len > batch_max_len:
            batch_max_len = utt_len

        batch_idx += 1

        if batch_idx == batch_size:
            for b in range(batch_size):
                utt_len = len_buffer[b]
                data_buffer[b, utt_len:batch_max_len, :] = 0
                #target_buffer[b, ali_len[b]:batch_max_len]  = -1

            data = data_buffer[:, :batch_max_len, :]
            target = target_buffer[:target_len]

            if not args.batch_first:
                data = np.transpose(data, (1, 0, 2))
                #target = np.transpose(target, (1, 0))

            data = np.copy(data)
            target = np.copy(target)
            lens = np.copy(len_buffer)
            ali_lens = np.copy(ali_len)

            data = torch.from_numpy(data)
            target = torch.from_numpy(target)

            if args.cuda:
                data, target = data.cuda(args.local_rank), target

            yield Variable(data), (Variable(target), ali_lens), lens, start_flag

            batch_idx = 0
            target_len = 0
            batch_max_len = -1

    yield None



def utt_generator(align_rspec, feats_rspec, shuffle, args):
    """
    Args:
        align_rspec: kaldi style read rspecifier for alignment
        feats_rspec: kaldi stule read rspecifier for feature
        shuffle: deprecated
        args: arguments
    """
    ali_reader = SequentialIntVectorReader(align_rspec)
    feats_reader = SequentialMatrixReader(feats_rspec)
    max_len = args.max_len
    batch_size = args.batch_size
    data_buffer = np.zeros((batch_size, max_len, get_inputdim(args)),
                           dtype=np.float32)
    target_buffer = np.zeros((batch_size, max_len), dtype=np.int32)
    len_buffer = np.zeros(batch_size, dtype=np.int32)
    ali_len = np.zeros(batch_size, dtype=np.int32)
    start_flag = torch.IntTensor([1] * batch_size)

    if args.cuda:
        start_flag = start_flag.cuda(args.local_rank)

    batch_idx = 0
    target_len = 0
    batch_max_len = -1
    target_max_len = -1
    for (uttid, ali), (uttid2, feats) in zip(ali_reader, feats_reader):
        assert uttid2 == uttid
        ali = np.array(ali)
        feats = _matrix_ext.matrix_to_numpy(feats)
        utt_len = feats.shape[0] // args.stride + int(feats.shape[0] % args.stride != 0)
        #ali/targets should be shorter
        #assert ali.shape[0] <= utt_len
        ali_len[batch_idx] = ali.shape[0]
        data_buffer[batch_idx, :utt_len, :] = \
            splice(feats, args.lctx, args.rctx)[::args.stride]
        target_buffer[batch_idx, :ali_len[batch_idx]] = ali
        #target_len += ali_len[batch_idx]
        len_buffer[batch_idx] = utt_len

        if utt_len > batch_max_len:
            batch_max_len = utt_len

        if ali_len[batch_idx] > target_max_len:
            target_max_len = ali_len[batch_idx]

        batch_idx += 1

        if batch_idx == batch_size:
            for b in range(batch_size):
                utt_len = len_buffer[b]
                target_len = ali_len[b]
                #data and target padding
                data_buffer[b, utt_len:batch_max_len, :] = \
                    data_buffer[b, utt_len-1, :]
                target_buffer[b, target_len:target_max_len] = args.padding_tgt

            data = data_buffer[:, :batch_max_len, :]
            target = target_buffer[:, :target_max_len]

            if not args.batch_first:
                data = np.transpose(data, (1, 0, 2))
                target = np.transpose(target, (1, 0))

            data = np.copy(data)
            target = np.copy(target)
            lens = np.copy(len_buffer)
            ali_lens = np.copy(ali_len)

            data = torch.from_numpy(data)
            target = torch.from_numpy(target).long()

            if args.cuda:
                data = data.cuda(args.local_rank)
                target = target.cuda(args.local_rank)
            yield Variable(data), Variable(target), lens, ali_lens

            batch_idx = 0
            target_len = 0
            batch_max_len = -1
            target_max_len = -1

    yield None
