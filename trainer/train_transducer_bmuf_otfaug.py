"""
Transducer Training script
"""
#system utils
import sys
import argparse
import importlib
import os
import os.path
import math

import numpy as np
#torch related
import torch
import torch.optim as optim
from torch._six import inf

#supposed to be after kaldi import
from utils.logger import Logger
from utils.spec_augment import SpecAugment

from kaldi.matrix import _matrix_ext, DoubleMatrix
from kaldi.util import io
#from warprnnt_pytorch import RNNTLoss
from warp_rnnt import RNNTLoss

from trainer.bmuf import BmufTrainer


MASTER_NODE = 0

def run_one_epoch(epoch, model, log_f,
                  args, bmuf_trainer, training):
    """
    Run one epoch of training

    Args:
        epoch (int): zero based epoch index
        model (torch.nn.module): model
        log_f (file): logging file
        args : arguments from outer
        bmuf_trainer : initialized bmuf_trainer
        training (bool): training or validation
    """
    log_f.write('===> Epoch {} <===\n'.format(epoch))
    total_num_batches = args.num_epochs * args.num_batches_per_epoch
    num_batches_processed = epoch * args.num_batches_per_epoch
    lr = args.initial_lr * math.exp(num_batches_processed * \
                                    math.log(args.final_lr /\
                                             args.initial_lr) /\
                                    total_num_batches)
    log_f.write('===Using Learning Rate {}===\n'.format(lr))
    optimizer = optim.SGD(model.parameters(), lr,
                          momentum=args.momentum,
                          nesterov=True)

    loss_logger = Logger(args.log, args.log_per_n_frames, ['Loss'])
    transducer_loss = RNNTLoss(blank=0, reduction='sum').apply
    #transducer_loss = RNNTLoss(blank=0, reduction='sum')
    #for spec augment
    spec_augmentor = None
    if args.spec_augment:
        spec_augmentor = SpecAugment(args.max_freq_span, args.max_time_span)
    #for batchnorm and dropout
    if training:
        model.train()
    else:
        model.eval()
        optimizer = None

    for num_done, (data_cpu, target_cpu, len_cpu, ali_lens_cpu) in \
        enumerate(args.dataloader(args.data_lst, args.rir,
                                  args.noise, args)):
        if training:
            optimizer.zero_grad()

        if data_cpu is not None:
            #forward
            len_batch = len_cpu.cuda(args.local_rank)
            len_batch = len_batch - args.model_lctx - args.model_rctx
            len_batch = len_batch // args.model_stride + \
                        torch.ne((len_batch%args.model_stride), 0).int()
            ali_lens = ali_lens_cpu.cuda(args.local_rank)
            data_batch = data_cpu.cuda(args.local_rank)
            target_batch = target_cpu.long().cuda(args.local_rank)
            if args.cmvn_stats:
                #cmn#
                if args.cmn: 
                    data_batch -= data_batch.mean(dim=1).unsqueeze(dim=1)
                data_batch += args.offset.unsqueeze(dim=0).unsqueeze(dim=0).float()
                data_batch *= args.scale.unsqueeze(dim=0).unsqueeze(dim=0).float()
            if spec_augmentor is not None:
                spec_augmentor.apply(data_batch)

            outputs = model.forward(data_batch, target_batch, len_batch, True)

            loss = transducer_loss(outputs, target_batch.int(),
                                   len_batch, ali_lens)
            loss = loss.sum()
        else: #empty batch
            loss = torch.FloatTensor([0.0])

        if training:
            if data_cpu is not None:
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   args.grad_clip,
                                                   norm_type=inf)
                optimizer.step()

            if num_done != 0 and num_done % args.sync_period == 0:
                if not bmuf_trainer.update_and_sync():
                    return float('nan')
                num_batches_processed = (epoch * args.num_batches_per_epoch \
                                         + num_done)
                lr = args.initial_lr * math.exp(num_batches_processed * \
                                                math.log(args.final_lr /\
                                                         args.initial_lr) /\
                                                total_num_batches)
                optimizer = optim.SGD(model.parameters(), lr,
                                      momentum=args.momentum,
                                      nesterov=True)
        #destruct all cpu torch tensor
        #manully to prevent leaks if they
        #are created in different process
        del data_cpu, target_cpu, len_cpu, ali_lens_cpu

        labels = ali_lens.sum().item()
        loss_logger.update_and_log(labels, [loss.item()])

    if training:
        if not bmuf_trainer.update_and_sync():
            return float('nan')
        tot_loss, tot_num = loss_logger.summarize_and_log()
    else:
        tot_loss, tot_num = loss_logger.summarize_and_log()

    #aggregate across workers
    loss_tensor = torch.FloatTensor([tot_loss, float(tot_num)])
    loss_tensor = loss_tensor.cuda(args.local_rank)
    bmuf_trainer.sum_reduce(loss_tensor)
    bmuf_trainer.broadcast(loss_tensor)
    reduced_loss = loss_tensor[0] / loss_tensor[1]
    return reduced_loss.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transducer training')

    parser.add_argument('nnet_proto', type=str,
                        help='pytorch NN proto definition filename')
    parser.add_argument('data_lst', type=str,
                        help='list of mrk, seq, ali files for data')
    parser.add_argument('log', type=str,
                        help='log file for the job')
    parser.add_argument('output_dir', type=str,
                        help='path to save the final model')
    parser.add_argument('--init_model', type=str, default=None,
                        help='initial model')

    #rir and noise
    parser.add_argument('--rir_lst', type=str, default=None,
                        help='mrk and seq files for rir')
    parser.add_argument('--noise_lst', type=str, default=None,
                        help='mrk and seq files for noise')

    #Model Options
    parser.add_argument('--encoder_type', type=str, default='rnn',
                        choices=['rnn', 'transformer'],
                        help="""Type of encoder layer to use.""")
    parser.add_argument('--decoder_type', type=str, default='rnn',
                        choices=['rnn', 'transformer'],
                        help='Type of decoder layer to use.')

    parser.add_argument('--layers', type=int, default=-1,
                        help='Number of layers in enc/dec.')
    parser.add_argument('--enc_layers', type=int, default=2,
                        help='Number of layers in the encoder')
    parser.add_argument('--dec_layers', type=int, default=2,
                        help='Number of layers in the decoder')

    parser.add_argument('--rnn_size', type=int, default=512,
                        help='Size of LSTM hidden states')

    parser.add_argument('--rnn_type', type=str, default='LSTM',
                        choices=['LSTM'],
                        help="""rnn type""")

    parser.add_argument('--embd_dim', type=int, default=300,
                        help='embeddings dimension for decoder')
    parser.add_argument('--output_dim', type=int, default=8000,
                        help='output dimension of neural network')

    parser.add_argument('--model_lctx', type=int, default=0,
                        help='model left context')
    parser.add_argument('--model_rctx', type=int, default=0,
                        help='model right context')
    parser.add_argument('--model_stride', type=int, default=1,
                        help='model stride, ie., subsampling in the model')

    parser.add_argument('--brnn', action="store_true",
                        help="valid only when encoder type is rnn")
    parser.add_argument('--cmn', action="store_true",
                        help="apply cepstrum mean normalizaiton per utterance")
    parser.add_argument('--cmvn_stats', type=str, default=None,
                        help='cmvn_stats file')

    parser.add_argument('--optim', type=str, default='sgd',
                        choices=['sgd', 'adam', 'adadelta'],
                        help="""optimizer to use """)

    parser.add_argument('--grad_clip', type=float, default=-1.0,
                        help='gradient clipping threshold')

    parser.add_argument('--initial_lr', type=float, default=1.0,
                        help='initial learning rate')
    parser.add_argument('--final_lr', type=float, default=1.0,
                        help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum for SGD')
    parser.add_argument('--num_epochs', type=int, default=15,
                        help='number of epochs for training')
    parser.add_argument('--num_batches_per_epoch', type=int, default=1000,
                        help='number of batches per work per epoch')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help="Dropout probability; applied in LSTM stacks.")
    parser.add_argument('--padding_idx', type=int, default=-1,
                        help='padding index for targets')
    parser.add_argument('--loader', choices=['otf_utt'],
                        default='otf_utt',
                        help='loaders')
    parser.add_argument('--log_per_n_frames', type=int, default=1024*1024,
                        help='logging per n frames')
    parser.add_argument('--seed', type=int, default=777,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')

    #BMUF related
    parser.add_argument('--local_rank', type=int,
                        help='local process ID for parallel training')
    parser.add_argument('--block_momentum', type=float, default=0.9,
                        help='block momentum for BMUF')
    parser.add_argument('--block_lr', type=float, default=1.0)
    parser.add_argument('--sync_period', type=int, default=100)
    #Spec Augment related
    parser.add_argument('--spec_augment', action='store_true',
                        help='enable spec augmentation')
    parser.add_argument('--max_freq_span', type=int, default=15,
                        help='max frequency span to dropout input')
    parser.add_argument('--max_time_span', type=int, default=35,
                        help='max time span to dropout input')

    args, unk = parser.parse_known_args()

    # import loader
    loader_module = importlib.import_module('loader.' + args.loader \
                                             + '_loader')
    loader_module.register(parser)
    args = parser.parse_args()
    args.input_dim = loader_module.get_inputdim(args)
    args.dataloader = loader_module.dataloader

    # number of workers
    world_size = int(os.environ['WORLD_SIZE'])
    # set cuda device
    assert args.cuda
    assert torch.cuda.is_available()
    torch.cuda.set_device(args.local_rank)

    #this is just an example for those who
    #want to do on-the-fly noise and reverberation
    #both rir and noise are list of AudioSegments
    #see trainer/loader/audio.py for more details
    rir = []
    #noise adding augmentor has constrain
    #that noise lengths need to be longer
    #than utterance, assuming 100 fps
    #min_noise_len = args.max_len / 100 * args.sample_rate
    noise = []
    args.rir, args.noise = rir, noise

    #modify data and label rspec according to local_rank
    args.data_lst = args.data_lst.replace('WORKER-ID', str(args.local_rank))

    #modify log file according to local_rank
    args.log = args.log.replace('WORKER-ID', str(args.local_rank))
    log_f = open(args.log, 'w')
    args.log = log_f

    #manual seed for reproducibility
    nnet_module = importlib.import_module("model."+args.nnet_proto)
    frame_ctx = args.lctx + 1 + args.rctx
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    if args.init_model is None:
        model = nnet_module.Net(args, args.input_dim,
                                args.output_dim)

    else:
        model = torch.load(args.init_model,
                           map_location=lambda storage, loc: storage)

    model.cuda(args.local_rank)

    #initialize BMUF trainer
    bmuf_trainer = BmufTrainer(MASTER_NODE, args.local_rank, world_size,
                               model, args.block_momentum, args.block_lr)


    num_param = 0
    for param in model.parameters():
        num_param += param.numel()
    #print model proto
    log_f.write('*'*60+'\n')
    log_f.write('*'*60+'\n')
    log_f.write('*'*60+'\n')
    log_f.write('model proto: {}\n'
                'input  dim: {},\toutput dim: {},\n'
                'hidden dim: {},\tnum of enc_layers: {}\n'
                'num of dec_layers: {},\trnn_type: {}\n'
                'model size: {} M\n'
                .format(args.nnet_proto, args.input_dim,
                        args.output_dim, args.rnn_size,
                        args.enc_layers, args.dec_layers,
                        args.rnn_type, num_param/1000/1000))
    log_f.write('*'*60+'\n')
    log_f.write('*'*60+'\n')
    log_f.write('*'*60+'\n')
    log_f.flush()

    if torch.cuda.is_available():
        if args.cuda:
            torch.cuda.manual_seed(args.seed)
            model.cuda()
        else:
            print('WARNING: You have a CUDA device, '
                  'so you should probably run with --cuda')
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

    #start training with run_one_epoch, params:
    #(model, args, bmuf_trainer, train or valid)
    for epoch in range(0, args.num_epochs):
        train_loss = run_one_epoch(epoch, model, log_f,
                                   args, bmuf_trainer, True)
        #save current model
        current_model = '{}/model.epoch.{}.{}'.format(args.output_dir,
                                                      epoch, args.local_rank)
        with open(current_model, 'wb') as f:
            torch.save(model, f)

    log_f.write('Training Finished')
