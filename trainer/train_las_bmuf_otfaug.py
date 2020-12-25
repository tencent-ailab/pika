"""
LAS model/rescorer Training Script
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
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch._six import inf

#supposed to be after kaldi import
from utils.logger import Logger

from kaldi.matrix import _matrix_ext, DoubleMatrix
from kaldi.util import io

from trainer.bmuf import BmufTrainer

cudnn.benchmark = True

MASTER_NODE = 0

class LASLossCompute():
    """
    Utility Class used to calculate LAS model loss

    Args:
        dec_proj (nn.module): decoder projection layer
        enc_proj (nn.module): encoder projection layer
        dec_loss_scale (float): decoder loss scale
        enc_loss_scale (float): encoder loss scale
        padding_idx (int): padding index
    """

    def __init__(self, dec_proj, enc_proj,
                 dec_loss_scale=1.0, enc_loss_scale=0.0,
                 padding_idx=-1):
        super(LASLossCompute, self).__init__()
        self.padding_idx = padding_idx
        self.dec_loss_scale = dec_loss_scale
        self.enc_loss_scale = enc_loss_scale
        self.dec_proj = dec_proj
        self.enc_proj = enc_proj
        self.dec_criterion = nn.NLLLoss(ignore_index=padding_idx,
                                        size_average=False)
        self.enc_criterion = nn.CTCLoss()
        #the variables need to be backward
        #when self._shards() iterator ends
        self.variables = []

    def _compute_loss(self, output, target):
        output = F.log_softmax(self.dec_proj(self._bottle(output)))
        target = target.contiguous().view(-1)
        loss = self.dec_criterion(output, target)
        return loss

    def _compute_ctc_loss(self, output, lens, target):
        target = target.view(target.size(0), -1).transpose(0, 1)
        #prepare label and label_size for ctc loss
        #exclude  padding_idx, SOS = 0 and EOS = 1
        mask = torch.lt(target, self.padding_idx)
        mask = mask * torch.gt(target, 1)
        label = target[mask].cpu().int()
        mask = mask.cpu().int()
        label_size = torch.sum(mask.data, 1)
        enc_loss = self.enc_criterion(output, label, \
                                      Variable(torch.from_numpy(lens)), \
                                      Variable(label_size))
        return enc_loss

    def monolithic_compute_loss(self, output, enc_output,
                                enc_lens, target):
        """
        Compute the loss monolithically, not dividing into shards.
        """
        dec_loss = 0.0
        enc_loss = 0.0
        if self.dec_loss_scale > 0.0:
            #exclude SOS for target
            xent_loss = self.dec_loss_scale * self._compute_loss(output,
                                                                 target[1:])
            dec_loss += xent_loss.item()
        if self.enc_loss_scale > 0.0:
            l, b, _ = enc_output.size()
            enc_pout = self.enc_proj(self._bottle(enc_output)).view(l, b, -1)
            ctc_loss = self.enc_loss_scale * \
                       self._compute_ctc_loss(enc_pout, enc_lens, target)
            enc_loss += ctc_loss.item()
        return dec_loss, enc_loss

    def sharded_compute_loss(self, output, enc_output, enc_lens, target):
        """
        Compute the loss in shards for memory efficiency.
        """
        dec_loss = 0.0
        enc_loss = 0.0

        #compute sharded xent loss for decoder
        if self.dec_loss_scale > 0.0:
            #exclude SOS for target
            xent_loss = self.dec_loss_scale * self._compute_loss(output,
                                                                 target[1:])
            xent_loss.backward()
            dec_loss += xent_loss.item()
        #we can not really do sharded loss for the encoder
        #as CTC is a sequential loss, after the last shard
        #compute CTC loss monolithically, see self._shards
        if self.enc_loss_scale > 0.0:
            #perform encoder joint CTC training
            l, b, _ = enc_output.size()
            #enc_output detached from the graph
            enc_output_detached = Variable(enc_output.data, requires_grad=True,
                                           volatile=False)
            enc_pout = self.enc_proj(self._bottle(enc_output_detached)).view(l, b, -1)
            ctc_loss = self.enc_loss_scale * \
                       self._compute_ctc_loss(enc_pout,
                                              enc_lens,
                                              target)
            ctc_loss.backward()
            enc_loss += ctc_loss.item()
            self.variables.append((enc_output, enc_output_detached.grad.data))

        return dec_loss, enc_loss

    def _bottle(self, v):
        return v.view(-1, v.size(2))


###run one epoch of training/validation###
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

    dec_loss_logger = Logger(args.log, args.log_per_n_frames, ['DecLoss'])
    #set a large value to prevent progressive logging
    enc_loss_logger = Logger(args.log, 1e15, ['EncLoss'])
    las_loss = LASLossCompute(model.dec_proj,
                              model.enc_proj,
                              args.dec_loss_scale,
                              args.enc_loss_scale,
                              padding_idx=args.padding_idx)
    #for batchnorm and dropout
    if training:
        model.train()
        if args.shared_encoder is not None:
            args.shared_encoder.eval()
    else:
        model.eval()
        optimizer = None

    if args.sampling_decoder:
        if epoch >= args.increase_sampling_prob_epoch:
            args.sampling_prob = args.sampling_prob + 0.1
        if args.sampling_prob > 0.4:
            args.sampling_prob = 0.4
        model.decoder.set_sampling_prob(args.sampling_prob)
    #enable decoder when forward/backward
    # used when doing encoder pretraining
    enable_decoder = args.dec_loss_scale > 0.0
    #enable encoder when forward/backward
    # used when doing decoder pretraining
    enable_encoder = not args.pretrain_decoder

    for num_done, (data_cpu, target_cpu, len_cpu, _) in \
        enumerate(args.dataloader(args.data_lst, args.rir,
                                  args.noise, args)):

        if training:
            optimizer.zero_grad()

        if data_cpu is not None:
            data_batch = data_cpu.cuda(args.local_rank)
            target_batch = target_cpu.long().cuda(args.local_rank)
            if args.cmvn_stats:
                #cmn#
                data_batch -= data_batch.mean(dim=1).unsqueeze(dim=1)
                data_batch += args.offset.unsqueeze(dim=0).unsqueeze(dim=0)
                data_batch *= args.scale.unsqueeze(dim=0).unsqueeze(dim=0)
            if args.shared_encoder is not None:
                #shared encoder forward if any
                with torch.no_grad():
                    data_batch = args.shared_encoder(data_batch)
                #re-calculate lens due to shared encoder
                len_batch = len_cpu
                len_batch = len_batch - args.encoder_lctx - args.encoder_rctx
                len_batch = len_batch // args.encoder_stride + \
                            torch.ne((len_batch%args.encoder_stride), 0).int()
            #re-calculate lens due to model itself
            len_batch = len_batch - args.model_lctx - args.model_rctx
            len_batch = len_batch // args.model_stride + \
                        torch.ne((len_batch%args.model_stride), 0).int()

            #B, T, C -> T, B, C
            data_batch.transpose_(0, 1)
            target_batch.transpose_(0, 1)

            target_batch = torch.unsqueeze(target_batch, 2)
            dec_state = None

            # forward pass
            outputs, _, dec_state, enc_outputs = \
                model.forward(data_batch, target_batch,
                              len_batch,
                              dec_state, enable_decoder,
                              enable_encoder)
            if training:
                #note that backward is
                #done inside  function
                dec_loss, enc_loss = las_loss.sharded_compute_loss(outputs,
                                                                   enc_outputs,
                                                                   len_batch,
                                                                   target_batch)
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm(model.parameters(),
                                                  args.grad_clip,
                                                  norm_type=inf)
                optimizer.step()

        else: # empty batch
            dec_loss, enc_loss = 0.0, 0.0

        if training:
            if num_done != 0 and num_done % args.sync_period == 0:
                if not bmuf_trainer.update_and_sync():
                    return float('nan')
                num_batches_processed = (epoch * args.num_batches_per_epoch\
                                         + num_done)
                lr = args.initial_lr * math.exp(num_batches_processed * \
                                                math.log(args.final_lr /\
                                                         args.initial_lr) /\
                                                total_num_batches)
                optimizer = optim.SGD(model.parameters(), lr,
                                      momentum=args.momentum,
                                      nesterov=True)

        #exclude padding idx
        if data_cpu is not None:
            tokens = torch.numel(torch.nonzero(torch.lt(target_batch.data,
                                                        args.padding_idx)))
            frames = len_batch.sum().item()
            dec_loss_logger.update_and_log(tokens, [dec_loss])
            enc_loss_logger.update_and_log(frames, [enc_loss])

    if training:
        if not bmuf_trainer.update_and_sync():
            return float('nan')
        tot_loss, tot_num = dec_loss_logger.summarize_and_log()
        enc_loss_logger.summarize_and_log()
    else:
        tot_loss, tot_num = dec_loss_logger.summarize_and_log()

    #aggregate across workers
    loss_tensor = torch.FloatTensor([tot_loss, float(tot_num)])
    loss_tensor = loss_tensor.cuda(args.local_rank)
    bmuf_trainer.sum_reduce(loss_tensor)
    bmuf_trainer.broadcast(loss_tensor)
    reduced_loss = loss_tensor[0] / loss_tensor[1]
    return reduced_loss.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LAS training')

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
    parser.add_argument('--shared_encoder_model', type=str, default=None,
                        help='initial model')
    #Model Options
    parser.add_argument('--encoder_type', type=str, default='rnn',
                        choices=['rnn', 'brnn', 'mean', 'transformer', 'cnn'],
                        help="""Type of encoder layer to use.""")
    parser.add_argument('--decoder_type', type=str, default='rnn',
                        choices=['rnn', 'transformer', 'cnn'],
                        help='Type of decoder layer to use.')

    parser.add_argument('--layers', type=int, default=-1,
                        help='Number of layers in enc/dec.')
    parser.add_argument('--enc_layers', type=int, default=2,
                        help='Number of layers in the encoder')
    parser.add_argument('--dec_layers', type=int, default=2,
                        help='Number of layers in the decoder')

    parser.add_argument('--rnn_size', type=int, default=512,
                        help='Size of LSTM hidden states')
    parser.add_argument('--input_feed', type=int, default=1,
                        help="""Feed the context vector at each time step as
                        additional input (via concatenation with the word
                        embeddings) to the decoder.""")

    parser.add_argument('--input_feed_multihead', type=int, default=0,
                        help="""Feed the context vector at each time step as
                        additional input (via concatenation with the word
                        embeddings) to the decoder.""")

    parser.add_argument('--num_heads', type=int, default=0,
                        help=""" Number of heads for multihead attention""")

    parser.add_argument('--rnn_type', type=str, default='LSTM',
                        choices=['LSTM', 'GRU', 'SRU'],
                        help="""The gate type to use in the RNNs""")

    parser.add_argument('--downsampler_type', type=str, default='rnn',
                        choices=['rnn', 'cnn', 'dnn'],
                        help="""type of downsampler""")

    parser.add_argument('--use_downsampler', action='store_true',
                        help='enable downampler between encoder and decoder')
    parser.add_argument('--downsampler_layers', type=int, default=1,
                        help="""number of downsampler layers""")
    parser.add_argument('--downsampler_rate', type=int, default=2,
                        help="""downsampling rate""")
    parser.add_argument('--sampling_decoder', action='store_true',
                        help='enable sampling decoder')
    parser.add_argument('--sampling_prob', type=float, default=0.0,
                        help='sampling probability when sampling from'
                             ' previous decoder output')
    parser.add_argument('--embd_dim', type=int, default=300,
                        help='embeddings dimension for decoder')
    parser.add_argument('--input_dim', type=int, default=300,
                        help='input dimension of neural network')
    parser.add_argument('--output_dim', type=int, default=8000,
                        help='output dimension of neural network')

    parser.add_argument('--encoder_lctx', type=int, default=0,
                        help='shared encoder left context')
    parser.add_argument('--encoder_rctx', type=int, default=0,
                        help='shared encoder right context')
    parser.add_argument('--encoder_stride', type=int, default=1,
                        help='shared encoder stride, ie., '
                        'subsampling in the shared encoder')

    parser.add_argument('--model_lctx', type=int, default=0,
                        help='model left context')
    parser.add_argument('--model_rctx', type=int, default=0,
                        help='model right context')
    parser.add_argument('--model_stride', type=int, default=1,
                        help='model stride, ie., '
                        'subsampling in the model')

    parser.add_argument('--brnn', action="store_true",
                        help="Deprecated, use `encoder_type`.")
    parser.add_argument('--brnn_merge', default='concat',
                        choices=['concat', 'sum'],
                        help="Merge action for the bidir hidden states")

    parser.add_argument('--context_gate', type=str, default=None,
                        choices=['source', 'target', 'both'],
                        help="""Type of context gate to use.
                        Do not select for no context gate.""")

    parser.add_argument('--pretrain_decoder', action='store_true',
                        help='is current task decoder pretraing, ie LM training')
    parser.add_argument('--cmn', action="store_true",
                        help="apply cepstrum mean normalizaiton per utterance")
    parser.add_argument('--cmvn_stats', type=str, default=None,
                        help='cmvn_stats file')
    # Attention options
    parser.add_argument('--global_attention', type=str, default='mlp',
                        choices=['dot', 'general', 'mlp'],
                        help="""The attention type to use:
                        dotprot or general (Luong) or MLP (Bahdanau)""")

    # Genenerator and loss options.
    parser.add_argument('--copy_attn', action="store_true",
                        help='Train copy attention layer.')
    parser.add_argument('--copy_attn_force', action="store_true",
                        help='When available, train to copy.')
    parser.add_argument('--coverage_attn', action="store_true",
                        help='Train a coverage attention layer.')
    parser.add_argument('--lambda_coverage', type=float, default=1,
                        help='Lambda value for coverage.')


    parser.add_argument('--optim', type=str, default='sgd',
                        choices=['sgd', 'adam', 'adadelta'],
                        help="""optimizer to use """)

    parser.add_argument('--grad_clip', type=float, default=-1.0,
                        help='gradient clipping threshold, valid when greater than zero')

    parser.add_argument('--lr', type=float, default=1.0,
                        help='initial learning rate')
    parser.add_argument('--initial_lr', type=float, default=1.0,
                        help='initial learning rate')
    parser.add_argument('--final_lr', type=float, default=1.0,
                        help='final learning rate')
    parser.add_argument('--dec_loss_scale', type=float, default=1.0,
                        help="the scale for decoder loss when training")
    parser.add_argument('--enc_loss_scale', type=float, default=0.0,
                        help="the scale for encoder loss when training")
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum for SGD')
    parser.add_argument('--num_epochs', type=int, default=15,
                        help='max number of epochs for training')
    parser.add_argument('--num_batches_per_epoch', type=int, default=1000,
                        help='number of batches per work per epoch')
    parser.add_argument('--max_grad_norm', type=float, default=5,
                        help='If the norm of the gradient vector exceeds'
                        'this, renormalize it to have the norm equal to'
                        'max_grad_norm')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help="Dropout probability; applied in LSTM stacks.")
    parser.add_argument('--padding_idx', type=int, default=-1,
                        help='padding index for targets')
    parser.add_argument('--loader', choices=['utt', 'frame', 'otf_utt'],
                        default='frame', help='different loader for'
                        ' reading feature and labels')
    parser.add_argument('--log_per_n_frames', type=int, default=1024*1024,
                        help='logging per n frames')
    parser.add_argument('--seed', type=int, default=777,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--increase_sampling_prob_epoch', type=int, default=10000,
                        help='increase sampling probability after N epoch')
    parser.add_argument('--enable_ctc_before_epoch', action='store_true',
                        help='perform one epoch ctc training before each epoch')
    parser.add_argument('--anneal_factor', type=float, default=0.5)
    parser.add_argument('--start_anneal_impr', type=float, default=0.01)
    parser.add_argument('--stop_impr', type=float, default=0.001)

    #BMUF related
    parser.add_argument('--local_rank', type=int,
                        help='local process ID for parallel training')
    parser.add_argument('--block_momentum', type=float, default=0.9,
                        help='block momentum for BMUF')
    parser.add_argument('--block_lr', type=float, default=1.0)
    parser.add_argument('--sync_period', type=int, default=100)

    args, unk = parser.parse_known_args()

    # import loader
    loader_module = importlib.import_module('loader.' + args.loader \
                                             + '_loader')
    loader_module.register(parser)
    args = parser.parse_args()
    #args.input_dim = loader_module.get_inputdim(args)
    args.dataloader = loader_module.dataloader

    # number of workers
    world_size = int(os.environ['WORLD_SIZE'])
    # set cuda device
    assert args.cuda
    assert torch.cuda.is_available()
    torch.cuda.set_device(args.local_rank)

    args.rir, args.noise = [], []
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
                                args.output_dim, args.padding_idx)
    else:
        model = torch.load(args.init_model)
    model.cuda(args.local_rank)

    #load shared encoder if any
    args.shared_encoder = None
    if args.shared_encoder_model is not None:
        #shared_model is a module object having
        #encoder that is to be shared with LAS
        shared_model = torch.load(args.shared_encoder_model,
                                  map_location=lambda storage, loc: storage)
        args.shared_encoder = shared_model.encoder
        args.shared_encoder.cuda(args.local_rank)

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
