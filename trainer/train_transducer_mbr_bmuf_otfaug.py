"""
Transducer MBR Training Script
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
import torch.nn.functional as F
from torch.autograd import Variable
from torch._six import inf
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

#from warprnnt_pytorch import RNNTLoss
from warp_rnnt import RNNTLoss
import editdistance

#supposed to be after kaldi import
from utils.spec_augment import SpecAugment
from utils.logger import Logger

from kaldi.matrix import _matrix_ext, DoubleMatrix
from kaldi.util import io

from trainer.bmuf import BmufTrainer
from decoder.beam_transducer import GlobalScorer
from decoder.transducer_decoder import TransducerDecoder

MASTER_NODE = 0

###run one epoch of training###
def run_one_epoch(epoch, log_f, model, args, bmuf_trainer):
    """
    Run one epoch of MBR training

    Args:
        epoch (int): zero based epoch index
        log_f (file): logging file
        model (torch.nn.module): model
        args : arguments from outer
        bmuf_trainer : initialized bmuf_trainer
    """
    log_f.write('===> Epoch {} <===\n'.format(epoch))
    total_num_batches = args.num_epochs * args.num_batches_per_epoch
    num_batches_processed = (epoch * args.num_batches_per_epoch)
    lr = args.initial_lr * math.exp(num_batches_processed *\
                                    math.log(args.final_lr /\
                                             args.initial_lr) /\
                                    total_num_batches)
    log_f.write('===> Start Training with learning rate {} <===\n'.format(lr))
    optimizer = optim.SGD(model.parameters(), lr,
                          momentum=args.momentum,
                          nesterov=True)

    loss_logger = Logger(args.log, args.log_per_n_frames, ['MBR Loss', 'RNNT Loss'])
    transducer_loss = RNNTLoss(blank=0, reduction='sum').apply
    #for spec augment
    spec_augmentor = None
    if args.spec_augment:
        spec_augmentor = SpecAugment(args.max_freq_span, args.max_time_span)

    scorer = GlobalScorer()
    lm = None
    if args.lm != '':
        lm = torch.load(args.lm, map_location=lambda storage, loc: storage)
        lm.eval()
        if torch.cuda.is_available():
            lm.cuda()
    #initialize decoder for nbest generation
    args.las_rescorer, args.las_rescorer_bw, args.bilas_rescorer = None, None, None
    trans_decoder = TransducerDecoder(model, batch_size=args.batch_size,
                                      beam_size=args.beam_size,
                                      n_best=args.beam_size,
                                      blk=args.blk,
                                      global_scorer=scorer,
                                      lm=args.lm, lm_scale=args.lm_scale,
                                      sm_scale=args.sm_scale,
                                      cuda=args.cuda, 
                                      beam_prune=False, args=args)

    beam_size = args.beam_size
    #for batchnorm and dropout
    model.train()

    for num_done, (data_cpu, target_cpu, len_cpu, ali_lens_cpu) in \
        enumerate(args.dataloader(args.data_lst, args.rir,
                                  args.noise, args)):

        if data_cpu is not None:
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

            #nbest genereation
            model.eval()
            ret, _ = trans_decoder.decode_batch(data_batch, len_batch,
                                                len_batch + ali_lens + 3)
            hyps = ret["predictions"]
            scores = ret["scores"]

            #Here we do transducer forward inline to
            #save computation, note that we only need
            #one encoder forward for either reference
            #and nbest forward computation
            #Encoder forward,
            model.train()
            optimizer.zero_grad()
            #doing bsz instead of bsz * beamsize
            #x: batch, frame, dim
            if spec_augmentor is not None:
                spec_augmentor.apply(data_batch)
            x = data_batch
            x_len = len_batch
            if model.pack_seq and x_len is not None:
                packed_x = pack(x, x_len, batch_first=True,
                                enforce_sorted=True)
                packed_x, _ = model.encoder(packed_x)
                x = unpack(packed_x, batch_first=True)[0]
            else:
                x = model.encoder(x)
            #calculate RNN-T grads
            T = x.size()[1]
            yy = target_batch
            SOS = Variable(torch.zeros(yy.shape[0], 1).long())
            SOS = SOS.cuda(args.local_rank)
            yy = torch.cat((SOS, yy), dim=1)
            if model.decoder_type == 'rnn':
                yy = model.embed(yy)
                yy, _ = model.decoder(yy)
            else:
                yy = model.decoder(yy)
            xx = x.unsqueeze(2).expand(-1, -1, yy.size()[1], -1)
            yy = yy.unsqueeze(1).expand(-1, T, -1, -1)
            out_rnnt = torch.cat((xx, yy), dim=-1)
            out_rnnt = model.fc2(F.tanh(model.fc1(out_rnnt)) * \
                       F.sigmoid(model.fc_gate(out_rnnt)))
            #out_rnnt = F.log_softmax(args.sm_scale * out_rnnt, dim=-1)
            out_rnnt = F.log_softmax(out_rnnt, dim=-1)
            rnnt_loss = args.rnnt_scale * transducer_loss(out_rnnt,
                                                          target_batch.int(),
                                                          len_batch, ali_lens)
            rnnt_loss = rnnt_loss.sum()
            rnnt_loss.backward(retain_graph=True)

            #Expand encoder output x to bsz * beam_size
            bsz = x.size()[0]
            bb = bsz * beam_size
            x = x.unsqueeze(1).expand(-1, beam_size, -1, -1)\
                              .contiguous().view(bb, -1, model.hid_dim)

            #calculate MBR grads
            #I. prob of each sequence
            prob = torch.cuda.FloatTensor(scores).view(bsz, beam_size)
            prob = F.softmax(prob, dim=1)
            #II. edit distance
            dist = torch.cuda.FloatTensor(bsz, beam_size)
            max_len_nonblk = 0
            hyps_nonblk = []
            for i in range(bsz):
                hyps_nonblk.append([])
                for j in range(beam_size):
                    hyps_nonblk[i].append([])
                    hyps_nonblk[i][j] = [e.item() for e in hyps[i][j] if e.item() != args.blk]
                    max_len_nonblk = max(max_len_nonblk, len(hyps_nonblk[i][j]))
            U = max_len_nonblk + 1
            for i in range(bsz):
                ref = target_batch[i][:ali_lens[i]]
                reflist = ref.tolist()
                for j in range(beam_size):
                    dist[i][j] = editdistance.eval(reflist, hyps_nonblk[i][j])
                    hyps_nonblk[i][j].extend((max_len_nonblk \
                        - len(hyps_nonblk[i][j]))*[args.padding_idx])
            #III. avg_dist: bsz
            avg_dist = (prob * dist).sum(dim=1)
            mbr_loss = avg_dist.sum()
            avg_dist = avg_dist.unsqueeze(1).expand(bsz, beam_size)
            seq_grad = prob * (dist - avg_dist)

            #Decoder forward
            y = torch.cuda.LongTensor(hyps_nonblk).contiguous().view(bb, -1)
            SOS = Variable(torch.zeros(y.shape[0], 1).long())
            SOS = SOS.cuda(args.local_rank)
            y = torch.cat((SOS, y), dim=1)
            if model.decoder_type == 'rnn':
                y = model.embed(y)
                y, _ = model.decoder(y)
            else:
                y = model.decoder(y)
            assert U == y.size()[1]
            b_idx = []
            x_idx = []
            y_idx = []
            mbr_grad = x.data.new(bb, T+U, model.output_dim).zero_()
            for i in range(bsz):
                for j in range(beam_size):
                    t_idx = [0]
                    u_idx = [0]
                    for t in range(1, len(hyps[i][j])):
                        t_idx.append(t_idx[t-1] + int(hyps[i][j][t-1] == args.blk))
                        u_idx.append(u_idx[t-1] + int(hyps[i][j][t-1] != args.blk))
                    #padding
                    t_idx.extend((T+U-len(t_idx))*[0])
                    x_idx.extend(t_idx)
                    u_idx.extend((T+U-len(u_idx))*[0])
                    y_idx.extend(u_idx)
                    b_idx.extend([i*beam_size + j]*(T+U))
                    mbr_grad[i*beam_size+j,
                             torch.arange(len(hyps[i][j])),
                             hyps[i][j]] = seq_grad[i][j]
            #joint: [bb, T+U, dim]
            joint = torch.cat((x[b_idx, x_idx, :], y[b_idx, y_idx, :]), dim=-1)
            out = joint.contiguous().view(bb, T+U, -1)
            out = model.fc2(F.tanh(model.fc1(out)) * F.sigmoid(model.fc_gate(out)))
            out = F.log_softmax(args.sm_scale * out, dim=-1)
            #scale blk gradients by 1/T to stable training
            mbr_grad[:, :, args.blk] = mbr_grad[:, :, args.blk]/float(T)
            out.backward(mbr_grad)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               args.grad_clip,
                                               norm_type=inf)
            optimizer.step()
        else: #empty batch
            mbr_loss = torch.FloatTensor([0.0])
            rnnt_loss = torch.FloatTensor([0.0])

        if num_done != 0 and num_done % args.sync_period == 0:
            if num_done % 3000 == 0:
                tmp_model = '{}/model.{}.tmp'.format(args.output_dir,
                                                     args.local_rank)
                with open(tmp_model, 'wb') as tmp_f:
                    torch.save(model, tmp_f)
            if not bmuf_trainer.update_and_sync():
                return float('nan')
            num_batches_processed = (epoch * args.num_batches_per_epoch\
                                     + num_done)
            lr = args.initial_lr * math.exp(num_batches_processed *\
                                            math.log(args.final_lr /\
                                                     args.initial_lr) /\
                                            total_num_batches)
            optimizer = optim.SGD(model.parameters(), lr,
                                  momentum=args.momentum,
                                  nesterov=True)

        labels = ali_lens.sum().item()
        loss_logger.update_and_log(labels, [mbr_loss.item(), rnnt_loss.item()])

    if not bmuf_trainer.update_and_sync():
        return float('nan')
    tot_loss, tot_num = loss_logger.summarize_and_log()

    #aggregate across workers
    loss_tensor = torch.FloatTensor([tot_loss, float(tot_num)])
    loss_tensor = loss_tensor.cuda(args.local_rank)
    bmuf_trainer.sum_reduce(loss_tensor)
    bmuf_trainer.broadcast(loss_tensor)
    reduced_loss = loss_tensor[0] / loss_tensor[1]
    return reduced_loss.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transducer MBR training')

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

    #rir andnoise
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
                        help="""The gate type to use in the RNNs""")

    parser.add_argument('--beam_size', type=int, default=8,
                        help='beam size to generate nbest')
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
                        help='final learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum for SGD')
    parser.add_argument('--num_epochs', type=int, default=3,
                        help='max number of epochs for training')
    parser.add_argument('--num_batches_per_epoch', type=int, default=100000,
                        help='max number of epochs for training')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help="Dropout probability; applied in LSTM stacks.")
    parser.add_argument('--padding_idx', type=int, default=-1,
                        help='padding index for targets')
    parser.add_argument('--loader', choices=['utt', 'frame', 'otf_utt'],
                        default='frame', help='different loader ')
    parser.add_argument('--log_per_n_frames', type=int, default=1024*1024,
                        help='logging per n frames')
    parser.add_argument('--seed', type=int, default=777,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--rnnt_scale', type=float, default=0.01)
    parser.add_argument('--lm', type=str, help="lm filename", default='')
    parser.add_argument('--lm_scale', type=float, default=0.1,
                        help="LM scale used in decoding")
    parser.add_argument('--sm_scale', type=float, default=1.0,
                        help="softmax scale used in decoding")
    parser.add_argument('--blk', type=int, default=0,
                        help='blank ID ')

    #BMUF related
    parser.add_argument('--local_rank', type=int,
                        help='local process ID for parallel training')
    parser.add_argument('--block_momentum', type=float, default=0.9,
                        help='block momentum for BMUF')
    parser.add_argument('--block_lr', type=float, default=1.0)
    parser.add_argument('--sync_period', type=int, default=5)
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


    rir = []
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
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")

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
        train_loss = run_one_epoch(epoch, log_f, model, args, bmuf_trainer)
       #save current model
        current_model = '{}/model.epoch.{}.{}'.format(args.output_dir,
                                                      epoch, args.local_rank)
        with open(current_model, 'wb') as f:
            torch.save(model, f)

    log_f.write('Training Finished')
