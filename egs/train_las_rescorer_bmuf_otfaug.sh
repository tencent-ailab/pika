#activate python vitual env
source /root/anaconda3/bin/activate
#experiment dir
exp_dir=/mnt/ceph_asr_ts2/cweng/egs
. $exp_dir/path.sh
export OMP_NUM_THREADS=3
set -e
export PYTHONPATH=$PYTHONPATH:$PIKA_ROOT

#available cuda devices and number of nodes
cuda_devices="0,1,2,3,4,5,6,7"
world_size=`echo $cuda_devices | sed -e 's/,/\n/g' | wc -l`
batch_size=8
nnodes=1
proto=las
rnn_type=LSTM
rnn_size=1024
enc_type=rnn
dec_type=rnn
enc_layers=2
dec_layers=2
output_dim=6268
padding_idx=$output_dim
sample_rate=16000
#max and min length in seconds 
max_len=12
min_len=0.5
#padding_idx=0

node_id=0
task_flag=pika_baseline_lasrescorer
mkdir -p $exp_dir/logs.$task_flag
mkdir -p $exp_dir/output/${task_flag}.${node_id}
case 0 in
1)
;;
esac

#NOTE: add "--reverse_labels" option to train a 
#      backward las rescorer

#if shared encoder is used, 
#input_dim will be output dimension of the shared encoder

#--nnodes=2 --node_rank=$node_id --master_addr="${CHIEF_IP}" --master_port=$MASTER_PORT \
MASTER_PORT=36123
CUDA_VISIBLE_DEVICES=$cuda_devices NCCL_DEBUG=TRACE NCCL_IB_DISABLE=1 python -m torch.distributed.launch \
  --nproc_per_node $world_size $PIKA_ROOT/trainer/train_las_bmuf_otfaug.py \
  --verbose \
  --optim sgd \
  --initial_lr 0.003 \
  --final_lr 0.0003 \
  --enc_loss_scale 0.0 \
  --dec_loss_scale 1.0 \
  --grad_clip 3.0 \
  --num_batches_per_epoch 526264 \
  --shared_encoder_model $exp_dir/final.model \
  --num_epochs 5 \
  --momentum 0.9 \
  --block_momentum 0.9 \
  --sync_period 5 \
  --feats_dim 80 \
  --input_dim 1024 \
  --cuda --lr 0.001 --batch_size 8 \
  --encoder_type $enc_type \
  --enc_layers $enc_layers \
  --decoder_type $dec_type \
  --dec_layers $dec_layers \
  --rnn_type LSTM \
  --rnn_size $rnn_size \
  --brnn --embd_dim 100 \
  --SOS 0 --EOS 6268 \
  --dropout 0.2 \
  --padding_idx 6269 \
  --padding_tgt 6269 \
  --global_attention mlp \
  --stride 1 \
  --queue_size 8 \
  --loader otf_utt \
  --batch_first \
  --cmn \
  --cmvn_stats $exp_dir/global_cmvn.stats \
  --output_dim 6269 \
  --num_workers 1 \
  --sample_rate 16000 \
  --feat_config $exp_dir/fbank.conf \
  --TU_limit 15000 \
  --gain_range 50,10 \
  --speed_rate 0.9,1.0,1.1 \
  --log_per_n_frames 131072  \
  --max_len 1600 \
  --lctx 1 --rctx 1\
  --encoder_lctx 21 --encoder_rctx 21 \
  --encoder_stride 4 \
  $proto $exp_dir/lst/data.${node_id}.WORKER-ID.lst \
  $exp_dir/logs.$task_flag/train_las.${node_id}.WORKER-ID.log \
  $exp_dir/output/${task_flag}.${node_id} > $exp_dir/logs.$task_flag/main.${node_id}.log


exit 0


