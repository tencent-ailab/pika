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
proto=transducer
rnn_type=LSTM
rnn_size=512
enc_type=transformer
dec_type=rnn
enc_layers=9
dec_layers=2

#vocab size plus 1 (blank)
output_dim=6268
padding_idx=$output_dim
sample_rate=16000

task_flag=pika_baseline_mbr
node_id=0
mkdir -p $exp_dir/logs.$task_flag $exp_dir/output/${task_flag}.${node_id}

case 0 in
1)
;;
esac

# --nnodes=2 --node_rank=$node_id --master_addr="${CHIEF_IP}" --master_port=$MASTER_PORT \
MASTER_PORT=36123
CUDA_VISIBLE_DEVICES=$cuda_devices NCCL_DEBUG=TRACE NCCL_IB_DISABLE=1 python -m torch.distributed.launch \
  --nproc_per_node $world_size $PIKA_ROOT/trainer/train_transducer_mbr_bmuf_otfaug.py \
  --verbose \
  --optim sgd \
  --init_model $exp_dir/init.model \
  --rnnt_scale 1.0 \
  --sm_scale 0.8 \
  --initial_lr 0.0001 \
  --final_lr 0.00001 \
  --grad_clip 3.0 \
  --num_batches_per_epoch 526264\
  --num_epochs 1 \
  --beam_size 4 \
  --momentum 0.9 \
  --block_momentum 0.9 \
  --sync_period 5 \
  --feats_dim 80 \
  --cuda --batch_size 8 \
  --encoder_type $enc_type \
  --enc_layers $enc_layers \
  --decoder_type $dec_type \
  --dec_layers $dec_layers \
  --rnn_type $rnn_type \
  --rnn_size 1024 \
  --brnn --embd_dim 100 \
  --dropout 0.2 \
  --padding_idx $padding_idx \
  --padding_tgt $padding_idx \
  --stride 1 \
  --queue_size 8 \
  --loader otf_utt \
  --batch_first \
  --cmn \
  --cmvn_stats $exp_dir/global_cmvn.stats \
  --output_dim $output_dim \
  --num_workers 1 \
  --sample_rate 16000 \
  --feat_config $exp_dir/fbank.conf \
  --TU_limit 15000 \
  --gain_range 50,10 \
  --speed_rate 0.9,1.0,1.1 \
  --log_per_n_frames 13107  \
  --max_len 1600 \
  --lctx 1 --rctx 1\
  --model_lctx 21 --model_rctx 21 \
  --model_stride 4 \
  $proto $exp_dir/lst/data.${node_id}.WORKER-ID.lst \
  $exp_dir/logs.$task_flag/train_mbr.${node_id}.WORKER-ID.log \
  $exp_dir/output/${task_flag}.${node_id} > $exp_dir/logs.$task_flag/main.${node_id}.log


exit 0


