#activate python vitual env, optional  
source /root/anaconda3/bin/activate
#experiment dir
exp_dir=/mnt/ceph_asr_ts2/cweng/egs
. $exp_dir/path.sh
export OMP_NUM_THREADS=3
set -e
export PYTHONPATH=$PYTHONPATH:$PIKA_ROOT

#training data dir must contain wav.scp and label.txt files
#wav.scp: standard kaldi wav.scp file, see https://kaldi-asr.org/doc/data_prep.html 
#label.txt: label text file, the format is, uttid sequence-of-integer, where integer
#           is one-based indexing mapped label, note that zero is reserved for blank,  
#           ,eg., utt_id_1 3 5 7 10 23 
train_data_dir=



#data archive dir, for dumping data (eg., mounted disks)
ark_dir=$exp_dir/arks

#available cuda devices and number of nodes
cuda_devices="0,1,2,3,4,5,6,7"
world_size=`echo $cuda_devices | sed -e 's/,/\n/g' | wc -l`
nnodes=1

batch_size=8
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

#selecting training data with min and max length in seconds 
sample_rate=16000
min_len=0.5
max_len=12
max_bytes=`echo "($sample_rate * $max_len * 2) / 1" | bc`
min_bytes=`echo "($sample_rate * $min_len * 2) / 1" | bc`


mkdir -p $exp_dir $ark_dir $exp_dir/.tmp
tmpdir=$exp_dir/.tmp

case 0 in
1)
;;
esac

if [ ! -f $exp_dir/train.bytes ]; then
#get number of bytes for each wavs
nj=10
split_scps=""
for n in $(seq $nj); do
  split_scps="$split_scps $tmpdir/train.${n}.scp"
done
$PIKA_ROOT/utils/split_scp.pl $train_data_dir/wav.scp $split_scps || exit 1;
$PIKA_ROOT/utils/run.pl JOB=1:$nj $tmpdir/wav_to_bytes.JOB.log \
  python $PIKA_ROOT/utils//wav_to_bytes.py scp:$tmpdir/train.JOB.scp $tmpdir/train.JOB.bytes || exit 1;
for n in $(seq $nj); do
  cat $tmpdir/train.${n}.bytes || exit 1;
done > $exp_dir/train.bytes

rm -rf $tmpdir/*
fi


total_workers=`echo "$world_size * $nnodes" | bc`
python $PIKA_ROOT/utils/split_by_length.py \
    --batch_size $batch_size --max_len $max_bytes --min_len $min_bytes \
    --random --full_batch --world_size $total_workers \
    $exp_dir/train.bytes

last_id=`echo "$total_workers - 1" | bc`
for i in `seq 0 $last_id` ; do
  node_id=`echo "$i / $world_size" | bc`
  local_rank=`echo "$i % $world_size" | bc`
  # make wav.scp with same order as shuffled,
  awk '{ if(r==0) { utt_id=$1; wav[$1]=$0; }
         if(r==1) { if(wav[$1] != "") { print wav[$1]; } } 
  }' r=0 $train_data_dir/wav.scp r=1 $exp_dir/train.bytes.${i} > $exp_dir/train.${node_id}.${local_rank}.scp
  
  # make labels with same order as shuffled,
  awk '{ if(r==0) { utt_id=$1; label[$1]=$0; }
         if(r==1) { if(label[$1] != "") { print label[$1]; } } 
  }' r=0 $train_data_dir/label.txt r=1 $exp_dir/train.bytes.${i} > $exp_dir/train.label.${node_id}.${local_rank}.txt
done

# write all wav files to mrk,seq files
last_node=`echo "$nnodes - 1" | bc`
last_worker=`echo "$world_size - 1" | bc`
mkdir -p $ark_dir
for n in `seq 0 $last_node`; do
  for j in `seq 0 $last_worker`; do
    python $PIKA_ROOT/utils/wav_to_seq.py --num_wav_per_seq 2000 \
      scp:$exp_dir/train.${n}.${j}.scp \
      $ark_dir/train.${n}.${j}.mrk \
      $ark_dir/train.${n}.${j}.seq > $tmpdir/wav_to_bytes.train.${n}.${j}.log 2>&1 &  
  done
done
wait  



#split labels and generate lst
last_node=`echo "$nnodes - 1" | bc`
last_worker=`echo "$world_size - 1" | bc`
mkdir -p $ark_dir
mkdir -p $exp_dir/lst
for n in `seq 0 $last_node`; do
  for j in `seq 0 $last_worker`; do
    cat /dev/null > $exp_dir/lst/data.${n}.${j}.lst
    rm -rf $ark_dir/train.label.${n}.${j}.txt.*
    split -l 2000 $exp_dir/train.label.${n}.${j}.txt -d -a 3 $ark_dir/train.label.${n}.${j}.txt.
    f_list=`ls $ark_dir/train.label.${n}.${j}.txt.*`
    for k in $f_list; do
      suffix=`echo $k | awk -F '.' '{print $NF}'`
      idx=`echo $suffix | bc -l`
      if [ $suffix != $idx ] ; then
        mv $k $ark_dir/train.label.${n}.${j}.txt.$idx
      fi
      echo $ark_dir/train.${n}.${j}.mrk.$idx $ark_dir/train.${n}.${j}.seq.$idx ark:$ark_dir/train.label.${n}.${j}.txt.$idx >> $exp_dir/lst/data.${n}.${j}.lst
    done
  done
done



#calculate global cmvn stats

cat $exp_dir/lst/data.*.*.lst | $PIKA_ROOT/utils/shuffle_list.pl | head -n 50 > $exp_dir/lst/cmvn.lst

python $PIKA_ROOT/utils/compute_global_cmvn.py --sample_rate 16000 --feat_config fbank.conf \
                                               --cmn  --feat_dim 80  \
                                               $exp_dir/lst/cmvn.lst $exp_dir/global_cmvn.stats

#done with data prepartion




#you will need to change this configs according to your setup,
#num_batches_per_epoch: number of batches for each worker per epoch
#num_epochs: the fixed number of epochs, here we don't do early stop
MASTER_PORT=36123
node_id=0
task_flag=baseline
mkdir -p $exp_dir/logs.$task_flag $exp_dir/output/${task_flag}.${node_id}
CUDA_VISIBLE_DEVICES=$cuda_devices NCCL_DEBUG=TRACE python -m torch.distributed.launch \
  --nproc_per_node $world_size $PIKA_ROOT/trainer/train_transducer_bmuf_otfaug.py \
  --verbose \
  --optim sgd \
  --initial_lr 0.003 \
  --final_lr 0.0001 \
  --grad_clip 3.0 \
  --num_batches_per_epoch 526264\
  --num_epochs 8 \
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
  --embd_dim 100 \
  --dropout 0.2 \
  --brnn \
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
  --log_per_n_frames 131072  \
  --max_len 1600 \
  --lctx 1 --rctx 1\
  --model_lctx 21 --model_rctx 21 \
  --model_stride 4 \
  $proto $exp_dir/lst/data.${node_id}.WORKER-ID.lst \
  $exp_dir/logs.$task_flag/train_transducer.${node_id}.WORKER-ID.log \
  $exp_dir/output/${task_flag}.${node_id} > $exp_dir/logs.$task_flag/main.${node_id}.log

exit 0
