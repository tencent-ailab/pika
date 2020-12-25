#activate python vitual env
source /data/home/cweng/build/anaconda3/bin/activate
#experiment dir
exp_dir=/data/home/cweng/build/pika/egs
. $exp_dir/path.sh
export PYTHONPATH=$PYTHONPATH:$PIKA_ROOT:$PIKA_ROOT/trainer
set -e

##########configs#############
#rnn transducer model
rnnt_model=model.epoch.3.3
#forward and backward las rescorer model
lasrescorer_fw=
lasrescorer_bw=
#cepstral mean normalization
cmn=false
#batch size
batch_size=2
#cuda device
export CUDA_VISIBLE_DEVICES=3
dir=$exp_dir/eval
mkdir -p $dir
##############################

case 0 in
1)

;;
esac


for data_dir in data/spon_test_fbank; do
  #check evaluation set files
  #for feats.scp file you will need to call standard 
  #kaldi feature extraction pipeline, make sure to use the 
  #same feature config file fbank.conf as in the training 
  for f in feats.scp text; do
    [ ! -f $data_dir/$f ] && echo "missing eval data file $f" && exit 1;
  done
  #group utterances with similar length for batch decoding
  if [ ! -f $data_dir/feats.len.ark ]; then 
    $KALDI_ROOT/src/featbin/feat-to-len scp:$data_dir/feats.scp ark,t:$data_dir/feats.len.ark
  fi
  x=`basename $data_dir`
  if [ ! -f $dir/shuffled.len.${x}.ark ]; then 
    python $PIKA_ROOT/utils/shuffle_by_length.py \
        --batch_size $batch_size --max_len 1000000\
        $data_dir/feats.len.ark $dir/shuffled.len.${x}.ark
  fi
  awk '{ if(r==0) { utt_id=$1; feats[$1]=$0; }
         if(r==1) { if(feats[$1] != "") { print feats[$1]; } } 
  }' r=0 $data_dir/feats.scp  r=1  $dir/shuffled.len.${x}.ark > $dir/eval.${x}.scp

  x=`basename $data_dir`
  #char.txt is symbol table where each target unit is mapped to an integer, eg.,
  #blk   0
  #<unk> 1
  #a     2
  #b     3
  #... 
  #This should match the one used in training. 
  output_dim=`cat $exp_dir/char.txt  | wc -l`
  padding_idx=$output_dim
  
  #zero is used to pad dummpy labels which are needed by the loader
  cat $dir/eval.${x}.scp | awk '{print $1, "0"}' > $dir/eval.${x}.label.ark
  eval_label="ark:$dir/eval.${x}.label.ark"
  eval_feats="scp:$dir/eval.${x}.scp"
  #SOS EOS for LAS rescoring,
  #SOS and blk share the same index, i.e., 0
  #whereas EOS uses the last id plus 1, i.e., output_dim. 
  for beam in 8 ; do
    if [ ! -f $dir/raw_hyp_${x}_${beam} ]; then
      python $PIKA_ROOT/decoder/decode_transducer.py  \
        --verbose \
        --cuda \
        --min_len 50 \
        --blk 0 \
        --batch_first \
        --beam_size $beam \
        --output_scores \
        --sm_scale 0.8\
        --batch_size $batch_size \
        --n_best $beam \
        --SOS 0 --EOS $output_dim \
        --padding_idx $padding_idx \
        --padding_tgt $padding_idx \
        --loader utt \
        --lctx 1 --rctx 1 \
        --model_lctx 21 --model_rctx 21 \
        --model_stride 4 \
        --stride 1 \
        --cmvn_stats $exp_dir/global_cmvn_80fbank.stats \
        --symbols_map $exp_dir/char.txt \
        --feats_dim 80 \
        ${las_rescorer_fw:+ --las_rescorer_model $las_rescorer_fw}  \
        ${las_rescorer_bw:+ --las_rescorer_bw_model $las_rescorer_bw} \
        $rnnt_model \
        "$eval_feats" \
        "$eval_label" \
        $dir/raw_hyp_${x}_${beam}
    fi
    mkdir -p $dir/beam${beam}
    scoredir=$dir/beam${beam}
    if [ ! -z "$las_rescorer_fw" ] && [ ! -z "$las_rescorer_bw" ]; then 
      python local/nbest_rerank.py --las_rescore --nbest $beam $dir/raw_hyp_${x}_${beam} $dir/raw.hyp 
    else 
      python local/nbest_rerank.py --nbest $beam $dir/raw_hyp_${x}_${beam} $dir/raw.hyp
    fi
    paste $dir/eval.${x}.label.ark $dir/raw.hyp | awk '{$2=" "; print $0}' | sed 's/<unk>//g' > $scoredir/hyp 
    cat $data_dir/text | perl -CSDA -ane '
        {
          print $F[0];
          foreach $s (@F[1..$#F]) {
            if (($s =~ /\[.*\]/) || ($s =~ /\<.*\>/) || ($s =~ "!SIL")) {
              print " $s";
            } else {
              @chars = split "", $s;
              foreach $c (@chars) {
                print " $c";
              }
            }
          }
          print "\n";
        }' > $scoredir/ref
    $KALDI_ROOT/src/bin/compute-wer --text --mode=present \
      ark:$scoredir/ref ark:$scoredir/hyp
  



done

done
