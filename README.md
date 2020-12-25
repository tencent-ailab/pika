--------------------------------------------------------------------------------

# PIKA: a lightweight speech processing toolkit based on Pytorch and (Py)Kaldi #
PIKA is a lightweight speech processing toolkit based on Pytorch and (Py)Kaldi. The first release focuses on end-to-end speech recognition. We use [Pytorch](https://pytorch.org) as deep learning engine, [Kaldi](https://github.com/kaldi-asr/kaldi) for data formatting and feature extraction.

## Key Features ##

- On-the-fly data augmentation and feature extraction loader

- TDNN Transformer encoder and convolution and transformer based decoder model structure

- RNNT training and batch decoding  

- RNNT decoding with external Ngram FSTs (on-the-fly rescoring, aka, shallow fusion)

- RNNT Minimum Bayes Risk (MBR) training 

- LAS forward and backward rescorer for RNNT

- Efficient BMUF (Block model update filtering) based distributed training  

## Installation and Dependencies ##

In general, we recommend [Anaconda](https://www.anaconda.com/) since it comes with most dependencies. Other major dependencies include, 

### Pytorch ###

Please go to <https://pytorch.org/> for pytorch installation, codes and scripts should be able to run against pytorch 0.4.0 and above. But we recommend 1.0.0 above for compatibility with RNNT loss module (see below) 

### Pykaldi and Kaldi ###

We use Kaldi (<https://github.com/kaldi-asr/kaldi)>) and PyKaldi (a python wrapper for Kaldi) for data processing, feature extraction and FST manipulations. Please go to Pykaldi website <https://github.com/pykaldi/pykaldi> for installation and make sure to build Pykaldi with ninja for efficiency. After following the installation process of pykaldi, you should have both Kaldi and Pykaldi dependencies ready.
  
### CUDA-Warp RNN-Transducer ###

For RNNT loss module, we adopt the pytorch binding at <https://github.com/1ytic/warp-rnnt> 

### Others ###

Check requirements.txt for other dependencies.

## Get Started ##

To get started, check all the training and decoding scripts located in egs directory.

### I. Data preparation and RNNT training ### 

egs/train_transducer_bmuf_otfaug.sh contains data preparation and RNNT training. One need to prepare training data and specify the training data directory,

```bash
#training data dir must contain wav.scp and label.txt files
#wav.scp: standard kaldi wav.scp file, see https://kaldi-asr.org/doc/data_prep.html 
#label.txt: label text file, the format is, uttid sequence-of-integer, where integer
#           is one-based indexing mapped label, note that zero is reserved for blank,  
#           ,eg., utt_id_1 3 5 7 10 23 
train_data_dir=
```

### II. Continue with MBR training ###

With RNNT trained model, one can continued MBR training with egs/train_transducer_mbr_bmuf_otfaug.sh (assuming using the same training data, therefore data preparation is omitted). Make sure to specify the initial model,

```bash
--verbose \
--optim sgd \
--init_model $exp_dir/init.model \
--rnnt_scale 1.0 \
--sm_scale 0.8 \
``` 

### III. Training LAS forward and backward rescorer ###

One can train a forward and backward LAS rescorer for your RNN-T model using egs/train_las_rescorer_bmuf_otfaug.sh. The LAS rescorer will share the encoder part with RNNT model, and has extra two-layer LSTM as additional encoder, make sure to specify the encoder sharing as,

```bash
--num_batches_per_epoch 526264 \
--shared_encoder_model $exp_dir/final.model \
--num_epochs 5 \
```

We support bi-directional LAS rescoring, i.e., forward and backward rescoring. Backward (right-to-left) rescoring is achieved by reversing sequential labels when conducting LAS model training. One can easily perform a backward LAS rescorer training by specifying,
```bash
--reverse_labels

```

### IV. Decoding  ###

egs/eval_transducer.sh is the main evluation script, which contains the decoding pipeline. Forward and backward LAS rescoring can be enabled by specifying these two models,

```bash
##########configs#############
#rnn transducer model
rnnt_model=
#forward and backward las rescorer model
lasrescorer_fw=
lasrescorer_bw=
```

## Caveats ##

All the training and decoding hyper-parameters are adopted based on large-scale (e.g., 60khrs) training and internal evaluation data. One might need to re-tune hyper-parameters to acheive optimal performances. Also the WER (CER) scoring script is based on a Mandarin task, we recommend those who work on different languages rewrite scoring scripts. 

## References ##


[1] [Improving Attention Based Sequence-to-Sequence Models for End-to-End English Conversational Speech Recognition](https://www.isca-speech.org/archive/Interspeech_2018/abstracts/1030.html), Chao Weng, Jia Cui, Guangsen Wang, Jun Wang, Chengzhu Yu, Dan Su, Dong Yu, InterSpeech 2018

[2] [Minimum Bayes Risk Training of RNN-Transducer for End-to-End Speech Recognition](https://www.isca-speech.org/archive/Interspeech_2020/abstracts/1221.html), Chao Weng, Chengzhu Yu, Jia Cui, Chunlei Zhang, Dong Yu, InterSpeech 2020


## Citations ##

```
@inproceedings{Weng2020,
  author={Chao Weng and Chengzhu Yu and Jia Cui and Chunlei Zhang and Dong Yu},
  title={{Minimum Bayes Risk Training of RNN-Transducer for End-to-End Speech Recognition}},
  year=2020,
  booktitle={Proc. Interspeech 2020},
  pages={966--970},
  doi={10.21437/Interspeech.2020-1221},
  url={http://dx.doi.org/10.21437/Interspeech.2020-1221}
}

@inproceedings{Weng2018,
  author={Chao Weng and Jia Cui and Guangsen Wang and Jun Wang and Chengzhu Yu and Dan Su and Dong Yu},
  title={Improving Attention Based Sequence-to-Sequence Models for End-to-End English Conversational Speech Recognition},
  year=2018,
  booktitle={Proc. Interspeech 2018},
  pages={761--765},
  doi={10.21437/Interspeech.2018-1030},
  url={http://dx.doi.org/10.21437/Interspeech.2018-1030}
}
```

## Disclaimer ##

This is not an officially supported Tencent product

