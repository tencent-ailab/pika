export LC_ALL=C

#MKL library path
MKL_LIB_PATH=/data/home/cweng/build/anaconda3/lib/

#_clif.so for libkaldi
CLIF_LIB_PATH=/data/home/cweng/build/lib/kaldi/pykaldi/build/lib/kaldi

#kaldi root directory
KALDI_ROOT=/data/home/cweng/build/pykaldi/tools/kaldi

#pika root directory
PIKA_ROOT=/data/home/cweng/build/pika 

#openfst lib path
OPENFST_LIB_PATH=/data/home/cweng/build/pykaldi/tools/kaldi/tools/openfst/lib

export LD_PRELOAD=$MKL_LIB_PATH/libmkl_def.so:$MKL_LIB_PATH/libmkl_avx2.so:$MKL_LIB_PATH/libmkl_core.so:$MKL_LIB_PATH/libmkl_intel_lp64.so:$MKL_LIB_PATH/libmkl_intel_thread.so:$MKL_LIB_PATH/libiomp5.so
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64/:/usr/local/cuda-9.0/lib64:$CLIF_LIB_PATH:$KALDI_ROOT/src/lib:/root//kaldi_libs:$OPENFST_LIB_PATH:$LD_LIBRARY_PATH
