#!/bin/sh

cd models/

# de models:
#wget http://ltdata1.informatik.uni-hamburg.de/kaldi_tuda_de/de_683k_nnet3chain_tdnn1f_2048_sp_bi_smaller_fst.tar.bz2
#tar xvfj de_683k_nnet3chain_tdnn1f_2048_sp_bi_smaller_fst.tar.bz2

#wget https://ltdata1.informatik.uni-hamburg.de/kaldi_tuda_de/de_722k_nnet3chain_tdnn1f_2048_sp_bi.tar.bz2
#wget https://ltdata1.informatik.uni-hamburg.de/kaldi_tuda_de/de_722k_G.carpa.bz2
#wget https://ltdata1.informatik.uni-hamburg.de/kaldi_tuda_de/de_722k_rnnlm_lstm_2x.tar.bz2

wget http://ltdata1.informatik.uni-hamburg.de/kaldi_tuda_de/de_900k_nnet3chain_tdnn1f_2048_sp_bi.tar.bz2
wget http://ltdata1.informatik.uni-hamburg.de/kaldi_tuda_de/de_900k_G.carpa.bz2
wget http://ltdata1.informatik.uni-hamburg.de/kaldi_tuda_de/de_900k_rnnlm_lstm_4x.tar.bz2

wget https://ltdata1.informatik.uni-hamburg.de/subtitle2go/Model_subs_norm1_filt_5M_tageschau_euparl_h256_lr0.02.pcl

# en models
wget https://ltdata1.informatik.uni-hamburg.de/pykaldi/en_160k_nnet3chain_tdnn1f_2048_sp_bi.tar.bz2

wget https://ltdata1.informatik.uni-hamburg.de/kaldi_tuda_de/de_900k_rnnlm_lstm_4x.tar.bz2

tar xvfj de_900k_nnet3chain_tdnn1f_2048_sp_bi.tar.bz2
bunzip2 de_900k_G.carpa.bz2
tar xvfj de_900k_rnnlm_lstm_4x.tar.bz2

tar xvfj en_160k_nnet3chain_tdnn1f_2048_sp_bi.tar.bz2

tar xvfj de_900k_rnnlm_lstm_4x.tar.bz2

wget http://ltdata1.informatik.uni-hamburg.de/subtitle2go/interpunct_de_rpunct.tar.gz
tar xfvz interpunct_de_rpunct.tar.gz

cd ..
