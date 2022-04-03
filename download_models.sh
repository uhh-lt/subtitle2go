mkdir models/
cd models/
mv ../kaldi_tuda_de_nnet3_chain2.yaml ../kaldi_tuda_de_nnet3_chain2_de_722k.yaml ./
#wget http://ltdata1.informatik.uni-hamburg.de/kaldi_tuda_de/de_683k_nnet3chain_tdnn1f_2048_sp_bi_smaller_fst.tar.bz2
wget http://ltdata1.informatik.uni-hamburg.de/kaldi_tuda_de/de_722k_nnet3chain_tdnn1f_2048_sp_bi.tar.bz2
wget http://ltdata1.informatik.uni-hamburg.de/subtitle2go/Model_subs_norm1_filt_5M_tageschau_euparl_h256_lr0.02.pcl
tar xvfj de_722k_nnet3chain_tdnn1f_2048_sp_bi.tar.bz2
#tar xvfj de_683k_nnet3chain_tdnn1f_2048_sp_bi_smaller_fst.tar.bz2
