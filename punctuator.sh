cat $1 | python punctuator2/punctuator.py models/Model_subs_norm1_filt_5M_tageschau_euparl_h256_lr0.02.pcl $2
python punctuator2/convert_to_readable.py $2 $3