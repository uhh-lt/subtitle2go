# subtitle2go

## Installguide
```
virtualenv -p /usr/bin/python3.7 subtitle2go_env
pip install numpy pyyaml ffmpeg-python
wget http://ltdata1.informatik.uni-hamburg.de/pykaldi/pykaldi-0.1.2-cp37-cp37m-linux_x86_64.whl
source ./subtitle2go_env/bin/activate
pip install pykaldi-0.1.2-cp37-cp37m-linux_x86_64.whl
git clone https://github.com/pykaldi/pykaldi
pykaldi/tools/install_kaldi.sh ~/projects/subtitle2go/subtitle2go_env/bin/python3
```
Put a mediafile (eg `mediafile.mp4`) in the directory and change in the last row the parameter in the `start.sh` file to:
```
python nnet3-recognizer.py -f "videofile.mp4"
```
Then start the `start.sh` and the subtitle is generated as `mediafile.vtt`

## FAQ

### Error message "had nonzero return status 32512"
The path of kaldi is missing or incorrect
Use
```
export KALDI_ROOT=PATH_TO_KALDI
export PATH=$KALDI_ROOT/src/lmbin/:$KALDI_ROOT/../kaldi_lm/:$PWD/utils/:$KALDI_ROOT/src/bin:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/src/fstbin/:$KALDI_ROOT/src/gmmbin/:$KALDI_ROOT/src/featbin/:$KALDI_ROOT/src/lm/:$KALDI_ROOT/src/sgmmbin/:$KALDI_ROOT/src/sgmm2bin/:$KALDI_ROOT/src/fgmmbin/:$KALDI_ROOT/src/latbin/:$KALDI_ROOT/src/nnetbin:$KALDI_ROOT/src/nnet2bin/:$KALDI_ROOT/src/online2bin/:$KALDI_ROOT/src/ivectorbin/:$KALDI_ROOT/src/kwsbin:$KALDI_ROOT/src/nnet3bin:$KALDI_ROOT/src/chainbin:$KALDI_ROOT/tools/sph2pipe_v2.5/:$KALDI_ROOT/src/rnnlmbin:$PWD:$PATH
```
and replace PATH_TO_KALDI with the path to your kaldi binaries (eg. pykaldi/tools/kaldi)