# subtitle2go

## Changelog

### 24.10.2020
+ Added Punctuation Model
+ Added advanced Segmentation


## Requirements
+ Ubuntu 18.04
+ Python 3.7

## Installguide
```
#make sure you have Python 3.7 installed and also its dev package:
sudo apt-get install python3.7 python3.7-dev

# Now clone the subtitle2go package somwhere:
mkdir ~/projects/
cd ~/projects/
git clone https://github.com/uhh-lt/subtitle2go
cd subtitle2go/

# create virtual env and install python dependencies:

virtualenv -p /usr/bin/python3.7 subtitle2go_env
source subtitle2go_env/bin/activate
pip install numpy pyyaml ffmpeg-python theano spacy
python -m spacy download de

# Now install PyKaldi
wget http://ltdata1.informatik.uni-hamburg.de/pykaldi/pykaldi-0.1.2-cp37-cp37m-linux_x86_64.whl
pip install pykaldi-0.1.2-cp37-cp37m-linux_x86_64.whl
git clone https://github.com/pykaldi/pykaldi
pykaldi/tools/install_kaldi.sh ~/projects/subtitle2go/subtitle2go_env/bin/python3

# Install punctuator2 for automatic punctuation
git clone https://github.com/ottokart/punctuator2.git

# Patch punctuator2:

Open punctuator2/models.py in a file editor, go to line 54 and replace "from . import models" with "import models"

# Download pretrained models:
./download_models.sh
```
Put a mediafile (eg `mediafile.mp4`) in the directory and then run:

```
source subtitle2go_env/bin/activate
. path.sh
python nnet3-recognizer.py -f "mediafile.mp4"
```

The subtitle is then generated as `mediafile.vtt`

## FAQ

### Error message "had nonzero return status 32512"
The path of kaldi is missing or incorrect
Use
```
export KALDI_ROOT=PATH_TO_KALDI
export PATH=$KALDI_ROOT/src/lmbin/:$KALDI_ROOT/../kaldi_lm/:$PWD/utils/:$KALDI_ROOT/src/bin:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/src/fstbin/:$KALDI_ROOT/src/gmmbin/:$KALDI_ROOT/src/featbin/:$KALDI_ROOT/src/lm/:$KALDI_ROOT/src/sgmmbin/:$KALDI_ROOT/src/sgmm2bin/:$KALDI_ROOT/src/fgmmbin/:$KALDI_ROOT/src/latbin/:$KALDI_ROOT/src/nnetbin:$KALDI_ROOT/src/nnet2bin/:$KALDI_ROOT/src/online2bin/:$KALDI_ROOT/src/ivectorbin/:$KALDI_ROOT/src/kwsbin:$KALDI_ROOT/src/nnet3bin:$KALDI_ROOT/src/chainbin:$KALDI_ROOT/tools/sph2pipe_v2.5/:$KALDI_ROOT/src/rnnlmbin:$PWD:$PATH
```
and replace PATH_TO_KALDI with the path to your kaldi binaries (eg. pykaldi/tools/kaldi)
