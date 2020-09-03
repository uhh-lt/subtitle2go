# subtitle2go

Installguide
```
pip install numpy pyyaml
virtualenv -p /usr/bin/python3.7 subtitle2go_env
wget http://ltdata1.informatik.uni-hamburg.de/pykaldi/pykaldi-0.1.2-cp37-cp37m-linux_x86_64.whl
source ./subtitle2go_env/bin/activate
pip install pykaldi-0.1.2-cp37-cp37m-linux_x86_64.whl
git clone https://github.com/pykaldi/pykaldi
pykaldi/tools/install_kaldi.sh ~/projects/subtitle2go/subtitle2go_env/bin/python3
```
And than start:
Put a wavefile in the directory (new.wav) and start the program
The subtitle File is generated as subtitle.vtt
```
start.sh
```
