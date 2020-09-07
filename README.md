# subtitle2go

Installguide
```
pip install numpy pyyaml ffmpeg-python
virtualenv -p /usr/bin/python3.7 subtitle2go_env
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