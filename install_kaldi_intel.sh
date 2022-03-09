#!/bin/bash

# Installation script for Kaldi
#
set -e

KALDI_GIT="-b pykaldi_02 https://github.com/pykaldi/kaldi.git"

KALDI_DIR="$PWD/kaldi"

if [ ! -d "$KALDI_DIR" ]; then
  git clone $KALDI_GIT $KALDI_DIR
else
  echo "$KALDI_DIR already exists!"
fi

cd "$KALDI_DIR/tools"
git pull

# Prevent kaldi from switching default python version
mkdir -p "python"
touch "python/.use_default_python"

./extras/check_dependencies.sh

make -j$(nproc)

cd ../src
./configure --shared --cudatk-dir=/usr/local/cuda/ --mathlib=MKL --mkl-root=/opt/intel/mkl/ --use-cuda=no --static-math=yes
make clean -j$(nproc) && make depend -j$(nproc) && make -j$(nproc)

echo "Done installing Kaldi."
