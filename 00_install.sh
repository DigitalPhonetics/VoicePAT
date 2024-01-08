#!/bin/bash

set -e

#Install requirements for B2 generation and evaluation
conda_url=https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh
venv_dir=$PWD/venv
root_dir=$PWD
mark=.done-venv
if [ ! -f $mark ]; then
  echo 'Making python virtual environment'
  name=$(basename $conda_url)
  if [ ! -f $name ]; then
    wget $conda_url || exit 1
  fi
  [ ! -f $name ] && echo "File $name does not exist" && exit 1
  [ -d $venv_dir ] && rm -r $venv_dir
  sh $name -b -p $venv_dir || exit 1
  . $venv_dir/bin/activate
  echo 'Installing python dependencies'
  conda install -c conda-forge libflac -y || exit 1 
  conda install -c conda-forge python-sounddevice -y || exit 1
  conda install -c conda-forge cvxopt -y || exit 1
  conda install -c conda-forge typeguard==2.13.3 -y || exit 1 ## indicate version  
  pip install -r requirements_xx.txt || exit 1
  url=https://github.com/espeak-ng/espeak-ng/releases/download/1.50/espeak-ng-1.50.tgz

  #install espeak
  cd ${venv_dir}
  wget $url || exit 1
  tar -xf espeak-ng-1.50.tgz
  cd espeak-ng
  ./autogen.sh
  ./configure --prefix=$venv_dir
  make
  make install
  cd $root_dir
  touch $mark
fi
echo "if [ ! -x \"\$(command -v python)\" ] || [ \$(which python) != $venv_dir/bin/python ]; then source $venv_dir/bin/activate; fi" > env.sh


