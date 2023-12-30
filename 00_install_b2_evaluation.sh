#!/bin/bash

set -e


conda_url=https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh
venv_dir=$PWD/venv

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
  pip install -r requirements_b2_evaluation.txt || exit 1
  touch $mark
fi
echo "if [ \$(which python) != $venv_dir/bin/python ]; then source $venv_dir/bin/activate; fi" > env.sh



