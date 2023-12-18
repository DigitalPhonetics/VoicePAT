#!/bin/sh


ori_dir=/home/smg/zengchang/work_xx/VoicePAT
model=.
for dset in libri_dev_asr  libri_test_asr vctk_dev_asr  vctk_test_asr; do
    echo $dset
		python compute_wer.py --mode present ${ori_dir}/evaluation/utility/asr/dump/raw/$dset/text $dset.txt
		echo ${dset}_dsp
        python /compute_wer.py --mode present ${ori_dir}/evaluation/utility/asr/dump/raw/$dset/text ${dset}_dsp.txt
done 

