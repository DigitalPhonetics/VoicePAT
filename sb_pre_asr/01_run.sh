#!/bin/bash

source ../VoicePAT/env.sh
begin=$(date +%s) 
wav_dir=data
 for dset in  libri_dev_asr \
	        vctk_dev_asr \
		libri_test_asr \
		vctk_test_asr; do

    python main.py ${wav_dir}/$dset/wav.scp 
    python main.py ${wav_dir}/${dset}_dsp/wav.scp

done

end=$(date +%s)
tottime=$(expr $end - $begin)
echo $tottime
