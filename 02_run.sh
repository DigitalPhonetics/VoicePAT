#!/bin/bash
source env.sh

#GPU 1 A100



#generate b2 anonymized audio (libri/vctk dev+test set & libri-360h)
#takes around 24hours
python run_anonymization_dsp.py --config anon_dsp.yaml

#perform libri/vctk dev+test pre evaluation using pretrained ASV/ASR models
#ASV-20mins, ASR-9hours if eval_batchsize=3, ASR-12hours if evel_batchsize=2 (default)
python run_evaluation.py --config eval_pre.yaml


#train post ASV/ASR using anonymized libri-360 and perform libri/vctk dev+test post evaluation
#ASV training takes 2hours
#ASR training takes 30hours
#post evaluation: ASV-20mins, ASR-9hours
python run_evaluation.py --config eval_post_scratch.yaml

