#!/bin/bash
source env.sh

python run_anonymization.py --config anon_ims_sttts_pc.yaml --gpu_ids 0  --force_compute True
