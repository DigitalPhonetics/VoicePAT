#!/usr/bin/env bash
#SBATCH --time=72:00:00

cd evaluation/utility/asr_train

module load cuda11.1

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


train_set="train_clean_360"
valid_set="dev"
test_sets="test_clean test_other dev_clean dev_other"
lm_config=conf/tuning/train_lm_transformer2.yaml
inference_config=conf/decode_asr.yaml
anon_data_suffix=
LIBRISPEECH_anon=
pretrain_model=
lr=

LIBRISPEECH=$1
num_utt=$2
num_spk=$3
gpu=$4
exp_dir=$5

if [ $# -gt 8  ]; then
  pretrain_model_dir=$6
  pretrain_model=${pretrain_model_dir}/valid.acc.ave.pth
  LIBRISPEECH_anon=$7
  anon_data_suffix=$8
  lr=$9

  if [[ $lr == "0.002" ]]; then
    asr_config=conf/train_asr_transformer.yaml
  elif [[ $lr == "0.0002" ]]; then
    asr_config=conf/train_asr_transformer_anon.yaml
  elif [[ $lr == "0.00002" ]]; then
    asr_config=conf/train_asr_transformer_anon_2e-4.yaml
  fi
fi




./asr.sh \
    --lang en \
    --ngpu $gpu \
    --expdir ${exp_dir} \
    --pretrained_model ${pretrain_model} \
    --use_lm false \
    --local_data_opts "${LIBRISPEECH} ${LIBRISPEECH_anon} ${anon_data_suffix}" \
    --nbpe 5000 \
    --num_utt ${num_utt} \
    --num_spk ${num_spk} \
    --max_wav_duration 30 \
    --asr_config "${asr_config}" \
    --lm_config "${lm_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text data/local/other_text/text" \
    --bpe_train_text "data/${train_set}/text" #"$@"
