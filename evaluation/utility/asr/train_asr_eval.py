from pathlib import Path
import os
import subprocess
import torch


def train_asr_eval(params, libri_dir, model_name, model_dir, anon_data_suffix):
    backend = params.get('backend', 'speechbrain').lower()
    if backend == 'speechbrain':
        asr_train_speechbrain(params=params, libri_dir=libri_dir, model_name=model_name, model_dir=model_dir,
                              anon_data_suffix=anon_data_suffix)
    else:
        raise ValueError(f'Unknown backend {backend} for ASR evaluation. Available backends: speechbrain.')


def asr_train_speechbrain(params, libri_dir, model_name, model_dir, anon_data_suffix):
    raise NotImplementedError('ASR training with speechbrain backend not implemented yet. Use pretrained model '
                              'instead.')
