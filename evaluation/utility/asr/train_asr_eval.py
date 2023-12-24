from pathlib import Path
import os
import subprocess
import torch


def train_asr_eval(params, libri_dir, model_name, model_dir, anon_data_suffix):
    backend = params.get('backend', 'espnet').lower()
    if backend == 'espnet':
        asr_train_espnet(params=params, libri_dir=libri_dir, model_name=model_name, model_dir=model_dir,
                         anon_data_suffix=anon_data_suffix)
    elif backend == 'speechbrain':
        asr_train_speechbrain(params=params, libri_dir=libri_dir, model_name=model_name, model_dir=model_dir,
                              anon_data_suffix=anon_data_suffix)
    else:
        raise ValueError(f'Unknown backend {backend} for ASR evaluation. Available backends: espnet, speechbrain.')


def asr_train_espnet(params, libri_dir, model_name, model_dir, anon_data_suffix):
    print(f'Train ASR model: {model_dir}')
    exp_dir = Path('exp', model_name)
    ngpu = min(params.get('num_gpus', 0), torch.cuda.device_count())  # cannot use more gpus than available

    train_params = [
        '--lang', 'en',
        '--ngpu', str(ngpu),
        '--expdir', str(exp_dir),
        '--use_lm', 'false',
        '--nbpe', '5000',
        '--num_utt', str(params['num_utt']),
        '--num_spk', str(params['num_spk']),
        '--max_wav_duration', '30',
        '--test_sets', "test_clean test_other dev_clean dev_other",
        '--valid_set', "dev",
        '--bpe_train_text', "data/train_clean_360/text",
        '--nj', str(params.get('nj', 5))
    ]

    asr_config = 'conf/train_asr_transformer.yaml'

    if params.get('anon', False):
        local_data_opts = ' '.join([str(libri_dir), str(params['train_data_dir']), anon_data_suffix])
        train_set = f'train_clean_360_{anon_data_suffix}'
        if params.get('finetuning', False) is True:
            asr_config = 'conf/train_asr_transformer_anon.yaml'
            train_params.extend(['--pretrained_model', f'{str(params["pretrained_model"])}/valid.acc.ave.pth'])
    else:
        local_data_opts = str(libri_dir)
        train_set = 'train_clean_360'

    train_params.extend(['--local_data_opts', local_data_opts,
                         '--train_set', train_set,
                         '--asr_config', asr_config])

    cwd = Path.cwd()
    os.chdir('evaluation/utility/asr')  # espnet recipe needs several files at specific relative positions
    print(Path.cwd())
    subprocess.run(['./asr.sh'] + train_params)

    subprocess.run(['ln', '-srf', exp_dir, model_dir])
    os.chdir(cwd)


def asr_train_speechbrain(params, libri_dir, model_name, model_dir, anon_data_suffix):
    raise NotImplementedError('ASR training with speechbrain backend not implemented yet. Use pretrained model '
                              'instead.')
