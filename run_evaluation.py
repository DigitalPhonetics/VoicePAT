# We need to set CUDA_VISIBLE_DEVICES before we import Pytorch so we will read all arguments directly on startup
import os
from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
from typing import List

parser = ArgumentParser()
parser.add_argument('--config', default='config_eval.yaml')
parser.add_argument('--gpu_ids', default='0')
args = parser.parse_args()

if 'CUDA_VISIBLE_DEVICES' not in os.environ:  # do not overwrite previously set devices
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids

import torch
import time
import shutil
from copy import deepcopy
import subprocess
import itertools
import numpy as np

from evaluation import ASV, train_asv_speaker_embeddings, VoiceDistinctiveness
from evaluation.privacy.asv.asv_train.speechbrain_defaults import run_opts
from utils import parse_yaml
from utils.path_management import scan_checkpoint


def get_evaluation_steps(params):
    eval_steps = {}
    if 'privacy' in params:
        eval_steps['privacy'] = list(params['privacy'].keys())
    if 'utility' in params:
        eval_steps['utility'] = list(params['utility'].keys())

    if 'eval_steps' in params:
        param_eval_steps = params['eval_steps']

        for eval_part, eval_metrics in param_eval_steps.items():
            if eval_part not in eval_steps:
                raise KeyError(f'Unknown evaluation step {eval_part}, please specify in config')
            for metric in eval_metrics:
                if metric not in eval_steps[eval_part]:
                    raise KeyError(f'Unknown metric {metric}, please specify in config')
        return param_eval_steps
    return eval_steps


def get_eval_trial_datasets(datasets_list):
    eval_pairs = []

    for dataset in datasets_list:
        eval_pairs.extend([(f'{dataset["data"]}_{dataset["set"]}_{enroll}',
                            f'{dataset["data"]}_{dataset["set"]}_{trial}')
                           for enroll, trial in itertools.product(dataset['enrolls'], dataset['trials'])])
    return eval_pairs


def get_eval_asr_datasets(datasets_list):
    eval_data = set()

    for dataset in datasets_list:
        eval_data.add(f'{dataset["data"]}_{dataset["set"]}_asr')

    return list(eval_data)


def find_asv_model_checkpoint(model_dir):
    if list(model_dir.glob('CKPT+*')):  # select latest checkpoint
        model_dir = scan_checkpoint(model_dir, 'CKPT')
    return model_dir


def asv_train(train_params, output_dir):
    print(f'Train ASV model: {output_dir}')
    hparams = {
        'pretrained_path': str(train_params['pretrained_model']),
        'batch_size': train_params['batch_size'],
        'lr': train_params['lr'],
        'num_utt': train_params['num_utt'],
        'num_spk': train_params['num_spk'],
        'utt_selected_ways': train_params['utt_selection'],
        'number_of_epochs': train_params['epochs'],
        'anon': train_params['anon'],
        'data_folder': str(train_params['train_data_dir']),
        'output_folder': str(output_dir)
    }

    config = train_params['train_config']

    if train_params['num_spk'] == 'ALL':
        hparams['out_n_neurons'] = 921
    else:
        hparams['out_n_neurons'] = int(train_params['num_spk'])

    sb_run_opts = deepcopy(run_opts)
    if torch.cuda.device_count() > 0:
        sb_run_opts['data_parallel_backend'] = True
    train_asv_speaker_embeddings(config, hparams, run_opts=sb_run_opts)


def asv_eval(eval_datasets, eval_data_dir, params, device, anon_data_suffix, model_dir=None):
    model_dir = model_dir or find_asv_model_checkpoint(params['model_dir'])
    print(f'Use ASV model for evaluation: {model_dir}')

    save_dir = params['evaluation']['results_dir'] / f'{params["evaluation"]["distance"]}_out'
    asv = ASV(model_dir=model_dir, device=device, score_save_dir=save_dir, distance=params['evaluation']['distance'],
              plda_settings=params['evaluation']['plda'], vec_type=params['vec_type'])

    attack_scenarios = ['oo', 'oa', 'aa']
    get_suffix = lambda x: f'_{anon_data_suffix}' if x == 'a' else ''
    results = []

    for enroll, trial in eval_datasets:
        for scenario in attack_scenarios:
            enroll_name = f'{enroll}{get_suffix(scenario[0])}'
            test_name = f'{trial}{get_suffix(scenario[1])}'

            EER = asv.eer_compute(enrol_dir=eval_data_dir / enroll_name, test_dir=eval_data_dir / test_name,
                                  trial_runs_file=eval_data_dir / trial / 'trials')

            print(f'{enroll_name}-{test_name}: {scenario.upper()}-EER={EER}')
            trials_info = trial.split('_')
            gender = trials_info[3]
            if 'common' in trial:
                gender += '_common'
            results.append({'dataset': trials_info[0], 'split': trials_info[1], 'gender': gender,
                            'enrollment': 'original' if scenario[0] == 'o' else 'anon',
                            'trial': 'original' if scenario[1] == 'o' else 'anon', 'EER': round(EER * 100, 3)})

    results_df = pd.DataFrame(results)
    print(results_df)
    results_df.to_csv(save_dir / 'results.csv')


def gvd_eval(eval_datasets, eval_data_dir, params, device, anon_data_suffix):

    def get_similarity_matrix(vd_model, out_dir, exp_name, segments_folder):
        if (out_dir / f'similarity_matrix_{exp_name}.npy').exists() and not recompute:
            return np.load(out_dir / f'similarity_matrix_{exp_name}.npy')
        else:
            return vd_model.compute_voice_similarity_matrix(segments_dir_x=segments_folder,
                                                            segments_dir_y=segments_folder, exp_name=exp_name,
                                                            out_dir=out_dir)

    num_utt_per_spk = params['num_utt']
    gvd_tag = f'gvd-{num_utt_per_spk}' if isinstance(num_utt_per_spk, int) else 'gvd'
    save_dir = params['results_dir'] / f'{params["asv_params"]["evaluation"]["distance"]}_out' / gvd_tag
    recompute = params.get('recompute', False)

    vd_settings = {
        'device': device,
        'plda_settings': params['asv_params']['evaluation']['plda'],
        'distance': params['asv_params']['evaluation']['distance'],
        'vec_type': params['asv_params']['vec_type'],
        'num_per_spk': num_utt_per_spk
    }

    if 'model_dir' in params['asv_params']:  # use the same ASV model for original and anon speaker space
        spk_ext_model_dir = find_asv_model_checkpoint(params['asv_params']['model_dir'])
        vd = VoiceDistinctiveness(spk_ext_model_dir=spk_ext_model_dir, score_save_dir=save_dir,
                                  **vd_settings)
        vd_orig, vd_anon = None, None
        save_dir_orig, save_dir_anon = None, None
        print(f'Use ASV model {spk_ext_model_dir} for computing voice similarities of original and anonymized speakers')
    elif 'orig_model_dir' in params['asv_params'] and 'anon_model_dir' in params['asv_params']:
        # use different ASV models for original and anon speaker spaces
        spk_ext_model_dir_orig = find_asv_model_checkpoint(params['asv_params']['orig_model_dir'])
        save_dir_orig = params['orig_results_dir'] / f'{params["asv_params"]["evaluation"]["distance"]}_out' / gvd_tag
        vd_orig = VoiceDistinctiveness(spk_ext_model_dir=spk_ext_model_dir_orig, score_save_dir=save_dir_orig,
                                       **vd_settings)
        spk_ext_model_dir_anon = find_asv_model_checkpoint(params['asv_params']['anon_model_dir'])
        save_dir_anon = params['anon_results_dir'] / f'{params["asv_params"]["evaluation"]["distance"]}_out' / gvd_tag
        vd_anon = VoiceDistinctiveness(spk_ext_model_dir=spk_ext_model_dir_anon,  score_save_dir=save_dir_anon,
                                       **vd_settings)
        vd = None
        print(f'Use ASV model {spk_ext_model_dir_orig} for computing voice similarities of original speakers and ASV '
              f'model {spk_ext_model_dir_anon} for voice similarities of anonymized speakers')
    else:
        raise ValueError('GVD: You either need to specify one "model_dir" for both original and anonymized data or '
                         'one "orig_model_dir" and one "anon_model_dir"!')

    for _, trial in eval_datasets:
        osp_set_folder = eval_data_dir / trial
        psp_set_folder = eval_data_dir / f'{trial}_{anon_data_suffix}'
        trial_out_dir = save_dir / trial
        trial_out_dir.mkdir(exist_ok=True, parents=True)

        if vd_orig:
            orig_trial_out_dir = save_dir_orig / trial
            orig_trial_out_dir.mkdir(exist_ok=True, parents=True)
            oo_sim = get_similarity_matrix(vd_model=vd_orig, out_dir=orig_trial_out_dir, exp_name='osp_osp',
                                           segments_folder=osp_set_folder)
        else:
            oo_sim = get_similarity_matrix(vd_model=vd, out_dir=trial_out_dir, exp_name='osp_osp',
                                           segments_folder=osp_set_folder)
        if vd_anon:
            anon_trial_out_dir = save_dir_anon / trial
            anon_trial_out_dir.mkdir(exist_ok=True, parents=True)
            pp_sim = get_similarity_matrix(vd_model=vd_anon, out_dir=anon_trial_out_dir, exp_name='psp_psp',
                                           segments_folder=psp_set_folder)
        else:
            pp_sim = get_similarity_matrix(vd_model=vd, out_dir=trial_out_dir, exp_name='psp_psp',
                                           segments_folder=psp_set_folder)

        gvd_value = vd.gvd(oo_sim, pp_sim) if vd else vd_orig.gvd(oo_sim, pp_sim)
        with open(trial_out_dir / 'gain_of_voice_distinctiveness', 'w') as f:
            f.write(str(gvd_value))
        print(f'{trial} gvd={gvd_value}')


def asr_train(params: dict, libri_dir: Path, model_name: str, model_dir: Path, anon_data_suffix: str):
    print(f'Train ASR model: {model_dir}')
    exp_dir = Path('exp', model_name)
    libri_dir = Path(libri_dir).expanduser() # could be relative to userdir
    ngpu = min(params.get('num_gpus', 0), torch.cuda.device_count())  # cannot use more gpus than available

    train_params = [
        '--lang', 'en',
        '--ngpu', str(ngpu),
        '--expdir', str(exp_dir.absolute()),
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
        local_data_opts = ' '.join([str(libri_dir.absolute()), str(params['train_data_dir'].absolute()), anon_data_suffix])
        train_set = f'train_clean_360_{anon_data_suffix}'
        if params.get('finetuning', False) is True:
            asr_config = 'conf/train_asr_transformer_anon.yaml'
            train_params.extend(['--pretrained_model', f'{str(params["pretrained_model"].absolute())}/valid.acc.ave.pth'])
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


def asr_eval_sh(eval_datasets: List[str], eval_data_dir: Path, params, model_path, libri_dir, anon_data_suffix):
    print(f'Use ASR model for evaluation: {model_path}')
    test_sets = []

    for asr_dataset in eval_datasets:
        anon_asr_dataset = f'{asr_dataset}_{anon_data_suffix}'
        test_sets.append(str((eval_data_dir / asr_dataset).absolute()))
        test_sets.append(str((eval_data_dir / anon_asr_dataset).absolute()))

    ngpu = min(params.get('num_gpus', 0), torch.cuda.device_count())  # cannot use more gpus than available

    inference_params = [
        '--ngpu', str(ngpu),
        '--expdir', str(model_path.absolute()),
        '--asr_exp', str(model_path),
        '--use_lm', 'true',
        '--local_data_opts', str(libri_dir),
        '--nbpe', '5000',
        '--lm_config', str(params['lm_dir'] / 'config.yaml'),
        '--lm_exp', str(params['lm_dir']),
        '--inference_config', 'conf/decode_asr.yaml',
        '--test_sets', ' '.join(test_sets),
        '--skip_train', 'true',
        '--gpu_inference', 'true',
        '--inference_nj', str(params.get('nj', 5)),
        '--inference_asr_model', 'valid.acc.ave.pth',
    ]

    cwd = Path.cwd()
    os.chdir('evaluation/utility/asr')
    subprocess.run(['./asr.sh'] + inference_params)
    os.chdir(cwd)


if __name__ == '__main__':
    params = parse_yaml(Path('configs', args.config))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    eval_steps = get_evaluation_steps(params)
    eval_data_trials = get_eval_trial_datasets(params['datasets'])
    eval_data_asr = get_eval_asr_datasets(params['datasets'])
    eval_data_dir = params['eval_data_dir']
    anon_suffix = params['anon_data_suffix']

    # make sure given paths exist
    assert eval_data_dir.exists(), f'{eval_data_dir} does not exist'

    if 'privacy' in eval_steps:
        if 'asv' in eval_steps['privacy']:
            asv_params = params['privacy']['asv']
            model_dir = find_asv_model_checkpoint(asv_params['model_dir'])
            if 'training' in asv_params:
                asv_train_params = asv_params['training']
                if not model_dir.exists() or asv_train_params.get('retrain', True) is True:
                    start_time = time.time()
                    print('Perform ASV training')
                    asv_train(train_params=asv_train_params, output_dir=asv_params['model_dir'])
                    print("ASV training time: %f min ---" % (float(time.time() - start_time) / 60))
                    model_dir = scan_checkpoint(model_dir, 'CKPT')
                    if asv_params['vec_type'] == 'xvector':
                        shutil.copy('evaluation/privacy/asv/asv_train/hparams/xvector/hyperparams.yaml', model_dir)
                    else:
                        shutil.copy('evaluation/privacy/asv/asv_train/hparams/ecapa/hyperparams.yaml', model_dir)

            if 'evaluation' in asv_params:
                print('Perform ASV evaluation')
                start_time = time.time()
                asv_eval(eval_datasets=eval_data_trials, eval_data_dir=eval_data_dir, params=asv_params, device=device,
                         model_dir=model_dir, anon_data_suffix=anon_suffix)
                print("--- EER computation time: %f min ---" % (float(time.time() - start_time) / 60))

    if 'utility' in eval_steps:
        if 'asr' in eval_steps['utility']:
            asr_params = params['utility']['asr']
            model_name = asr_params['model_name']
            model_dir = asr_params['model_dir']
            libri_dir = asr_params['libri_dir']
            asr_model_path = None

            if 'training' in asr_params:
                asr_train_params = asr_params['training']
                asr_model_path = model_dir / 'asr_train_asr_transformer_raw_en_bpe5000'
                if asr_train_params['anon'] and asr_train_params['finetuning']:
                    asr_model_path = model_dir / 'asr_train_asr_transformer_anon_raw_en_bpe5000'

                if not model_dir.exists() or asr_train_params.get('retrain', True) is True:
                    start_time = time.time()
                    print('Perform ASR training')
                    asr_train(params=asr_train_params, libri_dir=libri_dir, model_name=model_name,
                              model_dir=model_dir, anon_data_suffix=anon_suffix)
                    print("--- ASR training time: %f min ---" % (float(time.time() - start_time) / 60))

            if 'evaluation' in asr_params:
                asr_eval_params = asr_params['evaluation']
                if not asr_model_path:
                    if asr_eval_params.get('anon_train_model', False):
                        asr_model_path = model_dir / 'asr_train_asr_transformer_anon_raw_en_bpe5000'
                    else:
                        asr_model_path = model_dir / 'asr_train_asr_transformer_raw_en_bpe5000'

                start_time = time.time()
                print('Perform ASR evaluation')
                asr_eval_sh(eval_datasets=eval_data_asr, eval_data_dir=eval_data_dir, params=asr_eval_params,
                            model_path=asr_model_path, anon_data_suffix=anon_suffix, libri_dir=libri_dir)
                print("--- ASR evaluation time: %f min ---" % (float(time.time() - start_time) / 60))

        if 'gvd' in eval_steps['utility']:
            gvd_params = params['utility']['gvd']
            start_time = time.time()
            print('Perform GVD evaluation')
            gvd_eval(eval_datasets=eval_data_trials, eval_data_dir=eval_data_dir, params=gvd_params, device=device,
                     anon_data_suffix=anon_suffix)
            print("--- GVD  computation time: %f min ---" % (float(time.time() - start_time) / 60))
