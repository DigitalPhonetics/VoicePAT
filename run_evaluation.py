# We need to set CUDA_VISIBLE_DEVICES before we import Pytorch so we will read all arguments directly on startup
import os
from argparse import ArgumentParser
from pathlib import Path

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
import itertools

from evaluation import evaluate_asv, train_asv_eval, evaluate_asr, train_asr_eval, evaluate_gvd
from utils import parse_yaml, find_asv_model_checkpoint, scan_checkpoint


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


if __name__ == '__main__':
    params = parse_yaml(Path('configs', args.config))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    eval_steps = get_evaluation_steps(params)
    eval_data_trials = get_eval_trial_datasets(params['datasets'])
    eval_data_asr = get_eval_asr_datasets(params['datasets'])
    eval_data_dir = params['eval_data_dir']
    anon_suffix = params['anon_data_suffix']

    if 'privacy' in eval_steps:
        if 'asv' in eval_steps['privacy']:
            asv_params = params['privacy']['asv']
            model_dir = find_asv_model_checkpoint(asv_params['model_dir'])
            if 'training' in asv_params:
                asv_train_params = asv_params['training']
                if not model_dir.exists() or asv_train_params.get('retrain', True) is True:
                    start_time = time.time()
                    print('Perform ASV training')
                    train_asv_eval(train_params=asv_train_params, output_dir=asv_params['model_dir'])
                    print("ASV training time: %f min ---" % (float(time.time() - start_time) / 60))
                    model_dir = scan_checkpoint(model_dir, 'CKPT')
                    if asv_params['vec_type'] == 'xvector':
                        shutil.copy('evaluation/privacy/asv/asv_train/hparams/xvector/hyperparams.yaml', model_dir)
                    else:
                        shutil.copy('evaluation/privacy/asv/asv_train/hparams/ecapa/hyperparams.yaml', model_dir)

            if 'evaluation' in asv_params:
                print('Perform ASV evaluation')
                start_time = time.time()
                evaluate_asv(eval_datasets=eval_data_trials, eval_data_dir=eval_data_dir, params=asv_params,
                             device=device,  model_dir=model_dir, anon_data_suffix=anon_suffix)
                print("--- EER computation time: %f min ---" % (float(time.time() - start_time) / 60))

    if 'utility' in eval_steps:
        if 'asr' in eval_steps['utility']:
            asr_params = params['utility']['asr']
            model_name = asr_params['model_name']
            model_dir = asr_params['model_dir']
            libri_dir = asr_params['libri_dir']
            backend = asr_params['backend'].lower()
            asr_model_path = None

            if 'training' in asr_params:
                asr_train_params = asr_params['training']
                asr_model_path = model_dir / 'asr_train_asr_transformer_raw_en_bpe5000'
                if asr_train_params['anon'] and asr_train_params['finetuning']:
                    asr_model_path = model_dir / 'asr_train_asr_transformer_anon_raw_en_bpe5000'

                if not model_dir.exists() or asr_train_params.get('retrain', True) is True:
                    start_time = time.time()
                    print('Perform ASR training')
                    train_asr_eval(params=asr_train_params, libri_dir=libri_dir, model_name=model_name,
                                   model_dir=model_dir, anon_data_suffix=anon_suffix)
                    print("--- ASR training time: %f min ---" % (float(time.time() - start_time) / 60))

            if 'evaluation' in asr_params:
                asr_eval_params = asr_params['evaluation']
                if not asr_model_path:
                    if backend == 'espnet':
                        if asr_eval_params.get('anon_train_model', False):
                            asr_model_path = model_dir / 'asr_train_asr_transformer_anon_raw_en_bpe5000'
                        else:
                            asr_model_path = model_dir / 'asr_train_asr_transformer_raw_en_bpe5000'
                    else:
                        asr_model_path = model_dir

                start_time = time.time()
                print('Perform ASR evaluation')
                evaluate_asr(eval_datasets=eval_data_asr, eval_data_dir=eval_data_dir, params=asr_eval_params,
                             model_path=asr_model_path, anon_data_suffix=anon_suffix, libri_dir=libri_dir,
                             device=device, backend=backend)
                print("--- ASR evaluation time: %f min ---" % (float(time.time() - start_time) / 60))

        if 'gvd' in eval_steps['utility']:
            gvd_params = params['utility']['gvd']
            start_time = time.time()
            print('Perform GVD evaluation')
            evaluate_gvd(eval_datasets=eval_data_trials, eval_data_dir=eval_data_dir, params=gvd_params,
                         device=device, anon_data_suffix=anon_suffix)
            print("--- GVD  computation time: %f min ---" % (float(time.time() - start_time) / 60))
