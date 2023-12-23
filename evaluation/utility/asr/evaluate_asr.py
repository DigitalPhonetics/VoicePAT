from pathlib import Path
import os
import subprocess
import torch
from torch.utils.data import DataLoader

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from .speechbrain_asr import InferenceSpeechBrainASR
from .speechbrain_asr.inference import MyDataset
from utils import read_kaldi_format


def evaluate_asr(eval_datasets, eval_data_dir, params, model_path, libri_dir, anon_data_suffix, device, backend):
    if backend == 'espnet':
        asr_eval_espnet_sh(eval_datasets=eval_datasets, eval_data_dir=eval_data_dir, params=params,
                           model_path=model_path, libri_dir=libri_dir, anon_data_suffix=anon_data_suffix)
    elif backend == 'speechbrain':
        asr_eval_speechbrain(eval_datasets=eval_datasets, eval_data_dir=eval_data_dir, params=params,
                             model_path=model_path, anon_data_suffix=anon_data_suffix, device=device)
    else:
        raise ValueError(f'Unknown backend {backend} for ASR evaluation. Available backends: espnet, speechbrain.')


def asr_eval_espnet_sh(eval_datasets, eval_data_dir, params, model_path, libri_dir, anon_data_suffix):
    print(f'Use ASR model for evaluation: {model_path}')
    test_sets = []

    for asr_dataset in eval_datasets:
        anon_asr_dataset = f'{asr_dataset}_{anon_data_suffix}'
        test_sets.append(str(eval_data_dir / asr_dataset))
        test_sets.append(str(eval_data_dir / anon_asr_dataset))

    ngpu = min(params.get('num_gpus', 0), torch.cuda.device_count())  # cannot use more gpus than available

    inference_params = [
        '--ngpu', str(ngpu),
        '--expdir', str(model_path),
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


def asr_eval_speechbrain(eval_datasets, eval_data_dir, params, model_path, anon_data_suffix, device):
    print(f'Use ASR model for evaluation: {model_path}')
    model = InferenceSpeechBrainASR(model_url=params['model_url'], model_path=model_path, device=device)
    results_dir = params['results_dir']
    test_sets = eval_datasets + [f'{asr_dataset}_{anon_data_suffix}' for asr_dataset in eval_datasets]


    with torch.no_grad():
        for test_set in test_sets:
            print(test_set)
            data_path = eval_data_dir / test_set
            dataset = MyDataset(wav_scp_file=Path(data_path, 'wav.scp'), asr_model=model.asr_model)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn)
            hypotheses = model.transcribe_audios(data=dataloader, out_file=Path(results_dir, test_set, 'text'))
            references = read_kaldi_format(Path(data_path, 'text'), values_as_string=True)
            scores = model.compute_wer(ref_texts=references, hyp_texts=hypotheses, out_file=Path(results_dir,
                                                                                                 test_set, 'wer'))
            print(f'{test_set} - WER: {scores.summarize("error_rate")}')

