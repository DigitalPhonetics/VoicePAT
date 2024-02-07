from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader

import logging
logger = logging.getLogger(__name__)

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from .speechbrain_asr import InferenceSpeechBrainASR
from .speechbrain_asr.inference import ASRDataset
from utils import read_kaldi_format


def evaluate_asr(eval_datasets, eval_data_dir, params, model_path, anon_data_suffix, device, backend):
    if backend == 'speechbrain':
        return asr_eval_speechbrain(eval_datasets=eval_datasets, eval_data_dir=eval_data_dir, params=params,
                                    model_path=model_path, anon_data_suffix=anon_data_suffix, device=device)
    else:
        raise ValueError(f'Unknown backend {backend} for ASR evaluation. Available backends: speechbrain.')


def asr_eval_speechbrain(eval_datasets, eval_data_dir, params, model_path, anon_data_suffix, device):
    print(f'Use ASR model for evaluation: {model_path}')
    model = InferenceSpeechBrainASR(model_path=model_path, device=device)
    results_dir = params['results_dir']
    test_sets = eval_datasets + [f'{asr_dataset}_{anon_data_suffix}' for asr_dataset in eval_datasets]
    results = []


    with torch.no_grad():
        for test_set in test_sets:
            data_path = eval_data_dir / test_set
            if (results_dir / test_set / 'wer').exists() and (results_dir / test_set / 'text').exists():
                logger.info("No WER computation  necessary; print exsiting WER results")
                hypotheses = read_kaldi_format(Path(data_path, 'text'), values_as_string=True)
                references = read_kaldi_format(Path(results_dir, test_set, 'text'), values_as_string=True)
                scores = model.compute_wer(ref_texts=references, hyp_texts=hypotheses, out_file=Path(results_dir,test_set, 'wer'))
            else:
                dataset = ASRDataset(wav_scp_file=Path(data_path, 'wav.scp'), asr_model=model.asr_model)
                dataloader = DataLoader(dataset, batch_size=params['eval_batchsize'], shuffle=False, num_workers=1, collate_fn=dataset.collate_fn)
                hypotheses = model.transcribe_audios(data=dataloader, out_file=Path(results_dir, test_set, 'text'))
                references = read_kaldi_format(Path(data_path, 'text'), values_as_string=True)
                scores = model.compute_wer(ref_texts=references, hyp_texts=hypotheses, out_file=Path(results_dir,
                                                                                     test_set, 'wer'))
            wer = scores.summarize("error_rate")
            test_set_info = test_set.split('_')
            results.append({'dataset': test_set_info[0], 'split': test_set_info[1],
                            'asr': 'anon' if anon_data_suffix in test_set else 'original', 'WER': round(wer, 3)})
            print(f'{test_set} - WER: {wer}')
    results_df = pd.DataFrame(results)
    print(results_df)
    results_df.to_csv(results_dir / 'results.csv')
    return results_df


