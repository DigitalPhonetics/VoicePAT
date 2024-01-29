import pandas as pd

from .asv import ASV
from utils import find_asv_model_checkpoint


def evaluate_asv(eval_datasets, eval_data_dir, params, device, anon_data_suffix, model_dir=None):
    backend = params.get('backend', 'speechbrain').lower()
    if backend == 'speechbrain':
        return asv_eval_speechbrain(eval_datasets=eval_datasets, eval_data_dir=eval_data_dir, params=params,
                                    device=device, anon_data_suffix=anon_data_suffix, model_dir=model_dir)
    else:
        raise ValueError(f'Unknown backend {backend} for ASR evaluation. Available backends: speechbrain.')


def asv_eval_speechbrain(eval_datasets, eval_data_dir, params, device, anon_data_suffix, model_dir=None):
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
    return results_df
