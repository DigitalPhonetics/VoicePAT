from argparse import ArgumentParser
import logging
from pathlib import Path
import time
import itertools
import os

parser = ArgumentParser()
parser.add_argument('--config', default='config_eval.yaml')
args = parser.parse_args()

from evaluation import evaluate_asv_bias 
from utils import parse_yaml


def get_eval_trial_datasets(datasets_list, anon_data_suffix):
    eval_pairs = []

    for dataset in datasets_list:
        eval_pairs.extend([(f'{dataset["data"]}_{dataset["set"]}_{enroll}',
                            f'{dataset["data"]}_{dataset["set"]}_{trial}')
                           for enroll, trial in itertools.product(dataset['enrolls'], dataset['trials'])])
        eval_pairs.extend([(f'{dataset["data"]}_{dataset["set"]}_{enroll}',
                            f'{dataset["data"]}_{dataset["set"]}_{trial}_{anon_data_suffix}')
                           for enroll, trial in itertools.product(dataset['enrolls'], dataset['trials'])])
        eval_pairs.extend([(f'{dataset["data"]}_{dataset["set"]}_{enroll}_{anon_data_suffix}',
                            f'{dataset["data"]}_{dataset["set"]}_{trial}_{anon_data_suffix}')
                            for enroll, trial in itertools.product(dataset['enrolls'], dataset['trials'])
                           ])

    return eval_pairs
                   

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s- %(levelname)s - %(message)s')

    params = parse_yaml(Path('configs', args.config))

    
    eval_steps = {}
    if 'bias' not in params:
        raise KeyError(f'please specify bias evaluation parameter in config')
    else:
        eval_steps['bias'] = list(params['bias'].keys())   

    if 'asv' in eval_steps['bias']:
        anon_data_suffix = params['anon_data_suffix']
        eval_data_trials = get_eval_trial_datasets(params['datasets'], anon_data_suffix)
        asv_params = params['bias']['asv']
        results_dir = asv_params['results_dir']
        bt4vt_config_file = asv_params['bt4vt_config_file']
        gender_split = asv_params['gender_split']

        logging.info('Perform ASV Bias evaluation')
        start_time = time.time()

        results = []
        for enroll, trial in eval_data_trials:
            logging.info(f'{enroll}-{trial}')

            anon_trial = False

            if gender_split:
                if trial.endswith(anon_data_suffix):
                    trial = trial[:-(len(anon_data_suffix)+1)]
                    anon_trial = True
                    if trial.endswith('f'):
                        scores_file_female = Path(f'{results_dir}/{enroll}-{trial}_{anon_data_suffix}/scores')
                        trial_m = trial[:-1] + 'm'
                        scores_file_male = Path(f'{results_dir}/{enroll}-{trial_m}_{anon_data_suffix}/scores')
                    elif trial.endswith('m'):
                        scores_file_male = Path(f'{results_dir}/{enroll}-{trial}_{anon_data_suffix}/scores')
                        trial_f = trial[:-1] + 'f'
                        scores_file_female = Path(f'{results_dir}/{enroll}-{trial_f}_{anon_data_suffix}/scores')
                    else:
                        raise KeyError(f'No Gender Split (f/m) found')
                else:
                    if trial.endswith('f'):
                        scores_file_female = Path(f'{results_dir}/{enroll}-{trial}/scores')
                        trial_m = trial[:-1] + 'm'
                        scores_file_male = Path(f'{results_dir}/{enroll}-{trial_m}/scores')
                    elif trial.endswith('m'):
                        scores_file_male = Path(f'{results_dir}/{enroll}-{trial}/scores')
                        trial_f = trial[:-1] + 'f'
                        scores_file_female = Path(f'{results_dir}/{enroll}-{trial_f}/scores')
                    else:
                        raise KeyError(f'No Gender Split (f/m) found')

                if not scores_file_male.is_file():
                    raise FileNotFoundError(f'Male Scores File does not exist')
                if not scores_file_female.is_file():
                    raise FileNotFoundError(f'Female Scores File does not exist')

                with open(scores_file_male, 'r') as file_male, open(scores_file_female, 'r') as file_female:
                    if anon_trial:
                        output_file_path = Path(f'{results_dir}/{enroll}-{trial[:-2]}_{anon_data_suffix}/scores')
                    else:
                        output_file_path = Path(f'{results_dir}/{enroll}-{trial[:-2]}/scores')
                    
                    if output_file_path.is_file():
                        logging.info("Output File already exists")
                        continue
                    else:
                        os.makedirs(os.path.dirname(output_file_path))

                    with open(output_file_path, 'w') as output_file:
                        for line in file_male:
                            output_file.write(line)
                        for line in file_female:
                            output_file.write(line)
                
                scores_file = output_file_path
            else:
                scores_file = Path(f'{results_dir}/{enroll}-{trial}/scores')
            
            if not scores_file.is_file():
                raise FileNotFoundError(f'Scores File does not exist')

            bias_eval = evaluate_asv_bias(scores_file=scores_file, bt4vt_config_file=bt4vt_config_file)
            logging.info(bias_eval.metrics)

            results.append(bias_eval)

        logging.info("--- Bias computation time: %f min ---" % (float(time.time() - start_time) / 60))