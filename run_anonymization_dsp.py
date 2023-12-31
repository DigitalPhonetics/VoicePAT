from pathlib import Path
from argparse import ArgumentParser
import torch
from anonymization.pipelines.dsp_pipeline import DSPPipeline
from utils import parse_yaml, get_datasets

PIPELINES = {
    'dsp': DSPPipeline
}

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', default='anon_config.yaml')
    parser.add_argument('--gpu_ids', default='0')
    parser.add_argument('--force_compute', default=False, type=bool)
    args = parser.parse_args()

    config = parse_yaml(Path('configs', args.config))
    datasets = get_datasets(config)

    gpus = args.gpu_ids.split(',')

    devices = []
    if torch.cuda.is_available():
        for gpu in gpus:
            devices.append(torch.device(f'cuda:{gpu}'))
    else:
        devices.append(torch.device('cpu'))

    pipeline = PIPELINES[config['pipeline']](config=config)
    pipeline.run_anonymization_pipeline(datasets)

