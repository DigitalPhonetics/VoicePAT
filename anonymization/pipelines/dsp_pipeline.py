from pathlib import Path
from anonymization.modules.dsp.anonymise_dir_mcadams_rand_seed import process_data


class DSPPipeline:
    """
    This pipeline consists of:
          - ASR -> phone sequence                             -
    input - (prosody extractor -> prosody anonymizer)         - TTS -> output
          - speaker embedding extractor -> speaker anonymizer -
    """

    def __init__(self, config):
        self.config = config
        self.libri_360_data_dir = Path(config['dataset_libri_360']) if 'dataset_libri_360' in config else None
        self.modules_config = config['modules']

    def run_anonymization_pipeline(self, datasets):

        for i, (dataset_name, dataset_path) in enumerate(datasets.items()):
            print(f'{i + 1}/{len(datasets)}: Processing {dataset_name}...')
            process_data(dataset_path=dataset_path, anon_level=self.modules_config['anon_level'], settings=self.modules_config)
            print('Done')

        if self.libri_360_data_dir:
            process_data(dataset_path=self.libri_360_data_dir, 
                    anon_level=self.modules_config['anon_level_libri_360'], 
                    settings=self.modules_config)
