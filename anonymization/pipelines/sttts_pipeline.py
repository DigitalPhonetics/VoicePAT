from pathlib import Path
from datetime import datetime
import time

from anonymization.modules import SpeechRecognition, SpeechSynthesis, ProsodyExtraction, ProsodyAnonymization, SpeakerExtraction, \
    SpeakerAnonymization
from utils import prepare_evaluation_data, save_yaml


class STTTSPipeline:
    """
    This pipeline consists of:
          - ASR -> phone sequence                             -
    input - (prosody extractor -> prosody anonymizer)         - TTS -> output
          - speaker embedding extractor -> speaker anonymizer -
    """

    def __init__(self, config, force_compute_all, devices):
        self.total_start_time = time.time()
        self.config = config
        model_dir = Path(config.get('models_dir', 'models'))
        vectors_dir = Path(config.get('vectors_dir', 'original_speaker_embeddings'))
        self.results_dir = Path(config.get('results_dir', 'results'))
        self.data_dir = Path(config['data_dir']) if 'data_dir' in config else None
        save_intermediate = config.get('save_intermediate', True)


        modules_config = config['modules']

        # ASR component
        self.speech_recognition = SpeechRecognition(devices=devices, save_intermediate=save_intermediate,
                                                    settings=modules_config['asr'], force_compute=force_compute_all)

        # Speaker component
        self.speaker_extraction = SpeakerExtraction(model_dir=model_dir, devices=devices,
                                                    save_intermediate=save_intermediate,
                                                    settings=modules_config['speaker_embeddings'],
                                                    force_compute=force_compute_all)
        if 'anonymizer' in modules_config['speaker_embeddings']:
            self.speaker_anonymization = SpeakerAnonymization(vectors_dir=vectors_dir, device=devices[0],
                                                              save_intermediate=save_intermediate,
                                                              settings=modules_config['speaker_embeddings'],
                                                              force_compute=force_compute_all)
        else:
            self.speaker_anonymization = None

        # Prosody component
        if 'prosody' in modules_config:
            self.prosody_extraction = ProsodyExtraction(device=devices[0], save_intermediate=save_intermediate,
                                                        settings=modules_config['prosody'],
                                                        force_compute=force_compute_all)
            if 'anonymizer' in modules_config['prosody']:
                self.prosody_anonymization = ProsodyAnonymization(save_intermediate=save_intermediate,
                                                                  settings=modules_config['prosody'],
                                                                  force_compute=force_compute_all)
            else:
                self.prosody_anonymization = None
        else:
            self.prosody_extraction = None

        # TTS component
        self.speech_synthesis = SpeechSynthesis(devices=[devices[0]], settings=modules_config['tts'],
                                                model_dir=model_dir, save_output=config.get('save_output', True),
                                                force_compute=force_compute_all)

    def run_anonymization_pipeline(self, datasets, prepare_results=True):
        anon_wav_scps = {}

        for i, (dataset_name, dataset_path) in enumerate(datasets.items()):
            print(f'{i + 1}/{len(datasets)}: Processing {dataset_name}...')
            # Step 1: Recognize speech, extract speaker embeddings, extract prosody
            start_time = time.time()
            texts = self.speech_recognition.recognize_speech(dataset_path=dataset_path, dataset_name=dataset_name)
            print("--- Speech recognition time: %f min ---" % (float(time.time() - start_time) / 60))

            start_time = time.time()
            spk_embeddings = self.speaker_extraction.extract_speakers(dataset_path=dataset_path,
                                                                      dataset_name=dataset_name)
            print("--- Speaker extraction time: %f min ---" % (float(time.time() - start_time) / 60))

            if self.prosody_extraction:
                start_time = time.time()
                prosody = self.prosody_extraction.extract_prosody(dataset_path=dataset_path, dataset_name=dataset_name,
                                                                  texts=texts)
                print("--- Prosody extraction time: %f min ---" % (float(time.time() - start_time) / 60))
            else:
                prosody = None

            # Step 2: Anonymize speaker, change prosody
            if self.speaker_anonymization:
                start_time = time.time()
                anon_embeddings = self.speaker_anonymization.anonymize_embeddings(speaker_embeddings=spk_embeddings,
                                                                                  dataset_name=dataset_name)
                print("--- Speaker anonymization time: %f min ---" % (float(time.time() - start_time) / 60))
            else:
                anon_embeddings = spk_embeddings

            if self.prosody_anonymization:
                start_time = time.time()
                anon_prosody = self.prosody_anonymization.anonymize_prosody(prosody=prosody)
                print("--- Prosody anonymization time: %f min ---" % (float(time.time() - start_time) / 60))
            else:
                anon_prosody = prosody

            # Step 3: Synthesize
            start_time = time.time()
            wav_scp = self.speech_synthesis.synthesize_speech(dataset_name=dataset_name, texts=texts,
                                                              speaker_embeddings=anon_embeddings,
                                                              prosody=anon_prosody, emb_level=anon_embeddings.emb_level)
            anon_wav_scps[dataset_name] = wav_scp
            print("--- Synthesis time: %f min ---" % (float(time.time() - start_time) / 60))
            print('Done')

        if prepare_results:
            if self.speaker_anonymization:
                anon_vectors_path = self.speaker_anonymization.results_dir,
                anon_suffix = '_anon'
            else:
                anon_vectors_path = self.speaker_extraction.results_dir
                anon_suffix = '_res'
            now = datetime.strftime(datetime.today(), '%d-%m-%y_%H:%M')
            prepare_evaluation_data(dataset_dict=datasets, anon_wav_scps=anon_wav_scps,
                                    anon_vectors_path=anon_vectors_path, anon_suffix=anon_suffix,
                                    output_path=self.results_dir / 'formatted_data' / now)
            save_yaml(self.config, self.results_dir / 'formatted_data' / now / 'config.yaml')

            print("--- Total computation time: %f min ---" % (float(time.time() - self.total_start_time) / 60))

        return anon_wav_scps
