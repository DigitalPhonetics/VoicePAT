from pathlib import Path
from datetime import datetime
import logging

from anonymization.modules import (
    SpeechRecognition,
    SpeechSynthesis,
    ProsodyExtraction,
    ProsodyAnonymization,
    SpeakerExtraction,
    SpeakerAnonymization,
)
import typing
from utils import prepare_evaluation_data, save_yaml

logger = logging.getLogger(__name__)

class STTTSPipeline:
    def __init__(self, config: dict, force_compute: bool, devices: list):
        """
        Instantiates a STTTSPipeline with the complete feature extraction,
        modification and resynthesis.

        This pipeline consists of:
              - ASR -> phone sequence                    -
        input - (prosody extr. -> prosody anon.)         - TTS -> output
              - speaker embedding extr. -> speaker anon. -

        Args:
            config (dict): a configuration dictionary, e.g., see anon_ims_sttts_pc.yaml
            force_compute (bool): if True, forces re-computation of 
                all steps. otherwise uses saved results.
            devices (list): a list of torch-interpretable devices
        """
        self.config = config
        model_dir = Path(config.get("models_dir", "models"))
        vectors_dir = Path(config.get("vectors_dir", "original_speaker_embeddings"))
        self.results_dir = Path(config.get("results_dir", "results"))
        self.data_dir = Path(config["data_dir"]) if "data_dir" in config else None
        save_intermediate = config.get("save_intermediate", True)

        modules_config = config["modules"]

        # ASR component
        self.speech_recognition = SpeechRecognition(
            devices=devices,
            save_intermediate=save_intermediate,
            settings=modules_config["asr"],
            force_compute=force_compute,
        )

        # Speaker component
        self.speaker_extraction = SpeakerExtraction(
            devices=devices,
            save_intermediate=save_intermediate,
            settings=modules_config["speaker_embeddings"],
            force_compute=force_compute,
        )
        self.speaker_anonymization = SpeakerAnonymization(
            vectors_dir=vectors_dir,
            device=devices[0],
            save_intermediate=save_intermediate,
            settings=modules_config["speaker_embeddings"],
            force_compute=force_compute,
        )

        # Prosody component
        if "prosody" in modules_config:
            self.prosody_extraction = ProsodyExtraction(
                device=devices[0],
                save_intermediate=save_intermediate,
                settings=modules_config["prosody"],
                force_compute=force_compute,
            )
            if "anonymizer" in modules_config["prosody"]:
                self.prosody_anonymization = ProsodyAnonymization(
                    save_intermediate=save_intermediate,
                    settings=modules_config["prosody"],
                    force_compute=force_compute,
                )
            else:
                self.prosody_anonymization = None
        else:
            self.prosody_extraction = None

        # TTS component
        self.speech_synthesis = SpeechSynthesis(
            devices=devices,
            settings=modules_config["tts"],
            model_dir=model_dir,
            save_output=config.get("save_output", True),
            force_compute=force_compute,
        )

    def run_anonymization_pipeline(
        self, 
        datasets: typing.Dict[str, Path], 
        prepare_results: bool = True,
    ):
        """
            Runs the anonymization algorithm on the given datasets. Optionally
            prepares the results such that the evaluation pipeline 
            can interpret them.

            Args:
                datasets (dict of str -> Path): The datasets on which the 
                    anonymization pipeline should be runned on. These dataset 
                    will be processed sequentially.
                prepare_results (bool): if True, the resulting anonymization 
                    .wavs are prepared for evaluation
        """
        anon_wav_scps = {}

        for i, (dataset_name, dataset_path) in enumerate(datasets.items()):
            logger.info(f"{i + 1}/{len(datasets)}: Processing {dataset_name}...")
            # Step 1: Recognize speech, extract speaker embeddings, extract prosody
            texts = self.speech_recognition.recognize_speech(
                dataset_path=dataset_path, dataset_name=dataset_name
            )
            spk_embeddings = self.speaker_extraction.extract_speakers(
                dataset_path=dataset_path, dataset_name=dataset_name
            )

            if self.prosody_extraction:
                prosody = self.prosody_extraction.extract_prosody(
                    dataset_path=dataset_path, dataset_name=dataset_name, texts=texts
                )
            else:
                prosody = None

            # Step 2: Anonymize speaker, change prosody
            anon_embeddings = self.speaker_anonymization.anonymize_embeddings(
                speaker_embeddings=spk_embeddings, dataset_name=dataset_name
            )

            if self.prosody_anonymization:
                anon_prosody = self.prosody_anonymization.anonymize_prosody(
                    prosody=prosody
                )
            else:
                anon_prosody = prosody

            # Step 3: Synthesize
            wav_scp = self.speech_synthesis.synthesize_speech(
                dataset_name=dataset_name,
                texts=texts,
                speaker_embeddings=anon_embeddings,
                prosody=anon_prosody,
                emb_level=anon_embeddings.emb_level,
            )
            anon_wav_scps[dataset_name] = wav_scp
            logger.info("Anonymization pipeline completed.")

        if prepare_results:
            logger.info("Preparing results according to the Kaldi format.")
            if self.speaker_anonymization:
                anon_vectors_path = self.speaker_anonymization.results_dir
            else:
                anon_vectors_path = self.speaker_extraction.results_dir
            now = datetime.strftime(datetime.today(), "%d-%m-%y_%H:%M")
            prepare_evaluation_data(
                dataset_dict=datasets,
                anon_wav_scps=anon_wav_scps,
                anon_vectors_path=anon_vectors_path,
                anon_suffix=self.speaker_anonymization.suffix,
                output_path=self.results_dir / "formatted_data" / now,
            )
            save_yaml(
                self.config, self.results_dir / "formatted_data" / now / "config.yaml"
            )

        return anon_wav_scps
