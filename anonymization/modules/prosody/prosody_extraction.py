import torch
torch.set_num_threads(1)

from tqdm import tqdm
from pathlib import Path

from .prosody import Prosody
from .extraction import *
from utils import read_kaldi_format


class ProsodyExtraction:

    def __init__(self, device, settings, results_dir=None, save_intermediate=True, force_compute=False):
        self.device = device
        self.save_intermediate = save_intermediate
        self.force_compute = force_compute if force_compute else settings.get('force_compute_extraction', False)
        extractor_type = settings.get('extractor_type', 'ims')

        if results_dir:
            self.results_dir = results_dir
        elif 'extraction_results_path' in settings:
            self.results_dir = settings['extraction_results_path']
        elif 'results_dir' in settings:
            self.results_dir = settings['results_dir']
        else:
            if self.save_intermediate:
                raise ValueError('Results dir must be specified in parameters or settings!')

        if extractor_type == 'ims':
            self.aligner_path = settings.get('aligner_model_path')
            self.on_line_fine_tune = settings.get('on_line_fine_tune', True)
            self.extractor = ImsProsodyExtractor(aligner_path=self.aligner_path, device=self.device,
                                                 on_line_fine_tune=self.on_line_fine_tune)

    def extract_prosody(self, dataset_path: Path, texts, dataset_name=None):
        dataset_name = dataset_name if dataset_name else dataset_path.name
        dataset_results_dir = self.results_dir / dataset_name if self.save_intermediate else Path('')
        wav_scp = read_kaldi_format(dataset_path / 'wav.scp')

        data_prosody = Prosody()
        text_is_phones = texts.is_phones

        if (dataset_results_dir / 'utterances').exists() and not self.force_compute:
            data_prosody.load_prosody(dataset_results_dir)
            unprocessed_utts = wav_scp.keys() - data_prosody.utterances.keys()
            wav_scp = {utt: wav_scp[utt] for utt in unprocessed_utts}

        if wav_scp:
            print(f'Extract prosody for {len(wav_scp)} of {len(wav_scp) + len(data_prosody)} utterances')
            data_prosody.new = True
            i = 0
            for utt, wav_path in tqdm(wav_scp.items()):
                text = texts[utt]
                try:
                    utt_prosody = self.extractor.extract_prosody(transcript=text, ref_audio_path=wav_path,
                                                                 input_is_phones=text_is_phones)
                except IndexError:
                    print(f'Index Error for {utt}')
                    continue
                duration, pitch, energy, start_silence, end_silence = utt_prosody
                data_prosody.add_instance(utterance=utt, duration=duration, pitch=pitch, energy=energy,
                                          start_silence=start_silence, end_silence=end_silence)
                i += 1
                if self.save_intermediate and i > 0 and i % 100 == 0:
                    data_prosody.save_prosody(dataset_results_dir)

            if self.save_intermediate:
                data_prosody.save_prosody(dataset_results_dir)

        elif len(data_prosody.utterances) > 0:
            print('No prosody extraction necessary; load stored values instead...')
        else:
            print(f'No utterances could be found in {dataset_path}!')

        return data_prosody
