from pathlib import Path

from .anonymization.base_anon import BaseAnonymizer
from .speaker_embeddings import SpeakerEmbeddings


class SpeakerAnonymization:

    def __init__(self, vectors_dir, device, settings, results_dir=None, save_intermediate=True, force_compute=False):
        self.vectors_dir = vectors_dir
        self.device = device
        self.save_intermediate = save_intermediate
        self.force_compute = force_compute if force_compute else settings.get('force_compute_anonymization', False)

        self.vec_type = settings['vec_type']
        self.emb_level = settings['emb_level']

        if results_dir:
            self.results_dir = results_dir
        elif 'anon_results_path' in settings:
            self.results_dir = settings['anon_results_path']
        elif 'results_dir' in settings:
            self.results_dir = settings['results_dir']
        else:
            if self.save_intermediate:
                raise ValueError('Results dir must be specified in parameters or settings!')

        self.anonymizer = self._load_anonymizer(settings)
    
    @property
    def suffix(self):
        return self.anonymizer.suffix

    def anonymize_embeddings(self, speaker_embeddings, dataset_name):
        dataset_results_dir = self.results_dir / dataset_name if self.save_intermediate else ''

        if dataset_results_dir.exists() and any(dataset_results_dir.iterdir()) and not speaker_embeddings.new and not\
                self.force_compute:
            # if there are already anonymized speaker embeddings from this model and the computation is not forced,
            # simply load them
            print('No computation of anonymized embeddings necessary; load existing anonymized speaker embeddings '
                  'instead...')
            anon_embeddings = SpeakerEmbeddings(vec_type=self.vec_type, emb_level=self.emb_level, device=self.device)
            anon_embeddings.load_vectors(dataset_results_dir)
            return anon_embeddings
        else:
            # otherwise, create new anonymized speaker embeddings
            print('Anonymize speaker embeddings...')
            anon_embeddings = self.anonymizer.anonymize_embeddings(speaker_embeddings, emb_level=self.emb_level)

            if self.save_intermediate:
                anon_embeddings.save_vectors(dataset_results_dir)
            return anon_embeddings

    def _load_anonymizer(self, settings: dict):
        anon_method = settings['anon_method'] #HyperPyYAML already does the loading
        assert isinstance(anon_method, BaseAnonymizer), \
            'The anonymizer must be an instance of BaseAnonymizer, or a ' \
            f'subclass of it, but received an instance of {type(anon_method)}'
            
        print(f'Model type of anonymizer: {type(anon_method).__name__}')
        return anon_method
