from pathlib import Path

from .anonymization import PoolAnonymizer, RandomAnonymizer, GANAnonymizer
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

    def _load_anonymizer(self, settings):
        anon_method = settings['anon_method']
        vec_type = settings.get('vec_type', 'xvector')
        model_name = settings.get('anon_name', None)

        if anon_method == 'random':
            anon_settings = settings.get('random_anon_settings', {})
            model = RandomAnonymizer(vec_type=vec_type, device=self.device, model_name=model_name, **anon_settings)

        elif anon_method == 'pool':
            anon_settings = settings.get('pool_anon_settings', {})
            model = PoolAnonymizer(vec_type=vec_type, device=self.device, model_name=model_name,
                                   embed_model_dir=settings.get('embed_model_path', Path()),
                                   save_intermediate=self.save_intermediate, **anon_settings)

        elif anon_method == 'gan':
            anon_settings = settings.get('gan_anon_settings', {})
            model = GANAnonymizer(vec_type=vec_type, device=self.device, model_name=model_name,
                                  save_intermediate=self.save_intermediate, **anon_settings)
        else:
            raise ValueError(f'Unknown anonymization method {anon_method}')

        print(f'Model type of anonymizer: {model_name}')
        return model
