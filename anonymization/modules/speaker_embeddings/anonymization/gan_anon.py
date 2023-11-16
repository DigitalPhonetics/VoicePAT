import torch
import numpy as np
from scipy.spatial.distance import cosine
from tqdm import tqdm

from .base_anon import BaseAnonymizer
from ..speaker_embeddings import SpeakerEmbeddings
from .utils.WGAN import EmbeddingsGenerator


class GANAnonymizer(BaseAnonymizer):

    def __init__(self, vec_type='xvector', device=None, model_name=None, vectors_file=None, sim_threshold=0.7,
                 gan_model_path=None, num_sampled=1000, save_intermediate=False, **kwargs):
        super().__init__(vec_type=vec_type, device=device)

        self.model_name = model_name if model_name else f'gan_{vec_type}'
        self.vectors_file = vectors_file
        self.unused_indices_file = self.vectors_file.with_name(f'unused_indices_{self.vectors_file.name}')
        self.sim_threshold = sim_threshold
        self.save_intermediate = save_intermediate
        self.n = num_sampled

        if self.vectors_file.is_file():
            self.gan_vectors = torch.load(self.vectors_file, map_location=self.device)
            if self.unused_indices_file.is_file():
                self.unused_indices = torch.load(self.unused_indices_file, map_location='cpu')
            else:
                self.unused_indices = np.arange(len(self.gan_vectors))
        else:
            self.gan_vectors, self.unused_indices = self._generate_artificial_embeddings(gan_model_path, self.n)

    def anonymize_embeddings(self, speaker_embeddings, emb_level='spk'):
        if emb_level == 'spk':
            print(f'Anonymize embeddings of {len(speaker_embeddings)} speakers...')
        elif emb_level == 'utt':
            print(f'Anonymize embeddings of {len(speaker_embeddings)} utterances...')

        identifiers = []
        speakers = []
        anon_vectors = []
        genders = []
        for i in tqdm(range(len(speaker_embeddings))):
            identifier, vector = speaker_embeddings[i]
            speaker = speaker_embeddings.original_speakers[i]
            gender = speaker_embeddings.genders[i]
            anon_vec = self._select_gan_vector(spk_vec=vector)
            identifiers.append(identifier)
            speakers.append(speaker)
            anon_vectors.append(anon_vec)
            genders.append(gender)

        anon_embeddings = SpeakerEmbeddings(vec_type=self.vec_type, device=self.device, emb_level=emb_level)
        anon_embeddings.set_vectors(identifiers=identifiers, vectors=torch.stack(anon_vectors, dim=0),
                                    speakers=speakers, genders=genders)
        if self.save_intermediate:
            torch.save(self.unused_indices, self.unused_indices_file)

        return anon_embeddings

    def _generate_artificial_embeddings(self, gan_model_path, n):
        print(f'Generate {n} artificial speaker embeddings...')
        generator = EmbeddingsGenerator(gan_path=gan_model_path, device=self.device)
        gan_vectors = generator.generate_embeddings(n=n)
        unused_indices = np.arange(len(gan_vectors))

        if self.save_intermediate:
            torch.save(gan_vectors, self.vectors_file)
            torch.save(unused_indices, self.unused_indices_file)
        return gan_vectors, unused_indices

    def _select_gan_vector(self, spk_vec):
        i = 0
        limit = 20
        while i < limit:
            idx = np.random.choice(self.unused_indices)
            anon_vec = self.gan_vectors[idx]
            sim = 1 - cosine(spk_vec.cpu().numpy(), anon_vec.cpu().numpy())
            if sim < self.sim_threshold:
                break
            i += 1
        self.unused_indices = self.unused_indices[self.unused_indices != idx]
        return anon_vec
