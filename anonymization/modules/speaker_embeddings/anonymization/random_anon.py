import json
from pathlib import Path
import torch
import numpy as np

from .base_anon import BaseAnonymizer
from ..speaker_embeddings import SpeakerEmbeddings


class RandomAnonymizer(BaseAnonymizer):

    def __init__(self, vec_type='xvector', device=None, model_name=None,  in_scale=False, stats_per_dim_path=None,
                 **kwargs):
        super().__init__(vec_type=vec_type, device=device)

        self.model_name = model_name if model_name else f'random_{vec_type}'

        if in_scale:
            self.scaling_ranges = self._load_scaling_ranges(stats_per_dim_path=stats_per_dim_path)
        else:
            self.scaling_ranges = None

    def anonymize_embeddings(self, speaker_embeddings, emb_level='spk'):
        if self.scaling_ranges:
            print('Anonymize vectors in scale!')
            return self._anonymize_data_in_scale(speaker_embeddings)
        else:
            identifiers = []
            anon_vectors = []
            speakers = speaker_embeddings.original_speakers
            genders = speaker_embeddings.genders
            for identifier, vector in speaker_embeddings:
                mask = torch.zeros(vector.shape[0]).float().random_(-40, 40).to(self.device)
                anon_vec = vector * mask
                identifiers.append(identifier)
                anon_vectors.append(anon_vec)

            anon_embeddings = SpeakerEmbeddings(vec_type=self.vec_type, device=self.device)
            anon_embeddings.set_vectors(identifiers=identifiers, vectors=torch.stack(anon_vectors, dim=0),
                                        genders=genders, speakers=speakers)

            return anon_embeddings

    def _load_scaling_ranges(self, stats_per_dim_path):
        if stats_per_dim_path is None:
            stats_per_dim_path = Path('stats_per_dim.json')

        with open(stats_per_dim_path) as f:
            dim_ranges = json.load(f)
            return [(v['min'], v['max']) for k, v in sorted(dim_ranges.items(), key=lambda x: int(x[0]))]

    def _anonymize_data_in_scale(self, speaker_embeddings):
        identifiers = []
        anon_vectors = []
        speakers = speaker_embeddings.original_speakers
        genders = speaker_embeddings.genders

        for identifier, vector in speaker_embeddings:
            anon_vec = torch.tensor([np.random.uniform(*dim_range)
                                     for dim_range in self.scaling_ranges]).to(self.device)
            identifiers.append(identifier)
            anon_vectors.append(anon_vec)

        anon_embeddings = SpeakerEmbeddings(vec_type=self.vec_type, device=self.device)
        anon_embeddings.set_vectors(identifiers=identifiers, vectors=torch.stack(anon_vectors, dim=0), genders=genders,
                                    speakers=speakers)

        return anon_embeddings