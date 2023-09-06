from pathlib import Path
import torch


class BaseAnonymizer:

    def __init__(self, vec_type='xvector', device=None, **kwargs):
        # Base class for speaker embedding anonymization.
        self.vec_type = vec_type

        if isinstance(device, torch.device):
            self.device = device
        elif isinstance(device, str):
            self.device = torch.device(device)
        elif isinstance(device, int):
            self.device = torch.device(f'cuda:{device}')
        else:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def anonymize_embeddings(self, speaker_embeddings, emb_level='spk'):
        # Template method for anonymizing a dataset. Not implemented.
        raise NotImplementedError('anonymize_data')
