import numpy as np
import torch
import torchaudio
import pyloudnorm as pyln


def normalize_wave(wave, sr, device):
    # adapted from IMSToucan/Preprocessing/AudioPreprocessor
    dur = wave.shape[1] / sr
    wave = wave.squeeze().cpu().numpy()

    # normalize loudness
    try:
        meter = pyln.Meter(sr, block_size=min(dur - 0.0001, abs(dur - 0.1)) if dur < 0.4 else 0.4)
        loudness = meter.integrated_loudness(wave)
        loud_normed = pyln.normalize.loudness(wave, loudness, -30.0)
        peak = np.amax(np.abs(loud_normed))
        norm_wave = np.divide(loud_normed, peak)
    except ZeroDivisionError:
        norm_wave = wave

    wave = torch.Tensor(norm_wave).to(device)

    if sr != 16000:
        wave = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000).to(device)(wave)

    return wave
