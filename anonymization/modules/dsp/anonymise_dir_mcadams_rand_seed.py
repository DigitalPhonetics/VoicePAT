#!/usr/bin/env python3.0
# -*- coding: utf-8 -*-
"""
@author: Jose Patino, Massimiliano Todisco, Pramod Bachhav, Nicholas Evans
Audio Security and Privacy Group, EURECOM
modified version (N.T.)
"""
import os
import librosa
import numpy as np
import scipy
import wave
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import random
from kaldiio import ReadHelper
import shutil

def load_utt2spk(path):
    assert os.path.isfile(path), f'File does not exist {path}'
    table = np.genfromtxt(path, dtype='U')
    utt2spk = {utt: spk for utt, spk in table}
    return utt2spk

def process_data(dataset_path, anon_level, settings):

    utt2spk = None
    if anon_level == 'spk':
        utt2spk = load_utt2spk( dataset_path / 'utt2spk')

    output_path = Path(str(dataset_path) + settings['anon_suffix'])
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    shutil.copytree(dataset_path, output_path)
    if not os.path.exists(output_path /  'wav'):
        os.makedirs(output_path / 'wav')
    wav_scp = dataset_path / 'wav.scp'
    path_wav_scp_out = output_path / 'wav.scp'
    with open(path_wav_scp_out, 'wt', encoding='utf-8') as writer:
        with ReadHelper(f'scp:{wav_scp}') as reader:
            #print(reader)
            for utid, (freq, samples) in reader:
                #print(utid)
                output_file = os.path.join(output_path / 'wav', f'{utid}.wav')
                print(f'Generating {output_file}')
                if os.path.exists(output_file):
                    print('file already exists')
                    continue
                samples = samples / (np.iinfo(np.int16).max + 1)
                if anon_level == 'spk':
                    assert utid in utt2spk, f'Failed to find speaker ID for utterance {utid}'
                    spid = utt2spk[utid]
                    random.seed(np.abs(hash(spid)))
                rand_mc_coeff = random.uniform(settings['mc_coeff_min'], settings['mc_coeff_max'])
               
                samples = anonym(freq=freq, samples=samples, 
                    winLengthinms=settings['winLengthinms'],
                    shiftLengthinms=settings['shiftLengthinms'], 
                    lp_order=settings['n_coeffs'], mcadams=rand_mc_coeff)

                with wave.open(output_file, 'wb') as stream:
                    stream.setframerate(freq)
                    stream.setnchannels(1)
                    stream.setsampwidth(2)
                    stream.writeframes(samples)
                print(f'{utid} {output_file}', file=writer)
    print('Done')

def anonym(freq, samples, winLengthinms=20, shiftLengthinms=10, lp_order=20, mcadams=0.8):


    #print(mcadams)
    eps = np.finfo(np.float32).eps
    samples = samples + eps
    
    # simulation parameters
    winlen = np.floor(winLengthinms * 0.001 * freq).astype(int)
    shift = np.floor(shiftLengthinms * 0.001 * freq).astype(int)
    length_sig = len(samples)
    
    # fft processing parameters
    NFFT = 2 ** (np.ceil((np.log2(winlen)))).astype(int)
    # anaysis and synth window which satisfies the constraint
    wPR = np.hanning(winlen)
    K = np.sum(wPR) / shift
    win = np.sqrt(wPR / K)
    Nframes = 1 + np.floor((length_sig - winlen) / shift).astype(int) # nr of complete frames   
    
    # carry out the overlap - add FFT processing
    sig_rec = np.zeros([length_sig]) # allocate output+'ringing' vector
    
    for m in np.arange(1, Nframes):
        # indices of the mth frame
        index = np.arange(m * shift, np.minimum(m * shift + winlen, length_sig))    
        # windowed mth frame (other than rectangular window)
        frame = samples[index] * win 
        # get lpc coefficients
        a_lpc = librosa.core.lpc(frame + eps, order=lp_order)
        # get poles
        poles = scipy.signal.tf2zpk(np.array([1]), a_lpc)[1]
        #index of imaginary poles
        ind_imag = np.where(np.isreal(poles) == False)[0]
        #index of first imaginary poles
        ind_imag_con = ind_imag[np.arange(0, np.size(ind_imag), 2)]
        
        # here we define the new angles of the poles, shifted accordingly to the mcadams coefficient
        # values >1 expand the spectrum, while values <1 constract it for angles>1
        # values >1 constract the spectrum, while values <1 expand it for angles<1
        # the choice of this value is strongly linked to the number of lpc coefficients
        # a bigger lpc coefficients number constraints the effect of the coefficient to very small variations
        # a smaller lpc coefficients number allows for a bigger flexibility
        new_angles = np.angle(poles[ind_imag_con]) ** mcadams
        #new_angles = np.angle(poles[ind_imag_con])**path[m]
        
        # make sure new angles stay between 0 and pi
        new_angles[np.where(new_angles >= np.pi)] = np.pi        
        new_angles[np.where(new_angles <= 0)] = 0  
        
        # copy of the original poles to be adjusted with the new angles
        new_poles = poles
        for k in np.arange(np.size(ind_imag_con)):
            # compute new poles with the same magnitued and new angles
            new_poles[ind_imag_con[k]] = np.abs(poles[ind_imag_con[k]]) * np.exp(1j * new_angles[k])
            # applied also to the conjugate pole
            new_poles[ind_imag_con[k] + 1] = np.abs(poles[ind_imag_con[k] + 1]) * np.exp(-1j * new_angles[k])            
        
        # recover new, modified lpc coefficients
        a_lpc_new = np.real(np.poly(new_poles))
        # get residual excitation for reconstruction
        res = scipy.signal.lfilter(a_lpc,np.array(1),frame)
        # reconstruct frames with new lpc coefficient
        frame_rec = scipy.signal.lfilter(np.array([1]),a_lpc_new,res)
        frame_rec = frame_rec * win    

        outindex = np.arange(m * shift, m * shift + len(frame_rec))
        # overlap add
        sig_rec[outindex] = sig_rec[outindex] + frame_rec
    sig_rec = (sig_rec / np.max(np.abs(sig_rec)) * (np.iinfo(np.int16).max - 1)).astype(np.int16)
    return sig_rec
    #scipy.io.wavfile.write(output_file, freq, np.float32(sig_rec))
    #awk -F'[/.]' '{print $5 " sox " $0 " -t wav -R -b 16 - |"}' > data/$dset$anon_data_suffix/wav.scp
