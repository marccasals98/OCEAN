import os
import numpy as np
import pandas as pd
from  scipy.io import wavfile 
from scipy import signal
from tqdm import tqdm
from datetime import datetime
import argparse
import random
import SpectralSubstraction as ss

def main(params):
    input_path = params.input_path
    output_path = params.output_path
    spectral_subtraction_prob = params.spectral_subtraction_prob

    if not os.path.exists(output_path):
        os.mkdir(output_path) # create output dir

    new_srate = 250 # we downsample to 250Hz

    for filename in tqdm(os.listdir(input_path)):

        wav_path = os.path.join(input_path, filename)
        sample_rate, raw_sig = wavfile.read(wav_path) # open file

        #RESAMPLING:
        resampling_factor = new_srate / sample_rate
        samples = int(resampling_factor * len(raw_sig))
        resampled_sig = signal.resample(raw_sig, samples).astype(np.int16)

        # removing noise if spectral_subtraction is set
        window_sec = 0.256 # window used for spectral subtraction
        window_samples = window_sec * new_srate


        if spectral_subtraction_prob >= (random.randint(0,10)/10):
            cleaned_sig = ss.reduce_noise(resampled_sig, resampled_sig, winsize=2**8, add_noisy_phase=True)
        else:
            cleaned_sig = raw_sig
        wavfile.write(os.path.join(output_path, filename), sample_rate, cleaned_sig)
    

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Applies spectral subtraction to the dataset in the input path and outputs a preprocessed dataset in output path.')

    parser.add_argument('input_path', type=str, help='path of the input directory')
    parser.add_argument('output_path', type=str, help='path of the output directory')
    parser.add_argument('--spectral_subtraction_prob', type=float, default=0.0, help='Probability on which Spectral Subtraction is applied to each audio')


    params=parser.parse_args()

    main(params)