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
    noise_path = params.noise_path
    spectral_subtraction_prob = params.spectral_subtraction_prob

    if not os.path.exists(output_path):
        os.mkdir(output_path) # create output dir

    new_srate = 250 # we downsample to 250Hz

    total_subtractions = 0 # We set a counter to know how many correct subtractions we have done.

    for filename in tqdm(os.listdir(input_path)):

        wav_path = os.path.join(input_path, filename)
        sample_rate, raw_sig = wavfile.read(wav_path) # open file

        # removing noise if spectral_subtraction is set
        window_sec = 0.256 # window used for spectral subtraction
        window_samples = window_sec * new_srate


        if spectral_subtraction_prob >= (random.randint(0,10)/10):

            try:
                wav_noise_path = os.path.join(noise_path, filename) # Load noise estimation of the signal:
                sample_rate, raw_noise_sig = wavfile.read(wav_noise_path) # open noise estimator.
                # Spectral subtraction with noise estimation.
                if len(raw_noise_sig) != 0:
                    total_subtractions += 1 # increase the counter by one.
                    cleaned_sig = ss.reduce_noise(raw_sig, raw_noise_sig, winsize=window_samples, add_noisy_phase=True)
                else:
                    cleaned_sig = raw_sig    
            except FileNotFoundError:
                cleaned_sig = raw_sig
        else:
            cleaned_sig = raw_sig
        wavfile.write(os.path.join(output_path, 'Spectral_Subtraction_' +filename), sample_rate, cleaned_sig)
    
    print("The number of correct spectral subtractions that we have done is: ", total_subtractions)
    

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Applies spectral subtraction to the dataset in the input path and outputs a preprocessed dataset in output path.')

    parser.add_argument('input_path', type=str, help='path of the input directory')
    parser.add_argument('output_path', type=str, help='path of the output directory')
    parser.add_argument('noise_path', type=str, help='path of the noise samples for noise estimation')
    parser.add_argument('--spectral_subtraction_prob', type=float, default=0.0, help='Probability on which Spectral Subtraction is applied to each audio')


    params=parser.parse_args()

    main(params)