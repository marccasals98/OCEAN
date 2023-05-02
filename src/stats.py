import os
import pathlib
import librosa # for wav files
import soundfile as sf
from  scipy.io import wavfile 
import pickle
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.image as img

class Stats:

    def __init__(self, path: str):
        self.path = path
        ...
    
    def upload(self):
        '''
        Reads a pickle file that contains a pandas dataframe

        Arguments:
        ---------
        
        Returns:
        -------
        df : pandas
            a pandas dataframe with all of the info about the dataset.

        '''
        
        # read the pickle
        df = pd.read_pickle(self.path)

        # calculate the duration of the vocalizations.
        duration = (df.iloc[:,6]-df.iloc[:,5])/df.iloc[:,7]
        df['Duration'] = duration
        
        print(df["species"].value_counts(dropna = False)) # printing the counts.
        return df
    
    def species(df, output_path: str):
        '''
        Histogram of the different species.

        Arguments:
        ----------
        output_path : str
            The path where the figure will be saved.
        '''

        plt.hist(df.species)
        plt.title('Histogram of the different species')
        plt.ylabel('frequency')
        plt.xlabel('specie')
        plt.savefig(output_path)

    
    def vocalizations(df, output_path: str):
        '''
        Histogram of the different vocalizations.

        Arguments:
        ----------
        output_path : str
            The path where the figure will be saved.
        '''     

        fig, ax = plt.subplots(figsize=(15, 3))
        ax.hist(df.vocalization)
        plt.title('Histogram of the different vocalizations')
        plt.ylabel('frequency')
        plt.xlabel('vocalizations')
        plt.savefig(output_path)

    def duration_species(df, output_path: str):
        '''
        Histogram of the duration of the vocalizations taking for each specie.

        Arguments:
        ----------
        output_path : str
            The path where the figure will be saved.
        '''      

        grouped = df.groupby('species')['Duration']
        fig, ax = plt.subplots()
        
        for name, group in grouped:
            ax.hist(group, bins=100, range =(0,20), alpha=0.6, label=name)
        # Add labels and title
        plt.xlabel('Duration')
        plt.ylabel('Frequency')
        plt.title('Duration by Species')
        plt.legend()
        plt.savefig(output_path)
      
    def duration_vocal(df, output_path: str):
        '''
        Histogram of the duration of the vocalizations taking for each 
        vocalization.

        Arguments:
        ----------
        output_path : str
            The path where the figure will be saved.
        '''      

        grouped = df.groupby('vocalization')['Duration']
        fig, ax = plt.subplots()

        for name, group in grouped:
            ax.hist(group, bins=100, range =(0,20), alpha=0.6, label=name)
        plt.xlabel('Duration')
        plt.ylabel('Frequency')
        plt.title('Duration by Vocalization')
        plt.legend()
        plt.savefig(output_path)

    def compute_statistics(self, output_path: str):
        '''
        This function calls all the histogram functions and it is the main core
        of the class.

        Arguments:
        ----------
        output_path : str
            The path where the figure will be saved.
        '''
        df = self.upload()
        
        # Making the histograms:
        Stats.species(df = df, output_path = output_path + 'species.png')
        Stats.duration_species(df = df, output_path = output_path + 'duration_species.png')
        Stats.vocalizations(df = df, output_path = output_path + 'vocalizations.png')
        Stats.duration_vocal(df = df, output_path = output_path + 'duration_vocal.png')