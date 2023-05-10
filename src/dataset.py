import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image
import torch



class BlueFinLib(Dataset):
    '''
    The AcousticTrends_BlueFinLibrary dataset. 

    This class needs a support dataframe that have relevant information of each vocalization (sample). 
    Each vocalization have a particular wav_name that works as an id and it is classified in the specie 
    that corresponds. For this, we have listed IN ORDER, all of the species and wav files names.

    Attributes:
    -----------
    species : list
            Listing each vocalization's specie in order.
    wav_name : list
            Listing each vocalization's wav name, the name that is used in the dataframe.
    img_dir : str
            The path where all the spectrograms are saved.
    transform : sequence of transforms
            The transformations that are done to the tensor.
    
    '''
    def __init__(self, pickle_path: str, img_dir: str, transform=None):
        '''
        Arguments:
        ----------
        pickle_path : str
                We want to open a pickle in which is stored the pandas dataframe.
        img_dir : str 
                the directory where the images are.
        transform : sequence of transforms
                the transformations of the tensor.
        '''
        super().__init__() # is this necessary?
        df = pd.read_pickle(pickle_path)
        self.species = list(df.species)
        self.wav_name = list(df['original name of wav file']) 
        self.img_dir = img_dir 
        #self.specie = df[df.iloc[:,1]==self.wav_name].species.values[0]
        self.transform = transform


    def __len__(self):
        return len(self.species)
    

    def sample_spectrogram_crop():
         ...     

    def get_feature_vector():
         ...

    def __getitem__(self, index):
        # TODO: canviar el nom del fitxer pel que sera el complet.
        # TODO: el espectograma es guarda com a pickle.
        img_path = os.path.join(self.img_dir, self.wav_name[index]+'.png') 
        label = self.species[index]
        try:
                image = read_image(img_path)
        except FileNotFoundError:
                print(f"File {img_path} not found.")
        # TODO: slice the histograms.
        features = ...
        if self.transform:
            features = self.transform(features)
        return features, label


