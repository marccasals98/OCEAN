import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image
from random import randint
import numpy as np



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
    def __init__(self, pickle_path: str, img_dir: str, config: dict, transform=None):
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
        self.transform = transform
        self.config = config


    def __len__(self):
        return len(self.species)
    
    @staticmethod
    def sample_spectrogram_crop(image, parameters):
        '''
        This function takes an image and returns a random crop of the image.
        '''
        # Cut the image with a fixed length a random place.
        file_frames = image.shape[0]
        # get a random start index
        index = randint(0, max(0, file_frames - parameters['random_crop_frames'] - 1))
        # get the end index
        end_index = np.array(range(min(file_frames, int(parameters['random_crop_frames'])))) + index
        # slice the image
        features = image[end_index, :]
        return features

    @staticmethod
    def get_feature_vector(image, parameters):
        '''
        This function takes an image and returns a feature vector. 
        The feature vector is a slice of the image.
        '''
        ima_trans = np.transpose(image)
        ima_norm = (ima_trans-np.min(ima_trans))/(np.max(ima_trans)-np.min(ima_trans))
        features = BlueFinLib.sample_spectrogram_crop(ima_norm, parameters)
        return features

    def __getitem__(self, index):
        # TODO: canviar el nom del fitxer pel que sera el complet.
        # TODO: el espectograma es guarda com a pickle.
        img_path = os.path.join(self.img_dir, self.wav_name[index]+'.png') 
        label = self.species[index]
        try:
                with open(img_path, 'rb') as f:
                        image = read_image(f)
        except FileNotFoundError:
                print(f"File {img_path} not found.")
        # TODO: slice the histograms.
        parameters = self.config
        features = BlueFinLib.get_feature_vector(image, parameters)
        if self.transform:
            features = self.transform(features)
        return features, label


