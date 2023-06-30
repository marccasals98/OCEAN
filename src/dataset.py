import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms as transforms
import pickle
from random import randint
import numpy as np
import torch.nn.functional as F



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
        base_df = pd.read_pickle(pickle_path)
        self.df = base_df[base_df['species'].isin(config['species'])].reset_index(drop=True)
        print(len(self.df))
        #print(self.df.head(n=50))
        self.img_dir = img_dir 
        self.transform = transform
        self.config = config
        self.num_classes = self.df.species.nunique()


    def __len__(self):
        return len(self.df.species)
        
    @staticmethod
    def seconds_to_frames(image_dict: dict, parameters: dict) -> int:
        '''
        Calculate the number of frames from the spectrogram we need to crop.  
        '''
        # Pass from the dictionary with all the info to the image and settings:
        image = image_dict['features']
        image_settings = image_dict['settings']

        # Get the parameters: 
        file_frames = image.shape[0]
        sampling_rate = image_settings.sampling_rate
        hop_length = int(image_settings.hop_length_secs * sampling_rate)
        n_fft = int(image_settings.n_fft_secs * sampling_rate)

        # Get estimation crop size in frames:
        estimated_samples = (file_frames - 1) * hop_length + n_fft
        estimated_audio_length_secs = estimated_samples / sampling_rate
        estimated_frames_1_sec = file_frames / estimated_audio_length_secs
        random_crop_frames = int(parameters['random_crop_secs'] * estimated_frames_1_sec)
        # print('el valor es ', random_crop_frames)
        return random_crop_frames



    @staticmethod
    def sample_spectrogram_crop(image: np.ndarray, parameters: dict, random_crop_frames: int) -> np.ndarray:
        '''
        This function takes an image and returns a random crop of the image.

        As we are working with spectrograms and we need all images to be the same size (to work with batches), 
        we will need to crop all these spectrograms equally to feed the network.
        '''
        # Cut the image with a fixed length a random place.
        file_frames = image.shape[0]
        # Check if we need padding:
        if file_frames < random_crop_frames:
                padding_frames = random_crop_frames - file_frames
                image = np.pad(image, pad_width=((0, padding_frames), (0, 0)), mode = 'constant')
                file_frames = image.shape[0]
        # Random start index:
        index = randint(0, max(0, file_frames - random_crop_frames - 1))
        end_index = np.array(range(min(file_frames, int(random_crop_frames)))) + index
        # slice the image
        features = image[end_index, :]
        return features

    @staticmethod
    def get_feature_vector(image_dict: dict, parameters: dict) -> np.ndarray:
        '''
        This function takes an image and returns a feature vector. 
        The feature vector is a slice of the image.
        '''
        ima_trans = np.transpose(image_dict['features'])
        ima_norm = (ima_trans-ima_trans.min())/(ima_trans.max()-ima_trans.min())
        random_crop_frames = BlueFinLib.seconds_to_frames(image_dict, parameters)
        features = BlueFinLib.sample_spectrogram_crop(ima_norm, parameters, random_crop_frames)
        return features

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir,
                                self.df['subdataset'][index]+'_'+
                                self.df['wav_name'][index]+'_'+
                                self.df['species'][index]+'_'+
                                self.df['vocalization'][index]+'_'+
                                self.df['date'][index]+'_'+
                                self.df['begin_sample'][index]+'_'+
                                self.df['end_sample'][index]+'_'+
                                self.df['sampling_rate'][index]+'Hz'+'.pickle'
                                )
        label = self.df['num_species'][index]
        one_hot = F.one_hot(torch.tensor(label), self.num_classes).float()
        try:
                with open(img_path, 'rb') as f:
                        image_dict = pickle.load(f)
                parameters = self.config
                # Slice the spectrogram to have all the same length:
                features = BlueFinLib.get_feature_vector(image_dict, parameters)
                if self.transform:
                        features = self.transform(features)
                return features, one_hot

        except FileNotFoundError:
                print(f"File {img_path} not found.")
    


