import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class BlueFinLib(Dataset):
    def __init__(self, species, wav_name, pickle_path, img_dir):
        super().__init__()
        df = pd.read_pickle(pickle_path)
        self.species = list(df.species_labels)
        self.wav_name = list(df['original name of wav file']) 
        self.img_dir = img_dir 
        #self.specie = df[df.iloc[:,1]==self.wav_name].species.values[0]


    def __len__(self):
        return len(self.species_labels)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.wav_name[index]) # arreglar: s'hauran de borrar les cometes 'audiofile.wav'
        label = species_labels[index]
        return image, label