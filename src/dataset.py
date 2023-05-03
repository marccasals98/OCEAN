import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from torchvision.io import read_image


class BlueFinLib(Dataset):
    def __init__(self, pickle_path: str, img_dir: str, transform=None):
        super().__init__() # is this necessary?
        
        df = pd.read_pickle(pickle_path)
        self.species = list(df.species_labels)
        self.wav_name = list(df['original name of wav file']) 
        self.img_dir = img_dir 
        #self.specie = df[df.iloc[:,1]==self.wav_name].species.values[0]
        self.transform = transform


    def __len__(self):
        return len(self.species_labels)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.wav_name[index]+'.png') 
        label = self.species_labels[index]
        image = read_image(img_path)
        if self.transform:
            image = self.transform(image)
        return image, label


