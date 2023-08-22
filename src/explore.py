import main
import torch
from torch.utils.data import DataLoader
from dataset import BlueFinLib
from ResNet import ResNet50, ResNet101, ResNet152
import torch.optim as optim
from torchvision import transforms
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import wandb
from DataframeCreator import DataframeCreator
import os
from LeNet import LeNet5
import matplotlib.pyplot as plt

config = {
    "architecture": "ResNet50",
    "lr": 1e-3,
    "batch_size": 64, # This number must be bigger than one (nn.BatchNorm).
    "epochs": 25,
    "num_samples_train": 0.6,
    "num_samples_val": 0.2,
    "num_samples_test": 0.2,
    "species": ['Fin', 'Blue'],
    "random_crop_secs": 5, # number of seconds that has the spectrogram.
    "random_erasing": 0.5, # probability that the random erasing operation will be performed.
    "df_dir": "/home/usuaris/veussd/DATABASES/Ocean/dataframes", # where the pickle dataframe is stored.
    "df_path": "",
    "img_dir" : "/home/usuaris/veussd/DATABASES/Ocean/Spectrograms_AcousticTrends/23_06_02_09_07_26_aty1jmit_wise-meadow-57", # directory of the spectrograms.
    "save_dir": "/home/usuaris/veussd/DATABASES/Ocean/checkpoints/" # where we save the model checkpoints.
}
df_creator = DataframeCreator(config['img_dir'], config['df_dir'])
config["df_path"] = df_creator.get_df_path()

train_loader, val_loader, test_loader = main.data_loaders(config)



data_transforms = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(0.5, 0.5),
                                    transforms.RandomErasing(p=config["random_erasing"]),])

total_data = BlueFinLib(pickle_path = config['df_path'], 
                        img_dir = config['img_dir'], 
                        config = config,
                        transform=data_transforms)
for i in tqdm(range(10)):
    features, label = total_data.__getitem__(i)
    image_np = features.numpy()
    # Transpose the array to match the expected image format
    image_np = image_np.transpose(1, 2, 0)
    # Plot the image
    plt.imshow(image_np)
    plt.axis('off')  # Disable axis
    plt.savefig(f'/home/usuaris/veu/marc.casals/IMAGES/{i}.png')

