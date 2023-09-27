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
import matplotlib.pyplot as plt
from DataframeCreator import DataframeCreator
import torchaudio.transforms as T


from dataset import BlueFinLib

config = {
    
    # MODEL CONFIG:
    "architecture": "ResNet50",
    "lr": 1e-3,
    "batch_size": 64, # This number must be bigger than one (nn.BatchNorm).
    "epochs": 25,

    # RUN CONFIG:
    "species": ['Fin', 'Blue'],
    "random_crop_secs": 5, # number of seconds that has the spectrogram.

    # DATA AUGMENTATION CONFIG:
    "random_erasing": 0, # probability that the random erasing operation will be performed.
    "time_mask_param": 10, # number of time steps that will be masked.
    "freq_mask_param": 10, # number of frequency steps that will be masked.
    "spec_aug_prob": 0,
    
    # PATHS:
    "df_dir": "/home/usuaris/veu/marc.casals/dataframes", # where the pickle dataframe is stored.
    "save_dir": "/home/usuaris/veussd/DATABASES/Ocean/checkpoints/", # where we save the model checkpoints.
    "train_specs": '/home/usuaris/veussd/DATABASES/Ocean/SPECTROGRAMS_NEW_SS/SS_50/TRAIN',
    "val_specs": '/home/usuaris/veussd/DATABASES/Ocean/SPECTROGRAMS_NEW_SS/SS_50/VALID',
    "test_specs": '/home/usuaris/veussd/DATABASES/Ocean/SPECTROGRAMS_NEW_SS/SS_50/TEST'
}


# Create the different pandas dataframes.
df_creator_train = DataframeCreator(config['train_specs'], config['df_dir'])
config["df_path_train"] = df_creator_train.get_df_path()

df_creator_val = DataframeCreator(config['val_specs'], config['df_dir'])
config["df_path_val"] = df_creator_val.get_df_path()

df_creator_test = DataframeCreator(config['test_specs'], config['df_dir'])
config["df_path_test"] = df_creator_test.get_df_path()

# ---------------------------------------------------------------------------------------------------

# define the transformations
# torch.nn.Sequential
train_transforms = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(0.5, 0.5),
                                    transforms.RandomErasing(p=config["random_erasing"]),
                                    transforms.RandomApply(torch.nn.ModuleList([
                                        T.TimeMasking(time_mask_param=config["time_mask_param"]),
                                        T.FrequencyMasking(freq_mask_param=config["freq_mask_param"])]), p=config["spec_aug_prob"]),
                                    ])

data_transforms = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(0.5, 0.5),])

# create the data with these transformations:
train_data = BlueFinLib(pickle_path = config['df_path_train'], 
                        img_dir = config['train_specs'], 
                        config = config,
                        transform=train_transforms)
val_data = BlueFinLib(pickle_path = config['df_path_val'], 
                        img_dir = config['val_specs'], 
                        config = config,
                        transform=data_transforms)
test_data = BlueFinLib(pickle_path = config['df_path_test'],
                        img_dir = config['test_specs'], 
                        config = config,
                        transform=data_transforms)

# ---------------------------------------------------------------------------------------------------

for i in tqdm(range(10)):
    features, label = train_data.__getitem__(i)
    image_np = features.numpy()
    # Transpose the array to match the expected image format
    image_np = image_np.transpose(1, 2, 0)
    # Plot the image
    plt.imshow(image_np)
    plt.axis('off')  # Disable axis
    plt.savefig(f'/home/usuaris/veu/marc.casals/IMAGES/{i}.png')