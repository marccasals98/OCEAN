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



from dataset import BlueFinLib

config = {
    "architecture": "ResNet50",
    "lr": 1e-3,
    "batch_size": 60, # This number must be bigger than one (nn.BatchNorm)
    "epochs": 1,
    "num_samples_train": 0.6,
    "num_samples_val": 0.2,
    "num_samples_test": 0.2,
    "species": ['Fin', 'Blue'],
    "random_crop_secs": 5, 
    "pickle_path": "/home/usuaris/veussd/DATABASES/Ocean/df_23_05_21_12_08_09_23hqmc53_zany-totem-48.pkl",
    "img_dir": "/home/usuaris/veussd/DATABASES/Ocean/Spectrograms_AcousticTrends/23_05_21_12_08_09_23hqmc53_zany-totem-48",
    "save_dir": "/home/usuaris/veussd/DATABASES/Ocean/checkpoints/"
}

data_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])

total_data = BlueFinLib(pickle_path = config['pickle_path'], 
                        img_dir = config['img_dir'], 
                        config = config,
                        transform=data_transforms)
for i in range(10):
    features, label = total_data.__getitem__(i)
    image_np = features.numpy()
    # Transpose the array to match the expected image format
    image_np = image_np.transpose(1, 2, 0)
    # Plot the image
    plt.imshow(image_np)
    plt.axis('off')  # Disable axis
    plt.savefig(f'/home/usuaris/veu/marc.casals/IMAGES/{i}.png')