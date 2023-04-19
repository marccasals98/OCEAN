import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class BlueFinLib(Dataset):
    def __init__(self):
        super().__init__()
    def __len__(self):
        # TO-DO
        return 1
    def __getitem__(self, index):
        # TO-DO
        return 1