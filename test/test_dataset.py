from src.dataset import BlueFinLib
import numpy as np
import torch

'''
To execute the tests, run the following command in the terminal:

pytest ./test

Be sure to be in the root folder.
'''

def test_dataset_init():
    dataset = BlueFinLib(r"C:\Users\marcc\OneDrive\Escritorio\data\extraction_df.pkl", r"C:\Users\marcc\OneDrive\Escritorio\data\imgs")
    assert isinstance(dataset, BlueFinLib)
    assert len(dataset) == 105868

def test_dataset_getitem():
    dataset = BlueFinLib(r"C:\Users\marcc\OneDrive\Escritorio\data\extraction_df.pkl", r"C:\Users\marcc\OneDrive\Escritorio\data\imgs")
    item = dataset[0]
    assert isinstance(item, tuple)
    assert len(item) == 2
    assert isinstance(item[0], torch.Tensor)
    assert isinstance(item[1], str)

def test_dataset_getitem_transform():
    dataset = BlueFinLib(r"C:\Users\marcc\OneDrive\Escritorio\data\extraction_df.pkl", r"C:\Users\marcc\OneDrive\Escritorio\data\imgs", transform=torch.nn.Identity())
    item = dataset[0]
    assert isinstance(item, tuple)
    assert len(item) == 2
    assert isinstance(item[0], torch.Tensor)
    assert isinstance(item[1], str)
    assert item[0].shape == torch.Size([1, 820, 920]) # tried with a random image from Internet.

