from src.dataset import BlueFinLib
import torch

'''
To execute the tests, run the following command in the terminal:

pytest ./test

Be sure to be in the root folder.
'''

config = {
    "lr": 1e-3,
    "batch_size": 64,
    "epochs": 10,
    "num_samples_train": 1000,
    "num_samples_val": 100,
    "num_samples_test": 100,
    "random_crop_frames": 100,
}


def test_dataset_init():
    dataset = BlueFinLib(r"C:\Users\marcc\OneDrive\Escritorio\data\extraction_df.pkl",
                        r"C:\Users\marcc\OneDrive\Escritorio\data\imgs",
                        config)
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


