from src.dataset import BlueFinLib


def test_dataset_init():
    dataset = BlueFinLib("data/BlueFinLib.pkl", "data/spectrograms")
    assert isinstance(dataset, BlueFinLib)
    assert len(dataset) == 1000