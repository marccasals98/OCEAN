from src.dataset import BlueFinLib

'''
To execute the tests, run the following command in the terminal:

pytest ./test

Be sure to be in the root folder.
'''

def test_dataset_init():
    dataset = BlueFinLib(r"C:\Users\marcc\OneDrive\Escritorio\data\extraction_df.pkl", r"C:\Users\marcc\OneDrive\Escritorio\data\imgs")
    assert isinstance(dataset, BlueFinLib)
    assert len(dataset) == 105868