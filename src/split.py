import numpy as np
from sklearn.model_selection import train_test_split
import os
from collections import defaultdict
import random


class Split():
    """
    Split the dataset in train and test.

    Attributes:
    -----------
    src_path: str
        The original path with the files you want to use.
    
    train_path: str
        The path where the train files will be saved.

    test_path: str
        The path where the test files will be saved.    
    
    test_size: float
        The proportion of the dataset to include in the test split.
    
    random_state: int
        Controls the shuffling applied to the data before applying the split.

    Methods:
    --------
    get_class_labels()
        Get class labels and their corresponding file paths.
    
    split()
        Split the dataset in train and test.

    """
    def __init__(self, src_path, train_path, test_path, test_size=0.2, random_state=None):
        self.src_path = src_path
        self.train_path = train_path
        self.test_path = test_path
        self.test_size = test_size
        self.random_state = random_state
    
    def get_class_labels(self):
        # Get class labels and their corresponding file paths
        class_files = defaultdict(list)
        for class_name in os.listdir(self.src_path):
            class_path = os.path.join(self.src_path, class_name)
            if os.path.isdir(class_path):
                class_files[class_name] = [os.path.join(class_path, file_name) for file_name in os.listdir(class_path)]
        return class_files

    def split(self):
        """
        Split the dataset in train and test.
        """
        class_files = self.get_class_labels(self)
        
        # Shuffle the data within each class
        for class_name, file_paths in class_files.items():
            random.shuffle(file_paths)

        # Perform the stratified split
        train_files = []
        test_files = []
        for class_name, file_paths in class_files.items():
            train_data, test_data = train_test_split(file_paths,
                                                    test_size=self.test_size,
                                                    random_state=self.random_state,
                                                    stratify=[class_name]*len(file_paths))
            train_files.extend(train_data)
            test_files.extend(test_data)

        # Create the training and testing folders
        os.makedirs(self.train_folder, exist_ok=True)
        os.makedirs(self.test_folder, exist_ok=True)


