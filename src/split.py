import os
import shutil
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Define directories and proportions
source_directory = '/home/usuaris/veussd/DATABASES/Ocean/DATASETS/Original_20230802/data'
train_directory = '/home/usuaris/veussd/DATABASES/Ocean/DATASETS/Original_20230802_Train/data'
valid_directory = '/home/usuaris/veussd/DATABASES/Ocean/DATASETS/Original_20230802_Valid/data'
test_directory = '/home/usuaris/veussd/DATABASES/Ocean/DATASETS/Original_20230802_Test/data'

# Create directories
os.makedirs(train_directory, exist_ok=True)
os.makedirs(valid_directory, exist_ok=True)
os.makedirs(test_directory, exist_ok=True)

# List all files and shuffle them
all_files = os.listdir(source_directory)
all_files = shuffle(all_files)

# Extract class labels
class_labels = [file_name.split('_')[2] for file_name in all_files]

# Perform stratified split
train_files, test_files, _, _ = train_test_split(all_files, class_labels, test_size=0.2, stratify=class_labels, random_state=42)
train_files, valid_files, _, _ = train_test_split(train_files, [file_name.split('_')[0] for file_name in train_files], test_size=0.25, stratify=[file_name.split('_')[0] for file_name in train_files], random_state=42)

# Function to move files
def move_files(file_list, destination_directory):
    for file_name in file_list:
        source_path = os.path.join(source_directory, file_name)
        destination_path = os.path.join(destination_directory, file_name)
        shutil.move(source_path, destination_path)

# Move files to respective directories
move_files(train_files, train_directory)
move_files(valid_files, valid_directory)
move_files(test_files, test_directory)