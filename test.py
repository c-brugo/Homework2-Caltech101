DATA_DIR = 'Caltech101/101_ObjectCategories'
from caltech_dataset import Caltech
import os

print(os.getcwd())

# Prepare Pytorch train/test Datasets
train_dataset = Caltech(DATA_DIR, split='train')