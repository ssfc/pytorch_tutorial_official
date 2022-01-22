# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

#################################################################################################################################
# 1: Loading a Dataset
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


training_data = datasets.FashionMNIST(
    root="data", # root is the path where the train/test data is stored; 
    train=True, # train specifies training or test dataset; 
    download=True, # download=True downloads the data from the internet if it’s not available at root; 
    transform=ToTensor() # transform and target_transform specify the feature and label transformations; 
)

test_data = datasets.FashionMNIST(
    root="data", # root is the path where the train/test data is stored; 
    train=False, # train specifies training or test dataset; 
    download=True, # download=True downloads the data from the internet if it’s not available at root; 
    transform=ToTensor() # transform and target_transform specify the feature and label transformations; 
)

#################################################################################################################################
# 2: Iterating and Visualizing the Dataset


#################################################################################################################################
# 3: Creating a Custom Dataset for your files
# custom Dataset class must implement three functions: __init__, __len__, and __getitem__. 
import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    # The __init__ function is run once when instantiating the Dataset object. 
    # We initialize the directory containing the images, the annotations file, and both transforms (covered in more detail in the next section).
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)  # their labels are stored separately in a CSV file annotations_file; 
        self.img_dir = img_dir  # the FashionMNIST images are stored in a directory img_dir; 
        self.transform = transform
        self.target_transform = target_transform

    # The __len__ function returns the number of samples in our dataset.
    def __len__(self):
        return len(self.img_labels)

    # The __getitem__ function loads and returns a sample from the dataset at the given index idx. 
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)  # it identifies the image’s location on disk, converts that to a tensor using read_image; 
        label = self.img_labels.iloc[idx, 1]  # retrieves the corresponding label from the csv data in self.img_labels; 
        if self.transform:  # calls the transform functions on them (if applicable); 
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label  # returns the tensor image and corresponding label in a tuple; 

#################################################################################################################################
# 4: Preparing your data for training with DataLoaders
from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)  # pass samples in “minibatches”, reshuffle the data at every epoch; 
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)























