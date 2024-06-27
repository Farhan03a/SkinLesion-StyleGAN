import os
import pandas as pd
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from random import random

from torch.utils.data import Dataset, DataLoader

# Define your dataset paths
TRAIN_IMAGES_DIR = '/content/drive/My Drive/Colab/ISIC2018_Task3_Training_Input/ISIC2018_Task3_Training_Input'
TRAIN_LABELS_FILE = '/content/drive/My Drive/Colab/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv'
VAL_IMAGES_DIR = '/content/drive/My Drive/Colab/ISIC2018_Task3_Validation_Input/ISIC2018_Task3_Validation_Input'
VAL_LABELS_FILE = '/content/drive/My Drive/Colab/ISIC2018_Task3_Validation_GroundTruth/ISIC2018_Task3_Validation_GroundTruth/ISIC2018_Task3_Validation_GroundTruth.csv'
TEST_IMAGES_DIR = '/content/drive/My Drive/Colab/ISIC2018_Task3_Test_Input/ISIC2018_Task3_Test_Input'
TEST_LABELS_FILE = '/content/drive/My Drive/Colab/ISIC2018_Task3_Test_GroundTruth/ISIC2018_Task3_Test_GroundTruth/ISIC2018_Task3_Test_GroundTruth.csv'

# Custom data generator class
class DataGenerator(object):
    def __init__(self, im_size, loc, n, flip=True, suffix='jpg'):
        self.loc = loc
        self.flip = flip
        self.suffix = suffix
        self.n = n
        self.im_size = im_size
        self.images_list = self.read_image_list(self.loc)

    def get_batch(self, amount):
        idx = np.random.randint(0, self.n, amount)
        out = []

        for i in idx:
            temp = Image.open(self.images_list[i]).convert('RGB').resize((self.im_size, self.im_size))
            temp1 = np.array(temp, dtype='float32') / 255
            if self.flip and random() > 0.5:
                temp1 = np.flip(temp1, 1)
            out.append(temp1)

        return np.array(out)

    def read_image_list(self, category):
        filenames = []
        for file in sorted(os.listdir(category)):
            if self.suffix in file:
                filenames.append(os.path.join(category, file))
        return filenames

# Custom PyTorch dataset class
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, labels_file, transform=None):
        self.img_dir = img_dir
        self.labels = pd.read_csv(labels_file)
        self.transform = transform
        self.data_generator = DataGenerator(128, img_dir, len(self.labels), flip=True)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img, label = self.data_generator.get_batch(1)[0], torch.tensor(self.labels.iloc[idx, 1:].values.astype('float')).argmax()
        img = Image.fromarray((img * 255).astype(np.uint8))
        if self.transform:
            img = self.transform(img)
        return img, label

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create dataset instances
train_dataset = CustomImageDataset(img_dir=TRAIN_IMAGES_DIR, labels_file=TRAIN_LABELS_FILE, transform=transform)
val_dataset = CustomImageDataset(img_dir=VAL_IMAGES_DIR, labels_file=VAL_LABELS_FILE, transform=transform)
test_dataset = CustomImageDataset(img_dir=TEST_IMAGES_DIR, labels_file=TEST_LABELS_FILE, transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
