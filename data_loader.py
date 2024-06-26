import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

# Define your dataset paths
TRAIN_IMAGES_DIR = '/content/drive/My Drive/shared_with_me/ISIC2018_Task3_Training_Input'
TRAIN_LABELS_FILE = '/content/drive/My Drive/Colab/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv'
VAL_IMAGES_DIR = '/content/drive/My Drive/Colab/ISIC2018_Task3_Validation_Input/ISIC2018_Task3_Validation_Input'
VAL_LABELS_FILE = '/content/drive/My Drive/Colab/ISIC2018_Task3_Validation_GroundTruth/ISIC2018_Task3_Validation_GroundTruth/ISIC2018_Task3_Validation_GroundTruth.csv'
TEST_IMAGES_DIR = '/content/drive/My Drive/Colab/ISIC2018_Task3_Test_Input/ISIC2018_Task3_Test_Input'
TEST_LABELS_FILE = '/content/drive/My Drive/Colab/ISIC2018_Task3_Test_GroundTruth/ISIC2018_Task3_Test_GroundTruth/ISIC2018_Task3_Test_GroundTruth.csv'

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, labels_file, transform=None):
        self.img_dir = img_dir
        self.labels = pd.read_csv(labels_file)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.labels.iloc[idx, 0] + '.jpg')
        image = Image.open(img_path).convert("RGB")
        label = torch.tensor(self.labels.iloc[idx, 1:].values.astype('float')).argmax()
        if self.transform:
            image = self.transform(image)
        return image, label

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

train_dataset = CustomImageDataset(img_dir=TRAIN_IMAGES_DIR, labels_file=TRAIN_LABELS_FILE, transform=transform)
val_dataset = CustomImageDataset(img_dir=VAL_IMAGES_DIR, labels_file=VAL_LABELS_FILE, transform=transform)
test_dataset = CustomImageDataset(img_dir=TEST_IMAGES_DIR, labels_file=TEST_LABELS_FILE, transform=transform)
