import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class CustomDataset(Dataset):
    def __init__(self, data_folder, transform=None):
        self.data_folder = data_folder
        self.file_list = os.listdir(data_folder)
        # self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_folder, self.file_list[idx])
        # image = Image.open(file_path)

        # if self.transform:
        #     image = self.transform(image)

        return file_path