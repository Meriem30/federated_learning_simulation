import pandas as pd
import os
import pydicom
from PIL import Image
import torch
from torch.utils.data import Dataset


class PneumoniaDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.dataframe = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        dicom_path = os.path.join(self.image_dir, f"{row['patientId']}.dcm")
        dicom = pydicom.dcmread(dicom_path)
        image = dicom.pixel_array
        image = Image.fromarray(image).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(row["class_encoded"], dtype=torch.float32)
        return image, label
