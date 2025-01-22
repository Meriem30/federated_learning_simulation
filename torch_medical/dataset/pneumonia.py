import pandas as pd
import os
import pydicom
from PIL import Image
import torch
from typing import Any, Callable, Optional
from torch.utils.data import IterableDataset

from other_libs.log import log_error, log_debug


class PneumoniaDataset(IterableDataset):
    def __init__(
            self,
            image_dir = str,
            csv_file = str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    )-> None:
        """
        Pneumonia Dataset constructor for iterable datasets.

        Args:
            root (str): Root directory where the images are stored.
            csv_file (str): Path to the CSV file containing metadata.
            transform (callable, optional): Transformation to apply to images.
            target_transform (callable, optional): Transformation to apply to labels.
        """
        self.image_dir = image_dir
        self.csv_file = csv_file
        self.dataframe = pd.read_csv(self.csv_file)
        self.transform = transform
        self.target_transform = target_transform

    def __iter__(self):
        """
        Iterate over the rows of the dataframe and yield (image, label) pairs.
        """
        for idx, row in enumerate(self.dataframe.itertuples(index=False)):
            dicom_path = os.path.join(self.image_dir, f"{row.patientId}.dcm")
            if not os.path.exists(dicom_path):
                # Skip if the DICOM file does not exist
                log_error("Warning: Missing file %s . Skipping.", dicom_path)
                continue

            # Load the DICOM image
            dicom = pydicom.dcmread(dicom_path)
            image = dicom.pixel_array
            image = Image.fromarray(image).convert("RGB")
            # Retrieve the label
            label = torch.tensor(row.target, dtype=torch.long)

            yield idx, (image, label)

    def __len__(self):
        log_debug("len PNEUMONIA dataset = ",len(self.dataframe) )
        return len(self.dataframe)


    def __repr__(self):
        """
        Provides a detailed string representation of the dataset.
        """
        num_datapoints = len(self)
        split_info = "Unknown Split"  # Replace with actual split info if available
        return (
            f"    Dataset Pneumonia\n"
            f"    Number of datapoints: {num_datapoints}\n"
            f"    Root location: {self.image_dir}\n"
            f"    CSV file: {os.path.basename(self.csv_file)}\n"
        )
