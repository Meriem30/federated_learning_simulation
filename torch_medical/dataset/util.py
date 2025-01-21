import functools
import os

import torch
import torch.utils.data
import torchvision.utils

from other_libs.log import log_debug
from torch_kit import DatasetUtil
from torch_vision import VisionDatasetUtil


class MedicalDatasetUtil(DatasetUtil):
    @functools.cached_property
    def channel(self):
        # CHANGE '3' ?
        x = self._get_sample_input(0)
        assert x.shape[0] <= 3
        return x.shape[0]
    

    def __len__(self) -> int:
        if self.dataset is not None:
            log_debug("calling the len att of MedicalDatasetUtil..")
            self.__len = self.dataset.__len__()
            log_debug(" the returned len : %s ", self.__len)
            return self.__len

""" def prepare_dataset_paths(self):
        #config =  # load files from config/ datasetutil method
        csv_path=os.path.join(os.path.expanduser("~"),"nvx_data_test.csv"),
        image_dir=os.path.join(os.path.expanduser("~"),"rsna-pneumonia-detection-challenge/stage_2_train_images/"),
        # Validate paths
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        if not os.path.isdir(image_dir):
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
 return csv_path, image_dir"""