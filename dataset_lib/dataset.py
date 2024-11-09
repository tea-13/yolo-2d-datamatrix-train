# -*- coding: utf-8 -*-

import os
import shutil
import random as rand
from typing import List

from .dataset_item import DatasetItem
from .yaml_config import YAMLConfig
from .workspace import Workspace


class Dataset:
    """
    Class for creating a dataset in a format suitable
    for training YOLO from another dataset
    """
    def __init__(
            self, dataset_root: str,
            new_dataset_root: str, new_dataset_name: str,
            test_prop: float, valid_prop: float):
        """
        Init method

        :param dataset_root:      path to current dataset
        :param new_dataset_root:  path to save new dataset
        :param new_dataset_name:  name for new dataset
        :param test_prop:         proportion of elements in the test sample
        :param valid_prop:        proportion of elements in the valid sample
        """
        self.dataset_path = dataset_root
        self.new_dataset_path = f"{new_dataset_root}\\{new_dataset_name}"
        self.test_prop = test_prop
        self.valid_prop = valid_prop

        self.ws = Workspace(new_dataset_root, new_dataset_name)
        self.ws.create_workspace()

        self.yml_conf = YAMLConfig(
            f"train/images", f"test/images",
            f"valid/images", 1, ["datamatrix"])

        self.yml_conf.save_config(
            f"{new_dataset_root}\\{new_dataset_name}\\config.yaml",
            sort_keys=False, default_flow_style=False)

        self.file_names = []
        self.dataset = []

    def create_dataset_path(self):
        """
        Method for creating a list with numbers of existing images

        :return:
        """
        content_dir: List[str] = os.listdir(self.dataset_path)

        file_names = []
        file_names2 = []
        for file_name in content_dir:
            if ".txt" in file_name:
                file_names.append(int(file_name[5:-4]))

            if ".jpg" in file_name:
                file_names2.append(int(file_name[5:-4]))

        # We choose the smallest list of the two
        # to avoid errors of non-existence of labels or images
        if len(file_names) > len(file_names2):
            self.file_names = file_names2
        else:
            self.file_names = file_names

    def create_dataset_list(self):
        """
        Method for creating a DatasetItem and bringing
        all dataset elements to a single form

        :return:
        """
        for image_num in self.file_names:
            ds_item = DatasetItem(self.dataset_path, image_num)
            ds_item.resize((640, 640))
            self.dataset.append(ds_item)

    def create_dataset(self):
        """
        Creating a dataset from DatasetItem elements
        without saving to device memory

        :return:
        """
        self.create_dataset_path()
        self.create_dataset_list()

    def save_dataset(self):
        """
        Method for saving dataset into folders and distributing
        elements in desired proportions

        :return:
        """
        rand.shuffle(self.dataset)  # shuffle the dataset elements

        amount = len(self.dataset)
        threshold_test = int(self.test_prop * amount)
        threshold_valid = int(self.valid_prop * amount)
        threshold_train = amount - threshold_test - threshold_valid

        # Save elements to each folder in accordance
        # with the specified proportions
        mode = "train"
        for i, ds_item in enumerate(self.dataset):
            if threshold_train < i:
                mode = "test"
            if (threshold_train+threshold_test) < i:
                mode = "valid"

            self.ws.save_image_with_labels(
                    ds_item.file_name, ds_item.image,
                    ds_item.get_labels_str(), mode)


if __name__ == "__main__":
    ROOT = r"D:\project\Python\YOLO_train_data_matrix\2d-data-matrix-code"
    DS_NAME = "DATASET_V2"
    NEW_DS_NAME = "main-dataset"

    shutil.rmtree(f"{ROOT}\\{NEW_DS_NAME}")  # if dataset exists

    ds = Dataset(f"{ROOT}\\{DS_NAME}", ROOT, NEW_DS_NAME, 0.2, 0.1)
    ds.create_dataset()
    ds.save_dataset()
