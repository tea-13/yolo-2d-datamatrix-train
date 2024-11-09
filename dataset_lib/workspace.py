# -*- coding: utf-8 -*-

import os
import shutil
from typing import Iterable

import numpy as np
import cv2


class Workspace:
    """
    Class for creating and working with a dataset in Explorer
    """
    def __init__(self, root_directory: str, workdir_name: str):
        """
        Init method

        :param root_directory: directory in which the dataset is created
        :param workdir_name:   name for dataset folder
        """
        self.root = root_directory
        self.workdir = workdir_name

        train_dir_name = "train"
        test_dir_name = "test"
        valid_dir_name = "valid"

        self.folders = {
            "train": f"{self.workdir}\\{train_dir_name}",
            "test": f"{self.workdir}\\{test_dir_name}",
            "valid": f"{self.workdir}\\{valid_dir_name}",
        }

    def create_workspace(self):
        """
        Create all folders for dataset

        :return:
        """
        current_dir = os.getcwd()  # Save current directory path
        # Create dataset folder in root
        os.chdir(self.root)
        os.mkdir(self.workdir)

        # Create train/test/valid folder
        for folder in self.folders.values():
            os.mkdir(f"{folder}")
            os.mkdir(f"{folder}\\images")
            os.mkdir(f"{folder}\\labels")

        os.chdir(current_dir)  # return to current directory

    def get_dataset_path(self) -> str:
        """
        Method returning path to dataset

        :return:
        """
        return f"{self.root}\\{self.workdir}"

    def get_folders_path(self, name: str) -> str:
        """
        Method returning path to train/test/valid folders

        :param name: train, test or valid
        :return: path to the specified folder
        """
        if name not in self.folders.keys():
            raise KeyError("The mode must be one of the predefined train/test/valid")

        folder = self.folders[name]
        return f"{self.root}\\{folder}"

    def save_image_with_labels(
            self, name: str, image: np.array, labels: Iterable[str], mode: str):
        """

        :param name: name of file
        :param image: image for save
        :param labels: labels for save
        :param mode: train/test/valid
        :return:
        """
        folder = self.get_folders_path(mode)

        current_dir = os.getcwd()  # Save current directory path

        # Create image and label in folder
        os.chdir(f"{folder}\\images")
        cv2.imwrite(f"{name}.jpg", image)

        os.chdir(f"{folder}\\labels")
        with open(f"{name}.txt", 'w') as f:
            for label in labels:
                f.write(f"{label}\n")

        os.chdir(current_dir)  # return to current directory


if __name__ == "__main__":
    root = r"D:\project\Python\YOLO_train_data_matrix\2d-data-matrix-code"

    shutil.rmtree(f"{root}\\my_dataset")

    ws = Workspace(root, "my_dataset")
    ws.create_workspace()

    ws.save_image_with_labels("image1", np.zeros((100, 100)), ['123sasewq', '21321312'], "train")
