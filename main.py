# -*- coding: utf-8 -*-

import shutil

from dataset_lib import Dataset


if __name__ == "__main__":
    ROOT = r"D:\project\Python\YOLO_train_data_matrix\2d-data-matrix-code"
    DS_NAME = "DATASET_V2"
    NEW_DS_NAME = "main-dataset"

    # shutil.rmtree(f"{ROOT}\\{NEW_DS_NAME}")  # if dataset exists

    ds = Dataset(f"{ROOT}\\{DS_NAME}", ROOT, NEW_DS_NAME, 0.2, 0.1)
    ds.create_dataset()
    ds.save_dataset()
