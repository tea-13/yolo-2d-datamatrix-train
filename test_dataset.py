# -*- coding: utf-8 -*-

from typing import List

import cv2
import numpy as np


class DatasetItem:
    NAME_PREFIX = "Image"

    def __init__(self, root_directory: str, image_num: int):
        self.root = root_directory
        self.name_postfix = DatasetItem.str_addition(image_num)
        self.file_name = f"{self.NAME_PREFIX}{self.name_postfix}"

        self.image_path = f"{self.root}\\images\\{self.NAME_PREFIX}{self.name_postfix}.jpg"
        self.label_path = f"{self.root}\\labels\\{self.NAME_PREFIX}{self.name_postfix}.txt"

        self.__image = None
        self.__labels = None

        self.__open_image()
        self.__open_labels()

    @property
    def image(self) -> np.array:
        return self.__image

    @property
    def labels(self) -> List[List[int]]:
        return self.__labels

    def __open_image(self):
        self.__image = cv2.imread(self.image_path)

    def __open_labels(self):
        with open(self.label_path) as _file:
            # save all fileline as arrays
            file_labels = [
                [
                    *map(float, file_line.split())
                ] for file_line in _file.read().split('\n')
            ]

        _height, _width = self.image.shape[:2]

        print(file_labels)
        self.__labels = []
        for _label in file_labels:
            if len(_label) < 5:
                continue

            # Label in format [class_number, x, y, w, h]
            # x, y - center of bounding box
            # w, h - width and height of bounding box
            transformed_label = [
                0,
                int(_label[1] * _width),
                int(_label[2] * _height),
                int(_label[3] * _width),
                int(_label[4] * _height),
            ]

            self.__labels.append(transformed_label)

    @staticmethod
    def str_addition(num: int) -> str:
        str_num = str(num)

        if len(str_num) == 1:
            return f"0{str_num}"

        return str_num


if __name__ == "__main__":
    root = r"D:\project\Python\YOLO_train_data_matrix\2d-data-matrix-code\main-dataset\train"

    ds_it = DatasetItem(root, 3)
    frame = ds_it.image.copy()

    for ll in ds_it.labels:
        print(f"label= {ll}")
        _, x, y, w, h = ll

        cv2.rectangle(frame, (x - h // 2, y - w // 2), (x + h // 2, y + w // 2), (255, 0, 0), 2)

    cv2.imshow("frame", frame)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
