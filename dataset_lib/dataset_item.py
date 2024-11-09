# -*- coding: utf-8 -*-

from typing import Tuple, List

import cv2
import numpy as np


class DatasetItem:
    """
    Class for creating and processing dataset elements
    """
    NAME_PREFIX = "Image"

    def __init__(self, root_directory: str, image_num: int):
        """
        Init method

        :param root_directory:  The directory where the dataset elements are located
        :param image_num:       image number
        """
        self.root = root_directory
        self.name_postfix = DatasetItem.str_addition(image_num)

        # The names in the dataset are named according
        # to the principle f"{self.NAME_PREFIX}{self.name_postfix}"
        self.file_name = f"{self.NAME_PREFIX}{self.name_postfix}"

        self.image_path = f"{self.root}\\{self.file_name}.jpg"
        self.label_path = f"{self.root}\\{self.file_name}.txt"

        self.__image = None
        self.__labels = None

        self.__open_image()
        self.__open_labels()

    @property
    def image(self) -> np.array:
        """
        Getter for field image

        :return: image
        """
        return self.__image

    @property
    def labels(self) -> List[List[int]]:
        """
        Getter for field labels

        :return: labels
        """
        return self.__labels

    def get_labels_str(self) -> List[str]:
        """
        Getter for the labels field, this is a list in a format suitable for YOLO training

        :return: labels in format YOLO coco x, y, w, h
        """

        labels = []

        _height, _width = self.image.shape[:2]

        for _label in self.__labels:
            labels.append([
                round(_label[0], 5),
                round(_label[1] / _width, 5),
                round(_label[2] / _height, 5),
                round(_label[3] / _width, 5),
                round(_label[4] / _height, 5)
            ])

        labels = self.__normalized_labels(labels)

        return [' '.join(map(str, label)) for label in labels]

    def __open_image(self):
        """
        The method opens and saves the image.

        :return:
        """
        self.__image = cv2.imread(self.image_path)

    def __open_labels(self):
        """
        The method opens and saves the labels.

        :return:
        """
        with open(self.label_path) as _file:
            # save all fileline as arrays
            file_labels = [
                [
                    *map(float, file_line.split())
                ] for file_line in _file.read().split('\n')
            ]

        _height, _width = self.image.shape[:2]

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
    def __normalized_labels(array) -> List:
        """
        Method for checking array normalization, sets 0 if less than 0 and 1 if greater than 1

        :param array: array to transform
        :return:      transformed array
        """
        for _i, label in enumerate(array):
            for _j, el in enumerate(label):
                if _j == 0:
                    continue

                if el < 0:
                    array[_i][_j] = 0.0
                elif el > 1:
                    array[_i][_j] = 1.0

        return array

    def resize(self, new_size: Tuple[int, int]):
        """
        The function changes the size of the image and the size of the bounding box

        :param new_size: size for new image
        :return:
        """
        _h, _w = self.image.shape[:2]
        _new_w, _new_h = new_size

        self.__image = cv2.resize(self.image, new_size, interpolation=cv2.INTER_NEAREST)

        for _label in self.__labels:
            _label[1] = int(_new_w / _w * _label[1])
            _label[2] = int(_new_h / _h * _label[2])
            _label[3] = int(_new_h / _h * _label[3])
            _label[4] = int(_new_w / _w * _label[4])

    @staticmethod
    def str_addition(num: int) -> str:
        """
        The method adds 0 to the number if it is single-digit.

        :param num:
        :return:
        """
        str_num = str(num)

        if len(str_num) == 1:
            return f"0{str_num}"

        return str_num


if __name__ == "__main__":
    root = r"D:\project\Python\YOLO_train_data_matrix\2d-data-matrix-code\DATASET_V2"

    for i in range(26, 28):
        ds_it = DatasetItem(root, i)
        print(ds_it.get_labels_str())
        frame = ds_it.image.copy()

        for ll in ds_it.labels:
            _, x, y, w, h = ll

            cv2.rectangle(frame, (x - h // 2, y - w // 2), (x + h // 2, y + w // 2), (255, 0, 0), 2)

        cv2.imshow("frame", frame)
        cv2.waitKey(0)

        ds_it.resize((640, 640))
        print(ds_it.get_labels_str())
        frame = ds_it.image.copy()

        for ll in ds_it.labels:
            _, x, y, w, h = ll

            cv2.rectangle(frame, (x - h // 2, y - w // 2), (x + h // 2, y + w // 2), (255, 0, 0), 2)

        cv2.imshow("frame", frame)
        cv2.waitKey(0)

    cv2.destroyAllWindows()
