# -*- coding: utf-8 -*-

from typing import List

import yaml


class YAMLConfig:
    """
    Class for creating a config for training a model in yaml format
    """
    config = {
        "train": "",
        "val": "",
        "test": "",
        "nc": 0,
        "names": [],
    }

    def __init__(self, train_path: str, test_path: str,
                 valid_path: str, number_of_class: int, names: List[str]):
        """
        Init methods

        :param train_path: (string) the path to the train folder is either absolute or relative to the location of the config
        :param test_path: (string) the path to the test folder is either absolute or relative to the location of the config
        :param valid_path: (string) the path to the valid folder is either absolute or relative to the location of the config
        :param number_of_class: (int)  number of classes
        :param names: (List of strings) class names
        """
        # Fill the dict
        self.config["train"] = train_path
        self.config["test"] = test_path
        self.config["val"] = valid_path
        self.config["nc"] = number_of_class
        self.config["names"] = names

    def get_data(self, key: str) -> str:
        """
        Returning data contents to configuration by key

        :param key: (string)
        :return: item of dict or None
        """

        if not isinstance(key, str):
            raise TypeError("The key must be of type string")

        item = self.config.get(key, "")

        return item

    def save_config(self, file_name: str, **kwargs):
        """
        Save yaml data to file

        :param file_name: (string) name or path with name to save file
        :return: None
        """
        with open(file_name, 'w') as _stream:
            yaml.safe_dump(self.config, stream=_stream, **kwargs)


if __name__ == "__main__":
    yml_conf = YAMLConfig(
        "train/images", "test/images",
        "valid/images", 1, ["datamatrix"]
    )

    print(yml_conf.get_data("train"))
    print(yml_conf.get_data("__train__"))

    path = "D:/project/Python/YOLO_train_data_matrix/2d-data-matrix-code/YAML/config.yaml"
    yml_conf.save_config(path, sort_keys=False, default_flow_style=False)
