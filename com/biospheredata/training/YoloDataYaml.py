from pathlib import Path
import yaml


class YoloDataYaml(object):
    """
    build a data.yaml for training later
    """

    def __init__(self,
                 class_names: list,
                 training_data_path: str,
                 validation_data_path: str,
                 test_data_path: str,
                 split: Path
                 ):
        self.class_names = class_names
        self.training_data_path = training_data_path
        self.validation_data_path = validation_data_path
        self.test_data_path = test_data_path
        self.split = split

    def to_dict(self):
        return {
            "names": self.class_names,
            "nc": len(self.class_names),
            "path": str(self.split),
            "train": self.training_data_path,
            "val": self.validation_data_path,
            "test": self.test_data_path

        }

    def to_yaml(self, file_path: Path):
        file_path.mkdir(parents=True, exist_ok=True)
        data_yaml = file_path.joinpath("data.yaml")
        with open(data_yaml, 'w') as f:
            yaml.dump(self.to_dict(), f, sort_keys=False, default_flow_style=False)

        return data_yaml
