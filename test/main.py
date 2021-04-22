import os
import tempfile

from deepclustering3.config.config_manager import ConfigManger
from deepclustering3.utils.io import yaml_write

dictionary1 = {"a": 1,
               "b": 2.5,
               "c": ["c1", "c2", "c3"],
               "d": {"e":
                         {"e1": 1, "e2": 2, "e3": [1, 2, 3]}
                     }
               }
dictionary2 = {"a": 2,
               "b": 3.5,
               "c": ["c3"],
               "d": {"e":
                         {"e4": None}
                     }
               }

with tempfile.TemporaryDirectory() as filepath:
    config1_path = os.path.join(str(filepath), "config1.yaml")
    config2_path = os.path.join(str(filepath), "config2.yaml")

    yaml_write(dictionary1, str(filepath), save_name="config1.yaml")
    yaml_write(dictionary2, str(filepath), save_name="config2.yaml")
    configManager = ConfigManger(base_path=config1_path, optional_paths=config2_path, verbose=True)

    pass
