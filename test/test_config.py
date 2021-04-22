import os
import tempfile
from pprint import pprint
from unittest import TestCase

from deepclustering3.config.config_manager import ConfigManger
from deepclustering3.config.utils import extract_subdictionary_from_large
from deepclustering3.utils.io import yaml_write


class TestConfigParser(TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.dictionary1 = {"a": 1,
                            "b": 2.5,
                            "c": ["c1", "c2", "c3"],
                            "d": {"e":
                                      {"e1": 1, "e2": 2, "e3": [1, 2, 3]}
                                  }
                            }
        self.dictionary2 = {"a": 2,
                            "b": 3.5,
                            "c": ["c3"],
                            "d": {"e":
                                      {"e4": None}
                                  }
                            }
        self.input_args = ["   a=null b=3.1415 c=[c1,c2,c3,c4] d=null d.e.e5=1  ", ]

        self.tempdir = tempfile.TemporaryDirectory()
        self._config1_path = os.path.join(str(self.tempdir.name), "config1.yaml")
        self._config2_path = os.path.join(str(self.tempdir.name), "config2.yaml")

        yaml_write(self.dictionary1, str(self.tempdir.name), save_name="config1.yaml")
        yaml_write(self.dictionary2, str(self.tempdir.name), save_name="config2.yaml")

    def test_without_args(self):
        configManager = ConfigManger(base_path=self._config1_path, optional_paths=[self._config2_path, ], verbose=True)
        config = configManager.config

    def test_with_args(self):
        configManager = ConfigManger(base_path=self._config1_path, optional_paths=[self._config2_path, ], verbose=True,
                                     _test_message=self.input_args)
        config = configManager.config

    def test_extract(self):
        pprint(self.dictionary1)
        pprint(extract_subdictionary_from_large(self.dictionary1, {"a": None, "d": {"e": {"e1": None}}}))

    def tearDown(self):
        super(TestConfigParser, self).tearDown()
        self.tempdir.cleanup()
