import functools
from abc import ABCMeta
from collections import defaultdict
from typing import DefaultDict, Callable, List, Dict

import pandas as pd
from termcolor import colored

from deepclustering2.meters2.meter_interface import EpochResultDict
from deepclustering2.utils import path2Path
from .historicalContainer import HistoricalContainer
from .utils import rename_df_columns

__all__ = ["Storage"]


class StorageIncomeDict:
    def __init__(self, **kwargs) -> None:
        for k, v in kwargs.items():
            setattr(self, f"{k}", v)

    def __repr__(self):
        string_info = ""
        for k, v in self.__dict__.items():
            string_info += colored(f"{k}:\n", "red")
            string_info += f"{v}"
        return string_info


class _IOMixin:
    _storage: DefaultDict[str, HistoricalContainer]
    summary: Callable[[], pd.DataFrame]

    def state_dict(self):
        return self._storage

    def load_state_dict(self, state_dict):
        self._storage = state_dict
        print("loading from checkpoint:")
        print(colored(self.summary(), "green"))

    def to_csv(self, path, name="storage.csv"):
        path = path2Path(path)
        path.mkdir(exist_ok=True, parents=True)
        self.summary().to_csv(path / name)


class Storage(_IOMixin, metaclass=ABCMeta):
    def __init__(self, csv_save_dir=None, csv_name="storage.csv") -> None:
        super().__init__()
        self._storage = defaultdict(HistoricalContainer)
        self._csv_save_dir = csv_save_dir
        self._csv_name = csv_name

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def put(
        self, name: str, value: Dict[str, float], epoch=None, prefix="", postfix=""
    ):
        self._storage[prefix + name + postfix].add(value, epoch)

    def put_all(
        self, result_name: str, epoch_result: EpochResultDict = None, epoch=None
    ):
        assert isinstance(result_name, str), result_name
        if epoch_result:
            for k, v in epoch_result.items():
                self.put(result_name + "_" + k, v, epoch)

    def put_from_dict(self, income_dict: StorageIncomeDict, epoch: int = None):
        for k, v in income_dict.__dict__.items():
            self.put_all(k, v, epoch)

        if self._csv_save_dir:
            self.to_csv(self._csv_save_dir, name=self._csv_name)

    def get(self, name, epoch=None):
        assert name in self._storage, name
        if epoch is None:
            return self._storage[name]
        return self._storage[name][epoch]

    def summary(self) -> pd.DataFrame:
        """
        summary on the list of sub summarys, merging them together.
        :return:
        """
        try:
            list_of_summary = [
                rename_df_columns(v.summary(), k) for k, v in self._storage.items()
            ]
            # merge the list
            summary = functools.reduce(
                lambda x, y: pd.merge(x, y, left_index=True, right_index=True),
                list_of_summary,
            )
            return pd.DataFrame(summary)
        except TypeError:
            return pd.DataFrame()

    @property
    def meter_names(self, sorted=False) -> List[str]:
        if sorted:
            return sorted(self._storage.keys())
        return list(self._storage.keys())

    @property
    def storage(self):
        return self._storage
