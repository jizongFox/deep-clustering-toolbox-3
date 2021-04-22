import numbers
from abc import abstractmethod, ABCMeta
from functools import wraps

import numpy as np
import torch


class CustomizedType(type):
    """
    This class is reserved for customized meter dictionary that return non-float results.
    """

    def __repr__(self):
        return f"{self.__class__.__name__}"


def check_parameter(func, k, v):
    if not isinstance(k, str) or not isinstance(
        v, (torch.Tensor, np.ndarray, numbers.Number, CustomizedType)
    ):
        raise KeyError(
            f"{func.__name__} does not allow `{k}: {str(v)}` to be set as a MeterResult"
        )


def check_parameters(func):
    @wraps(func)
    def checker(*args, **kwargs):
        if len(args):
            for _dict in args:
                for k, v in _dict.items():
                    check_parameter(func, k, v)
        return func(*args, **kwargs)

    return checker


class MeterResultDict(dict):
    """
    A meter Result dict that only allow key-value pairs as
        - (str, numbers)
        - (str, tensor)
        - (str, np.ndarray)
    """

    @check_parameters
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        check_parameter(self.__class__, key, value)
        super(MeterResultDict, self).__setitem__(key, value)

    def __repr__(self):
        string_info = ""
        kv_pairs = "\t".join([f"{k}:{v:.3f}" for k, v in self.items()])
        string_info += kv_pairs
        return string_info


class _Metric(metaclass=ABCMeta):
    """Base class for all metrics.
    record the values within a single epoch
    From: https://github.com/pytorch/tnt/blob/master/torchnet/meter/meter.py
    """

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def add(self, *args, **kwargs):
        pass

    @abstractmethod
    def summary(self) -> MeterResultDict:
        pass

    @abstractmethod
    def detailed_summary(self) -> MeterResultDict:
        pass
