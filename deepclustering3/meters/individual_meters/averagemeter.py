from collections import defaultdict
from typing import List

import numpy as np

from ._metric import _Metric, MeterResultDict


class AverageValueMeter(_Metric):
    def __init__(self):
        super(AverageValueMeter, self).__init__()
        self.reset()
        self.val = 0

    def add(self, value, n=1):
        self.val = value
        self.sum += value
        self.var += value * value
        self.n += n

        if self.n == 0:
            self.mean, self.std = np.nan, np.nan
        elif self.n == 1:
            self.mean = 0.0 + self.sum  # This is to force a copy in torch/numpy
            self.std = np.inf
            self.mean_old = self.mean
            self.m_s = 0.0
        else:
            self.mean = self.mean_old + (value - n * self.mean_old) / float(self.n)
            self.m_s += (value - self.mean_old) * (value - self.mean)
            self.mean_old = self.mean
            self.std = np.sqrt(self.m_s / (self.n - 1.0))

    def value(self):
        return self.mean, self.std

    def reset(self):
        self.n = 0
        self.sum = 0.0
        self.var = 0.0
        self.val = 0.0
        self.mean = np.nan
        self.mean_old = 0.0
        self.m_s = 0.0
        self.std = 0.0

    def summary(self) -> dict:
        # this function returns a dict and tends to aggregate the historical results.
        return MeterResultDict({"mean": self.value()[0]})

    def detailed_summary(self) -> dict:
        # this function returns a dict and tends to aggregate the historical results.
        return MeterResultDict({"mean": self.value()[0], "val": self.value()[1]})

    def __repr__(self):
        return f"{self.__class__.__name__}: n={self.n} \t {self.detailed_summary()}"


class MultipleAverageValueMeter(_Metric):
    def __init__(self) -> None:
        super().__init__()
        self._meter_dicts = defaultdict(AverageValueMeter)

    def reset(self):
        for k, v in self._meter_dicts.items():
            v.reset()

    def add(self, *_, **kwargs):
        for k, v in kwargs.items():
            self._meter_dicts[k].add(v)

    def summary(self) -> MeterResultDict:
        result = {}
        for k, v in self._meter_dicts.items():
            result[k] = v.summary()["mean"]
        return MeterResultDict(result)

    def detailed_summary(self) -> MeterResultDict:
        return self.summary()


class AverageValueListMeter(MultipleAverageValueMeter):
    def add(self, list_value: List[float]):
        for i, v in enumerate(list_value):
            self._meter_dicts[str(i)].add(v)
