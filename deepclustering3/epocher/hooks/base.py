import weakref
from abc import abstractmethod
from typing import Any

from torch import nn

from ...meters.averagemeter import AverageValueMeter
from ...meters.meter_interface import MeterInterface


class IDIdentifier(type):
    identifiers: set = set()

    def __call__(cls, *args: Any, **kwds: Any) -> Any:
        instance = super().__call__(*args, **kwds)
        name = instance._hook_name  # noqa
        if name in cls.identifiers:
            raise KeyError(f"hook_name: {name} has been created")
        cls.identifiers.add(name)
        return instance


class EpocherHook(nn.Module, metaclass=IDIdentifier):
    """
    The TrainerHook is an extension to enable extra change on Trainer and Epocher,
    but it is mainly designed for Epocher.


    >> epocher = Epocher(**kwargs) #noqa
    >> epocher.init(**kwargs, hooks=EpocherHook(**kwargs)) # they are just reassign references. #noqa
    # before run: it is the begin of a new epoch
    # end run: it is the end of a new epoch
    # before batch: it is the begin of a new batch
    # after batch: it is the end of a new batch, after optimizer.step()
    
    when creating EpocherHook, the epocher has meters, indicators
    """

    def __init__(self, *, hook_name: str = None, **kwargs):
        super().__init__()
        if hook_name is None:
            hook_name = self.__class__.__name__
        self._hook_name = hook_name

    def bind_epocher(self, epocher):
        self._epocher = weakref.proxy(epocher)
        self._register_meters()

    def _register_meters(self):
        meters = self.meters
        with meters.focus_on(self._hook_name):
            self.configure_meters(meters)

    def configure_meters(self, meters):
        """all meters have been put into hook name scope"""
        ...

    @property
    def epocher(self):
        if hasattr(self, "_epocher"):
            return self._epocher
        return None

    @property
    def trainer(self):
        if hasattr(self, "_trainer"):
            return self._trainer
        return None

    @property
    def meters(self):
        if hasattr(self, "_epocher"):
            return self.epocher.meters
        return None

    def before_epoch(self):
        ...

    def end_epoch(self):
        ...

    def before_update(self):
        ...

    def end_update(self):
        ...


class RegularizeHook(EpocherHook):
    def __init__(self, *, weight: float, **kwargs):
        super().__init__(**kwargs)
        self._weight = float(weight)

    def configure_meters(self, meters: MeterInterface):
        meters.register_meter("weight", AverageValueMeter())

    def __call__(self, **kwargs):
        with self.meters.focus_on(self._hook_name):
            self.meters["weight"].add(self._weight)

            return self._regularize(**kwargs)

    @abstractmethod
    def _regularize(self, **kwargs):
        ...

    def __repr__(self):
        return f"{self._hook_name}: reg_weight = {self._weight}"
