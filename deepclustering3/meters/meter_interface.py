from collections import OrderedDict
from typing import Dict, List, Optional

from deepclustering2.utils import nice_dict
from .individual_meters._metric import _Metric, MeterResultDict

_Record_Type = Dict[str, float]


class EpochResultDict(dict):
    """
    The dictionary only allows input as an instance of `MeterResult`
    """

    def __init__(self, *args, **kwargs) -> None:

        if len(args):
            for _dict in args:
                for k, v in _dict.items():
                    assert isinstance(k, str), k
                    assert isinstance(v, MeterResultDict), v
        if len(kwargs):
            for k, v in kwargs.items():
                assert isinstance(k, str), k
                assert isinstance(v, MeterResultDict), v
        super(EpochResultDict, self).__init__(*args, **kwargs)

    def __repr__(self):
        string_info = ""
        for k, v in self.items():
            string_info += f"{k}: \n"
            string_info += f"\t{nice_dict(v)}\n"
        return string_info

    def __setitem__(self, key, value):
        assert isinstance(key, str), key
        assert isinstance(value, MeterResultDict), value
        super(EpochResultDict, self).__setitem__(key, value)


class MeterInterface:
    """
    meter interface only concerns about the situation in one epoch,
    without considering historical record and save/load state_dict function.
    """

    def __init__(self) -> None:
        """
        :param meter_config: a dict of individual meter configurations
        """
        self._ind_meter_dicts: Dict[str, _Metric] = OrderedDict()
        self._group_dicts: Dict[str, List[str]] = OrderedDict()

    def __getitem__(self, meter_name: str) -> _Metric:
        try:
            return self.meters[meter_name]
        except KeyError as e:
            raise KeyError(e)

    def register_meter(self, name: str, meter: _Metric, group_name=None) -> None:
        assert isinstance(name, str), name
        assert isinstance(
            meter, _Metric
        ), f"{meter.__class__.__name__} should be a subclass of {_Metric.__name__}, given {meter}."
        # add meters
        self._ind_meter_dicts[name] = meter
        if group_name is not None:
            if group_name not in self._group_dicts:
                self._group_dicts[group_name] = []
            self._group_dicts[group_name].append(name)

    def delete_meter(self, name: str) -> None:
        assert (
            name in self.meter_names
        ), f"{name} should be in `meter_names`: {self.meter_names}, given {name}."
        del self.meters[name]
        for group, meter_namelist in self._group_dicts.items():
            if name in meter_namelist:
                meter_namelist.remove(name)

    def delete_meters(self, name_list: List[str]):
        assert isinstance(
            name_list, list
        ), f" name_list must be a list of str, given {name_list}."
        for name in name_list:
            self.delete_meter(name)

    @property
    def meter_names(self) -> List[str]:
        if hasattr(self, "_ind_meter_dicts"):
            return list(self._ind_meter_dicts.keys())

    @property
    def meters(self) -> Optional[Dict[str, _Metric]]:
        if hasattr(self, "_ind_meter_dicts"):
            return self._ind_meter_dicts
        raise NotImplementedError("_ind_meter_dicts")

    @property
    def group(self) -> List[str]:
        return list(self._group_dicts.keys())

    def _tracking_status(
        self, group_name=None, detailed_summary=False
    ) -> EpochResultDict:
        """
        return current training status from "ind_meters"
        :param group_name:
        :return:
        """
        if group_name:
            assert group_name in self.group
            return EpochResultDict(
                **{
                    k: v.detailed_summary() if detailed_summary else v.summary()
                    for k, v in self.meters.items()
                    if k in self._group_dicts[group_name]
                }
            )
        return EpochResultDict(
            **{
                k: v.detailed_summary() if detailed_summary else v.summary()
                for k, v in self.meters.items()
            }
        )

    def tracking_status(self, group_name=None, final=False, cache_time=10):
        if final:
            return self._tracking_status(group_name=group_name)
        if not hasattr(self, "__n__"):
            self.__n__ = 0
        if not hasattr(self, "__cache__"):
            self.__cache__ = self._tracking_status(group_name=group_name)

        self.__n__ += 1
        if self.__n__ % cache_time == 0:
            self.__cache__ = self._tracking_status(group_name=group_name)
        return self.__cache__

    def add(self, meter_name, *args, **kwargs):
        assert meter_name in self.meter_names
        self.meters[meter_name].add(*args, **kwargs)

    def reset(self) -> None:
        """
        reset individual meters
        :return: None
        """
        for v in self.meters.values():
            v.reset()
