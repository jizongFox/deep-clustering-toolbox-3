from deepclustering3.criterion import Entropy
from .base import RegularizeHook
from ...meters.averagemeter import AverageValueMeter
from ...meters.meter_interface import MeterInterface


class EntropyMinHook(RegularizeHook):
    def __init__(self, *, weight: float, **kwargs):
        super().__init__(weight=weight, **kwargs)
        self._criterion = Entropy()

    def configure_meters(self, meters: MeterInterface):
        super().configure_meters(meters)
        meters.register_meter("entropy", AverageValueMeter())

    def _regularize(self, *, logits, **kwargs):
        entropy = self._criterion(logits.softmax(1))
        self.meters["entropy"].add(entropy.item())
        return entropy * self._weight
