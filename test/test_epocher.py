from unittest import TestCase

from torch import nn
from torchvision.models import resnet18

from deepclustering3.epocher import Epocher
from deepclustering3.epocher.hooks.ent_min import EntropyMinHook
from deepclustering3.meters.averagemeter import AverageValueMeter
from deepclustering3.meters.meter_interface import MeterInterface

model = resnet18()


class TestEpocher(Epocher):

    def __init__(self, *, model: nn.Module, num_batches: int, cur_epoch=0, device="cpu") -> None:
        super().__init__(model=model, num_batches=num_batches, cur_epoch=cur_epoch, device=device)

    def configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters.register_meter("meter1", AverageValueMeter())
        return meters

    def _run(self, **kwargs):
        for i in range(100):
            self.meters["meter1"].add(1)


class TestHook(TestCase):

    def setUp(self) -> None:
        super().setUp()
        self._epocher = TestEpocher(model=model, num_batches=100, )

    def test_no_same_hooks(self):
        hook1 = EntropyMinHook(weight=0.1, epocher=self._epocher, hook_name="reg1")
        hook2 = EntropyMinHook(weight=0.2, epocher=self._epocher, hook_name="reg2")

        with self.assertRaises(KeyError):
            hook1 = EntropyMinHook(weight=0.1, epocher=self._epocher, hook_name="same_name")
            hook2 = EntropyMinHook(weight=0.2, epocher=self._epocher, hook_name="same_name")
