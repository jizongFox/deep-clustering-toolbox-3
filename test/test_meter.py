from unittest import TestCase

from deepclustering3.meters.individual_meters import AverageValueMeter
from deepclustering3.meters.meter_interface import MeterInterface
from deepclustering3.meters.storage_interface import Storage


class TestMeterInterface(TestCase):

    def test_meter(self):
        meters = MeterInterface()
        meters.register_meter("loss", AverageValueMeter())

        with meters.focus_on("reg"):
            meters.register_meter("loss", AverageValueMeter())

        with meters:
            for i in range(10):
                meters["loss"].add(1.0)
                with meters.focus_on("reg"):
                    meters["loss"].add(10)

        meter_generator = meters.statistics()
        for g, meters in meter_generator:
            print(g, meters)

    def test_del_meter(self):
        meters = MeterInterface()
        meters.register_meter("loss", AverageValueMeter())

        with meters.focus_on("reg"):
            meters.register_meter("loss", AverageValueMeter())
        meters.delete_meter("loss")
        with meters.focus_on("reg"):
            meters.delete_meter("loss")

        print(meters.groups())

    def test_storage(self):
        storage = Storage()
        meters = MeterInterface()
        meters.register_meter("loss", AverageValueMeter())

        with meters.focus_on("reg"):
            meters.register_meter("loss", AverageValueMeter())
            meters.register_meter("loss2", AverageValueMeter())
        print(storage.summary())
        with storage:
            for epoch in range(10):
                with meters:
                    for i in range(100):
                        meters["loss"].add(epoch)
                        with meters.focus_on("reg"):
                            meters["loss"].add(epoch + 10)
                            meters["loss2"].add(epoch + 5)

                statistics = meters.statistics()
                for g, group_dict in statistics:
                    storage.put_group(g, epoch_result=group_dict, epoch=epoch)
            print(storage.summary())

        meter_generator = meters.statistics()
        for g, meters in meter_generator:
            print(g, meters)
