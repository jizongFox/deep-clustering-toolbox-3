from abc import ABCMeta

import torch
from torch import nn
from torch.utils.data import DataLoader

from deepclustering3.epocher import EpocherWithPlugin
from deepclustering3.meters.averagemeter import AverageValueMeter
from deepclustering3.meters.meter_interface import MeterInterface
from deepclustering3.types import dataIterType, optimizerType, criterionType


class _ProcessData:
    @staticmethod
    def _preprocess_data(data, device: torch.device):
        image, label = data
        return image.to(device), label.to(device)


class TrainEpocher(EpocherWithPlugin, _ProcessData, metaclass=ABCMeta):

    def __init__(self, *, model: nn.Module, optimizer: optimizerType, criterion: criterionType,
                 train_iter: dataIterType, num_batches: int = None, cur_epoch=0, device="cpu") -> None:
        super().__init__(model=model, num_batches=num_batches, cur_epoch=cur_epoch, device=device)
        self._train_iter = train_iter
        self._optimizer = optimizer
        self._criterion = criterion

    def configure_meters(self, meters: MeterInterface) -> MeterInterface:
        with meters.focus_on("train"):
            meters.register_meter("loss", AverageValueMeter())
        with meters.focus_on("reg"):
            meters.register_meter("acc", AverageValueMeter())
        return meters

    def _run(self, **kwargs):
        self._model.train()
        for self._cur_batch, data in zip(self.indicator, self._train_iter):
            image, label = self._preprocess_data(data, self.device)
            self.hooks_before_update()
            prediction_with_logits = self._model(image)
            sup_loss = self._criterion(prediction_with_logits, label)
            reg_loss = sum([h(logits=prediction_with_logits) for h in self._hooks])
            self._optimizer.zero_grad()
            total_loss = sup_loss + reg_loss
            total_loss.backward()
            self._optimizer.step()
            with torch.no_grad():
                with self.meters.focus_on("train"):
                    self.meters["loss"].add(sup_loss.item())
                with self.meters.focus_on("reg"):
                    self.meters["acc"].add(
                        torch.eq(prediction_with_logits.max(1)[1], label.squeeze()).float().mean().item())
                statics = self.meters.statistics()  # no execution here.
                self.indicator.set_postfix_statics(statics, group_iter_time=None, cache_time=10)
            self.hooks_end_update()


class EvalEpocher(EpocherWithPlugin, _ProcessData):

    def __init__(self, *, model: nn.Module, val_loader: DataLoader, criterion: criterionType, cur_epoch=0,
                 device="cpu") -> None:
        num_batches = len(val_loader)
        super().__init__(model=model, num_batches=num_batches, cur_epoch=cur_epoch, device=device)
        self._val_loader = val_loader
        self._criterion = criterion

    def configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters.register_meter("loss", AverageValueMeter())
        with meters.focus_on("acc"):
            meters.register_meter("acc", AverageValueMeter())
        return meters

    @torch.no_grad()
    def _run(self, **kwargs):
        self._model.eval()
        for self._cur_batch, data in zip(self.indicator, self._val_loader):
            image, label = self._preprocess_data(data, self.device)
            prediction_with_logits = self._model(image)
            loss = self._criterion(prediction_with_logits, label)
            self.meters["loss"].add(loss.item())
            with self.meters.focus_on("acc"):
                self.meters["acc"].add(
                    torch.eq(prediction_with_logits.max(1)[1], label.squeeze()).float().mean().item())
            statics = self.meters.statistics()
            self.indicator.set_postfix_statics(statics)

    def get_score(self):
        with self.meters.focus_on("acc"):
            return self.meters["acc"].summary()["mean"]
