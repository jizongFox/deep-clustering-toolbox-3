import os

from torch import nn

from deepclustering3.epocher import Epocher
from deepclustering3.trainer import Trainer as _Trainer
from deepclustering3.types import criterionType as _criterion_type, \
    dataIterType as _dataiter_type, genericLoaderType as _loader_type
from .epochers import TrainEpocher, EvalEpocher


class Trainer(_Trainer):
    RUN_PATH = f"{os.path.dirname(__file__)}/runs"

    def __init__(self, *, model: nn.Module, criterion: _criterion_type, tra_loader: _dataiter_type,
                 val_loader: _loader_type, save_dir: str, max_epoch: int = 100, num_batches: int = 10, device="cpu",
                 **kwargs) -> None:
        super().__init__(model=model, criterion=criterion, tra_loader=tra_loader, val_loader=val_loader,
                         save_dir=save_dir, max_epoch=max_epoch, num_batches=num_batches, device=device, **kwargs)

    def _init(self, **kwargs):
        pass

    def _create_tra_epoch(self, **kwargs) -> Epocher:
        epocher = TrainEpocher(
            model=self._model, optimizer=self._optimizer, criterion=self._criterion, train_iter=self._tra_loader,
            num_batches=self._num_batches, cur_epoch=self._cur_epoch, device=self.device
        )
        return epocher

    def _init_tra_epoch(self, epocher: Epocher) -> Epocher:
        epocher.init()
        return epocher

    def _create_eval_epoch(self, *, model, loader, **kwargs) -> Epocher:
        epocher = EvalEpocher(model=model, val_loader=loader, criterion=self._criterion, cur_epoch=self._cur_epoch,
                              device=self.device)
        epocher.init()
        return epocher
