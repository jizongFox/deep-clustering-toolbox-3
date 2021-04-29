from abc import ABCMeta, abstractmethod
from typing import Union, List

import torch
from torch import nn

from deepclustering3 import optim
from deepclustering3.scheduler.warmup import GradualWarmupScheduler
from ._functional import _ToMixin
from ._io import _IOMixin, _TensorWriterMixin, _StorageMixin
from ..amp.ddp import _DDPMixin
from ..epocher import Epocher
from ..epocher.hooks import EpocherHook
from ..types import criterionType as _criterion_type, dataIterType as _dataiter_type, genericLoaderType as _loader_type, \
    optimizerType as _optimizer_type


class Trainer(_DDPMixin, _StorageMixin, _TensorWriterMixin, _ToMixin, _IOMixin, metaclass=ABCMeta):

    def __init__(self, *, model: nn.Module, criterion: _criterion_type, tra_loader: _dataiter_type,
                 val_loader: _loader_type, save_dir: str, max_epoch: int = 100, num_batches: int = 100, device="cpu",
                 **kwargs) -> None:
        super().__init__(save_dir=save_dir, max_epoch=max_epoch, num_batches=num_batches, device=device, **kwargs)
        self._model = model
        self._optimizer: _optimizer_type = None  # noqa
        self._criterion = criterion
        self._tra_loader: _dataiter_type = tra_loader
        self._val_loader: _loader_type = val_loader
        self.__hooks__ = nn.ModuleList()
        self.__initialized__ = False

    def init(self, *, config, **kwargs):
        self._config = config  # noqa
        self.dump_config(self._save_dir)
        self._init(**kwargs)
        self._optimizer = self._init_optimizer()
        self._init_scheduler(self._optimizer)
        self.__initialized__ = True

    def register_hooks(self, hooks: Union[EpocherHook, List[EpocherHook]]):
        if self.__initialized__:
            raise RuntimeError("`register_hook must be called before `init()``")
        hooks = hooks if isinstance(hooks, (list, tuple)) else [hooks, ]
        self.__hooks__.extend(hooks)

    @abstractmethod
    def _init(self, **kwargs):
        ...

    def _init_optimizer(self) -> _optimizer_type:
        optim_params = self._config["Optim"]
        optimizer = optim.__dict__[optim_params["name"]](
            params=filter(lambda p: p.requires_grad, self._model.parameters()),
            **{k: v for k, v in optim_params.items() if k != "name" and k != "pre_lr" and k != "ft_lr"}
        )
        optimizer.add_param_group({"params": self.__hooks__.parameters(),
                                   **{k: v for k, v in optim_params.items()
                                      if k != "name" and k != "pre_lr" and k != "ft_lr"}
                                   })
        return optimizer

    def _init_scheduler(self, optimizer):
        scheduler_params = self._config.get("Scheduler", None)
        if scheduler_params is None:
            return
        max_epoch = self._max_epoch
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max_epoch - self._config["Scheduler"]["warmup_max"],
            eta_min=1e-7
        )
        scheduler = GradualWarmupScheduler(optimizer, scheduler_params["multiplier"],
                                           total_epoch=scheduler_params["warmup_max"],
                                           after_scheduler=scheduler)
        self._scheduler = scheduler

    def start_training(self, **kwargs):
        if not self.__initialized__:
            raise RuntimeError(f"{self.__class__.__name__} should call `init()` first")
        self.to(self.device)
        if self.on_master():
            with self._writer:
                return self._start_training(**kwargs)
        return self._start_training(**kwargs)

    def _start_training(self, **kwargs):
        start_epoch = max(self._cur_epoch + 1, self._start_epoch)
        self._cur_score: float

        for self._cur_epoch in range(start_epoch, self._max_epoch):
            with self._storage:  # save csv each epoch
                train_metrics = self.run_tra_epoch()
                if self.on_master():
                    eval_metrics, cur_score = self.run_eval_epoch(model=self._model, loader=self._val_loader)
                    self._storage.add_from_meter_interface(tra=train_metrics, val=eval_metrics, epoch=self._cur_epoch)
                    self._writer.add_scalars_from_meter_interface(
                        tra=train_metrics, val=eval_metrics, epoch=self._cur_epoch
                    )
                self.save_to(save_name="last.pth")
                if self._best_score < cur_score:
                    self.save_to(save_name="best.pth")

    def run_tra_epoch(self, **kwargs):
        epocher = self._create_tra_epoch(**kwargs)
        epocher = self._init_tra_epoch(epocher)
        if len(self.__hooks__) > 0:
            epocher.register_hooks(list(self.__hooks__))
        return self._run_tra_epoch(epocher)

    @staticmethod
    def _run_tra_epoch(epocher):
        epocher.run()
        return epocher.get_metric()

    @abstractmethod
    def _create_tra_epoch(self, **kwargs) -> Epocher:
        ...

    @abstractmethod
    def _init_tra_epoch(self, epocher: Epocher) -> Epocher:
        ...

    @torch.no_grad()
    def run_eval_epoch(self, *, model, loader, **kwargs):
        epocher = self._create_eval_epoch(model=model, loader=loader, **kwargs)
        return self._run_eval_epoch(epocher)

    @abstractmethod
    def _create_eval_epoch(self, *, model, loader, **kwargs) -> Epocher:
        ...

    @staticmethod
    def _run_eval_epoch(epocher):
        epocher.run()
        return epocher.get_metric(), epocher.get_score()
