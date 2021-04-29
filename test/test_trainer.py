from unittest import TestCase

import numpy as np
import torch

from deepclustering3.trainer._buffer import _BufferMixin
from deepclustering3.trainer._io import _IOMixin


class TestBuffer(TestCase):
    def test_initialize(self):
        buffer = _BufferMixin()
        buffer._register_buffer("string", 123)
        buffer._register_buffer("int", 1)
        buffer._register_buffer("numpy", np.random.randn(1, 1))
        buffer._register_buffer("tensor_cpu", torch.randn(1, 1))
        buffer._register_buffer("tensor_gpu", torch.randn(1, 1, device="cuda"))
        buffer._register_buffer("none", 1)
        buffer._random_value = 12

        state_dict = buffer.state_dict()
        print(state_dict)

        buffer2 = _BufferMixin()
        buffer2._register_buffer("string", None)
        buffer2._register_buffer("int", 1)
        buffer2._register_buffer("numpy", np.random.randn(1, 1))
        buffer2._register_buffer("tensor_cpu", torch.randn(1, 1))
        buffer2._register_buffer("tensor_gpu", torch.randn(1, 1, device="cuda"))
        buffer2._register_buffer("none", None)
        buffer2.load_state_dict(state_dict, strict=True)
        print(buffer2.state_dict())


class TestTrainerIO(TestCase):
    def test_trainer_io(self):
        trainer = _IOMixin(
            save_dir="123", max_epoch=100, num_batches=100, config={},
        )
        from torchvision import models

        trainer._model = models.resnet18()
        trainer._optim = torch.optim.Adam(trainer._model.parameters())
        trainer._sche = torch.optim.lr_scheduler.StepLR(
            trainer._optim, gamma=0.1, step_size=10
        )
        state_dict = trainer.state_dict()

        trainer2 = _IOMixin(
            save_dir="123", max_epoch=100, num_batches=100, config={},
        )
        trainer2._model = models.resnet18()
        trainer2._optim = torch.optim.Adam(trainer._model.parameters())
        trainer2._sche = torch.optim.lr_scheduler.StepLR(
            trainer._optim, gamma=0.1, step_size=10
        )
        trainer2.load_state_dict(state_dict, strict=True)
