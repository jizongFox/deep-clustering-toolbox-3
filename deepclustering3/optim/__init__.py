from typing import List

from torch.optim import *  # noqa
from torch_optimizer import *  # noqa


def get_lrs_from_optimizer(optimizer: Optimizer) -> List[float]:
    return [p["lr"] for p in optimizer.param_groups]
