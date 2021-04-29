import os
from typing import Optional

import torch
import torch.distributed as dist
from torch import nn
from torch.backends import cudnn


def initialize_ddp_environment(
    rank,
    ngpus_per_node,
    dist_backend="nccl",
    world_size=None,
    dist_url="tcp://localhost:11111",
):
    if world_size is None:
        world_size = ngpus_per_node
    ddp_params = {
        "rank": rank,
        "dist_backend": dist_backend,
        "dist_url": dist_url,
        "ngpus_per_node": ngpus_per_node,
    }
    torch.cuda.set_device(rank)
    cudnn.benchmark = True
    print("Use GPU: {} for training".format(rank))

    os.environ["PYTHONWARNINGS"] = "ignore:semaphore_tracker:UserWarning"
    dist.init_process_group(
        backend=dist_backend, init_method=dist_url, world_size=world_size, rank=rank
    )
    if dist.get_rank() != 0:
        disable_output()
        # pass
    return ddp_params


def convert2syncBN(network: nn.Module):
    return nn.SyncBatchNorm.convert_sync_batchnorm(network)


def disable_output():
    def print_pass(*args):
        pass

    import builtins

    builtins.print = print_pass


class _DDPMixin:
    @property
    def rank(self) -> Optional[int]:
        try:
            return dist.get_rank()
        except (AssertionError, AttributeError, RuntimeError):
            return None

    def on_master(self) -> bool:
        return (self.rank == 0) or (self.rank is None)
