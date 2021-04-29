import os
import random
from contextlib import contextmanager

import numpy as np
import torch


# reproducibility
def deterministic(seed):
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    fix_all_seed(seed)


def fix_all_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def fix_all_seed_within_context(seed):
    random_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_support = torch.cuda.is_available()
    if cuda_support:
        torch_cuda_state = torch.cuda.get_rng_state()
        torch_cuda_state_all = torch.cuda.get_rng_state_all()
    fix_all_seed(seed)
    yield

    random.setstate(random_state)
    np.random.set_state(np_state)  # noqa
    torch.random.set_rng_state(torch_state)  # noqa
    if cuda_support:
        torch.cuda.set_rng_state(torch_cuda_state)
        torch.cuda.set_rng_state_all(torch_cuda_state_all)
