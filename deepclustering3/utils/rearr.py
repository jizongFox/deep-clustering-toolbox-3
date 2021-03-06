from functools import partial
from functools import reduce
from itertools import repeat
from multiprocessing import Pool
from operator import and_
from typing import Iterable, Set, Tuple, TypeVar, Callable, List, Any

import torch
import torch.nn.functional as F
from torch import Tensor

A = TypeVar("A")
B = TypeVar("B")
T = TypeVar("T")


class _ntuple:
    def __init__(self, n: int) -> None:
        super().__init__()
        self._n = n

    def __call__(self, input_: T) -> Tuple[T, ...]:
        if isinstance(input_, str):
            return tuple(repeat(input_, self._n))
        if isinstance(input_, Iterable):
            input_ = list(input_)
            iterable_length = len(input_)
            if iterable_length == 1:
                return tuple(repeat(input_[0], self._n))
            if iterable_length != self._n:
                raise ValueError(f"input's length mismatched with n = {self._n}, given {iterable_length}: {input_}")
            return tuple(input_)
        else:
            return tuple(repeat(input_, self._n))


single = _ntuple(1)
pair = _ntuple(2)
triple = _ntuple(3)
quadruple = _ntuple(4)


# Assert utils
def uniq(a: Tensor) -> Set:
    """
    return unique element of Tensor
    Use python Optimized mode to skip assert statement.
    :rtype set
    :param a: input tensor
    :return: Set(a_npized)
    """
    return set([x.item() for x in a.unique()])


def sset(a: Tensor, sub: Iterable) -> bool:
    """
    if a tensor is the subset of the other
    :param a:
    :param sub:
    :return:
    """
    return uniq(a).issubset(sub)


def eq(a: Tensor, b: Tensor) -> bool:
    """
    if a and b are equal for torch.Tensor
    :param a:
    :param b:
    :return:
    """
    return torch.eq(a, b).all()


def simplex(t: Tensor, axis=1) -> bool:
    """
    check if the matrix is the probability distribution
    :param t:
    :param axis:
    :return:
    """
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones, rtol=1e-4, atol=1e-4)


def one_hot(t: Tensor, axis=1) -> bool:
    """
    check if the Tensor is one hot.
    The tensor shape can be float or int or others.
    :param t:
    :param axis: default = 1
    :return: bool
    """
    return simplex(t, axis) and sset(t, [0, 1])


def intersection(a: Tensor, b: Tensor) -> Tensor:
    assert a.shape == b.shape
    assert a.dtype == torch.int, a.dtype
    assert b.dtype == torch.int, b.dtype
    assert sset(a, [0, 1])
    assert sset(b, [0, 1])
    return a & b


def union(a: Tensor, b: Tensor) -> Tensor:
    assert a.shape == b.shape
    assert sset(a, [0, 1])
    assert sset(b, [0, 1])
    return a | b


def probs2class(probs: Tensor, class_dim: int = 1) -> Tensor:
    assert simplex(probs, axis=class_dim)
    res = probs.argmax(dim=class_dim)
    return res


def class2one_hot(seg: Tensor, C: int, class_dim: int = 1) -> Tensor:
    """
    make segmentaton mask to be onehot
    """
    if len(seg.shape) == 2:  # Only w, h, used by the dataloader
        seg = seg.unsqueeze(dim=0)
    assert sset(seg, list(range(C)))

    b, *wh = seg.shape  # type:  Tuple[int, int, int]

    res: Tensor = torch.stack([seg == c for c in range(C)], dim=class_dim).type(
        torch.long
    )
    assert one_hot(res, axis=class_dim)
    return res


def probs2one_hot(probs: Tensor, class_dim: int = 1) -> Tensor:
    C = probs.shape[class_dim]
    assert simplex(probs, axis=class_dim)
    res = class2one_hot(probs2class(probs, class_dim=class_dim), C, class_dim=class_dim)
    assert res.shape == probs.shape
    assert one_hot(res, class_dim)
    return res


def logit2one_hot(logit: Tensor, class_dim: int = 1) -> Tensor:
    probs = F.softmax(logit, class_dim)
    return probs2one_hot(probs, class_dim)


# functions
def map_(fn: Callable[[A], B], iter: Iterable[A]) -> List[B]:
    return list(map(fn, iter))


def mmap_(fn: Callable[[A], B], iter: Iterable[A]) -> List[B]:
    with Pool() as pool:
        return list(pool.map(fn, iter))


def uc_(fn: Callable) -> Callable:
    return partial(uncurry, fn)


def uncurry(fn: Callable, args: List[Any]) -> Any:
    return fn(*args)


def id_(x):
    return x


def assert_list(func: Callable[[A], bool], Iters: Iterable) -> bool:
    """
    List comprehensive assert for a function and a list of iterables.
    >>> assert assert_list(simplex, [torch.randn(2,10)]*10)
    :param func: assert function
    :param Iters:
    :return:
    """
    return reduce(and_, [func(x) for x in Iters])
