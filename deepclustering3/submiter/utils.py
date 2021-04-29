import random
import string

import numpy as np

norm_account = ["def-chdesa", "def-mpederso"]
prio_account = ["rrg-mpederso", ]


def randomString(stringLength=8):
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for i in range(stringLength))


def random_account(include_norm=True, include_pri=True):
    account_list = []
    if include_norm:
        account_list.extend(norm_account)
    if include_pri:
        account_list.extend(prio_account)
    assert len(account_list) > 0, account_list
    while True:
        yield from np.random.permutation(account_list)
