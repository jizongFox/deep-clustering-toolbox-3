from copy import deepcopy as dcopy
from typing import Dict, Any

from deepclustering3.types import mapType, is_map


def dictionary_merge_by_hierachy(dictionary1: Dict[str, Any], dictionary2: Dict[str, Any] = None, deepcopy=True,
                                 hook_after_merge=None):
    """
    Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
    updating only top-level keys, dict_merge recurses down into dicts nested
    to an arbitrary depth, updating keys. The ``merge_dct`` is merged into``dct``.
    :return: None
    """
    if deepcopy:
        dictionary1, dictionary2 = dcopy(dictionary1), dcopy(dictionary2)
    if dictionary2 is None:
        return dictionary1
    for k, v in dictionary2.items():
        if k in dictionary1 and isinstance(dictionary1[k], mapType) and isinstance(dictionary2[k], mapType):
            dictionary1[k] = dictionary_merge_by_hierachy(dictionary1[k], dictionary2[k], deepcopy=False)
        else:
            dictionary1[k] = dictionary2[k]
    if hook_after_merge:
        dictionary1 = hook_after_merge(dictionary1)
    return dictionary1


def remove_dictionary_callback(dictionary):
    new_dictionary = dcopy(dictionary)
    for k, v in dictionary.items():
        if isinstance(v, mapType):
            new_dictionary[k] = remove_dictionary_callback(v)
        try:
            if v.lower() == "remove":
                del new_dictionary[k]
        except AttributeError:
            pass
    return new_dictionary


def extract_dictionary_from_large(large_dictionary, small_dictionary, deepcopy=True):
    if deepcopy:
        small_dictionary = dcopy(small_dictionary)

    for k, v in small_dictionary.items():
        if k in large_dictionary:
            if not isinstance(v, mapType):
                small_dictionary[k] = large_dictionary[k]
            else:
                small_dictionary[k] = extract_dictionary_from_large(large_dictionary[k], small_dictionary[k])
    return small_dictionary


def flatten_dict(dictionary, parent_key="", sep="_"):
    items = []
    for k, v in dictionary.items():
        new_key = parent_key + sep + k if parent_key else k
        if is_map(v):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
