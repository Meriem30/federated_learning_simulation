from collections.abc import Mapping, MutableMapping, Sequence
from typing import Callable, Generator


def get_mapping_items_by_key_order(d: Mapping) -> Generator:
    """
        Return a generator (tuple) giving the items by key order
    """
    for k in sorted(d.keys()):
        yield (k, d[k])


def get_mapping_values_by_key_order(d: Mapping) -> Generator:
    """
        Return a generator giving the values by key order
    """
    for _, v in get_mapping_items_by_key_order(d):
        yield v


def change_mapping_keys(
    d: MutableMapping, f: Callable, recursive: bool = False
) -> MutableMapping:
    """
        Return a new mapping with keys changed according to a given function
        It can change keys recursively if needed
    """
    new_d = type(d)()
    for k, v in d.items():
        if recursive and isinstance(v, MutableMapping):
            v = change_mapping_keys(v, f, recursive)
        new_k = f(k)
        assert new_k not in new_d
        new_d[new_k] = v
    return new_d


def change_mapping_values(d: MutableMapping, key, f: Callable) -> MutableMapping:
    """
        Return a new mapping with its values changed according to a given function
        It can handle nested mapping, lists, and tuples
    """
    match d:
        case MutableMapping():
            new_d = type(d)()
            for k, v in d.items():
                if k == key:
                    v = f(v)
                else:
                    v = change_mapping_values(v, key, f)
                new_d[k] = v
            return new_d
        case list() | tuple():
            return [change_mapping_values(elm, key, f) for elm in d]
    return d


def flatten_mapping(d: Mapping) -> list:
    """
        Return a list with values ordered by keys, flattening nested mapping
    """

    res = []
    for v in get_mapping_values_by_key_order(d):
        if isinstance(v, Mapping):
            res += flatten_mapping(v)
        else:
            res.append(v)
    return res


def reduce_values_by_key(f: Callable, maps: Sequence[dict]) -> dict:
    """
        Apply a reduction function to the values of the same key across multiple mapping
        Return a new dict
    """
    value_seq_dict: dict = {k: [] for k in maps[0]}
    for m in maps:
        for k, v in m.items():
            value_seq_dict[k].append(v)
    # Apply a reduction function of each list of values, creating a new dict
    return {k: f(v) for k, v in value_seq_dict.items()}
