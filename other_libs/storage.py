import hashlib
import os
import pickle
import tempfile
import time
from enum import IntEnum, auto
from typing import Any, Callable


class DataLocation(IntEnum):
    NoData = auto()
    Memory = auto()
    Disk = auto()


class DataStorage:
    """
        Encapsulating data storage and caching operations
        Efficiently retrieve and manage this data
    """

    def __init__(self, data: Any = None, data_path: str | None = None):
        self.__data: Any = data
        self.__data_path: str | None = data_path
        self.__data_hash: str | None = None
        self.__data_location: DataLocation = DataLocation.NoData
        if data is not None:
            self.__data_location = DataLocation.Memory
        elif data_path is not None:
            self.__data_location = DataLocation.Disk
        self.__fd: int | None = None
        self.__use_tmp_file: bool = False

    def has_data(self) -> bool:
        return self.__data_location != DataLocation.NoData

    def set_data_path(self, data_path: str) -> None:
        if self.__data_path == data_path:
            return
        assert self.__data_location != DataLocation.Disk
        self.__data_path = data_path
        self.__use_tmp_file = False

    def set_data(self, data: Any) -> None:
        self.__data = data
        self.mark_new_data()

    def mark_new_data(self) -> None:
        self.__data_hash = None
        self.__data_location = DataLocation.Memory

    @property
    def data_path(self) -> str | None:
        return self.__data_path

    @property
    def data(self) -> Any:
        if self.__data_location == DataLocation.Disk:
            self.__data = self.__load_data()
        return self.__data

    def __load_data(self) -> Any:
        assert self.__data_path
        with open(self.__data_path, "rb") as f:
            return pickle.load(f)

    def __close_data_file(self) -> None:
        if self.__data_path is not None:
            if self.__fd is not None:
                os.close(self.__fd)
            self.__fd = None

    def __remove_data_file(self) -> None:
        if self.__data_path is not None:
            if os.path.isfile(self.__data_path):
                self.__close_data_file()
                os.remove(self.__data_path)
            self.__data_path = None

    def __getitem__(self, key) -> Any:
        return self.data[key]

    def __contains__(self, key) -> bool:
        return key in self.data

    def __del__(self):
        if self.__use_tmp_file:
            self.__remove_data_file()

    @property
    def data_hash(self) -> str:
        if self.__data_hash is not None:
            return self.__data_hash
        hash_sha256 = hashlib.sha256()
        hash_sha256.update(pickle.dumps(self.data))
        self.__data_hash = hash_sha256.hexdigest()
        return self.__data_hash

    def clear(self) -> None:
        self.__remove_data_file()
        self.__data = None
        self.__data_hash = None
        self.__data_path = None
        self.__data_location = DataLocation.NoData

    def save(self) -> None:
        if self.__data_location == DataLocation.Memory:
            if self.__data_path is None:
                self.__fd, self.__data_path = tempfile.mkstemp()
                self.__use_tmp_file = True
            else:
                os.makedirs(os.path.dirname(self.__data_path), exist_ok=True)
            assert self.__data_path is not None
            with open(self.__data_path, "wb") as f:
                pickle.dump(self.__data, f)
                self.__data = None
                self.__data_location = DataLocation.Disk

    def save_in(self, path) -> None:
        assert path is not None
        with open(path, "wb") as f:
            pickle.dump(self.__data, f)

def persistent_cache(
    path: str | None = None, cache_time: float | None = None
) -> Callable:
    def read_data(path: str) -> Any:
        if not os.path.isfile(path):
            return None
        if cache_time is not None and time.time() > cache_time + os.path.getmtime(path):
            return None
        fd = os.open(path, flags=os.O_RDONLY)
        with os.fdopen(fd, "rb") as f:
            res = pickle.load(f)
        return res

    def write_data(data: Any, path: str) -> None:
        if os.path.isfile(path):
            os.remove(path)
        fd = os.open(path, flags=os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "wb") as f:
            pickle.dump(data, f)

    def wrap(fun: Callable) -> Callable:
        def wrap2(*args, **kwargs):
            cache_path = path
            if cache_path is None:
                cache_path = kwargs.get("cache_path", None)
            assert cache_path is not None
            hash_sha256 = hashlib.sha256()
            if args:
                hash_sha256.update(pickle.dumps(args))
            else:
                hash_sha256.update(pickle.dumps([]))
            if kwargs:
                hash_sha256.update(pickle.dumps(kwargs))
            else:
                hash_sha256.update(pickle.dumps({}))
            if os.path.isfile(cache_path):
                os.remove(cache_path)
            os.makedirs(cache_path, exist_ok=True)
            new_path = os.path.join(cache_path, f"{hash_sha256.hexdigest()}")
            data = read_data(new_path)
            if data is not None:
                return data
            data = fun(*args, **kwargs)
            write_data(data, new_path)
            return data

        return wrap2

    return wrap


def get_cached_data(path: str, data_fun: Callable) -> Any:
    return persistent_cache(path=path)(data_fun)()
