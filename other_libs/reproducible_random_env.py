import copy
import os
import pickle
import random
import threading
from typing import Any

try:
    import numpy as np

    has_np = True
except ImportError:
    has_np = False

from other_libs.log import get_logger


class ReproducibleRandomEnv:
    """
        Save and restore the state of random number generators
    """
    lock = threading.RLock()

    def __init__(self) -> None:
        self.__randomlib_state: Any = None
        self.__numpy_state: Any = None
        self._enabled: bool = False
        self.__last_seed_path: None | str = None

    @property
    def enabled(self):
        return self._enabled

    @property
    def last_seed_path(self):
        return self.__last_seed_path

    def enable(self) -> None:
        with self.lock:
            if self._enabled:
                get_logger().warning("%s use reproducible env", id(self))
            else:
                get_logger().warning("%s initialize and use reproducible env", id(self))

            if self.__randomlib_state is not None:
                get_logger().debug("overwrite random lib state")
                random.setstate(self.__randomlib_state)
            else:
                get_logger().debug("get random lib state")
                self.__randomlib_state = random.getstate()

            if has_np:
                if self.__numpy_state is not None:
                    get_logger().debug("overwrite numpy random lib state")
                    np.random.set_state(copy.deepcopy(self.__numpy_state))
                else:
                    get_logger().debug("get numpy random lib state")
                    self.__numpy_state = np.random.get_state()
            self._enabled = True

    def disable(self) -> None:
        get_logger().warning("disable reproducible env")
        with self.lock:
            self._enabled = False

    def __enter__(self):
        self.enable()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if traceback:
            return
        self.disable()

    def get_state(self) -> dict:
        return {
            "randomlib_state": self.__randomlib_state,
            "numpy_state": self.__numpy_state,
        }

    def save(self, seed_dir: str) -> Any:
        seed_path = os.path.join(seed_dir, "random_seed.pk")
        get_logger().warning("%s save reproducible env to %s", id(self), seed_path)
        with self.lock:
            assert self._enabled
            os.makedirs(seed_dir, exist_ok=True)
            self.__last_seed_path = seed_path
            with open(seed_path, "wb") as f:
                return pickle.dump(
                    self.get_state(),
                    f,
                )

    def load_state(self, state: dict) -> None:
        self.__randomlib_state = state["randomlib_state"]
        self.__numpy_state = state["numpy_state"]

    def load(self, path: str | None = None, seed_dir: str | None = None) -> None:
        if path is None:
            assert seed_dir is not None
            path = os.path.join(seed_dir, "random_seed.pk")
        with self.lock:
            assert not self._enabled
            with open(path, "rb") as f:
                get_logger().warning("%s load reproducible env from %s", id(self), path)
                self.load_state(pickle.load(f))

    def load_last_seed(self) -> None:
        self.load(self.last_seed_path)