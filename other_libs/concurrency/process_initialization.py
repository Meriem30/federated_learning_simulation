import threading
from typing import Any

from other_libs.log import apply_logger_setting

__local_data: threading.local = threading.local()
__local_data.data = {}


def reinitialize_logger(logger_setting: dict, **kwargs: Any) -> None:
    apply_logger_setting(logger_setting)


def default_initializer(init_arg_dict) -> None:
    # initialize a process with a given number
    # save fun_kwargs for further processing and call the initialization function
    for initializer, init_args in zip(
        init_arg_dict["initializers"], init_arg_dict["initargs_list"]
    ):
        if "process_data" in init_args:
            __local_data.data.update(init_args.pop("process_data"))
        if initializer is None:
            continue
        initializer(**init_args)


def get_process_data() -> dict:
    return __local_data.data