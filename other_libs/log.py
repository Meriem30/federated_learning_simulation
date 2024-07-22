import contextlib
import copy
import logging
import logging.handlers
import os
import threading
from multiprocessing import Manager, Queue, process
from typing import Any

from colorlog import ColoredFormatter


def __set_default_formatter(handler: logging.Handler, with_color: bool = True) -> None:
    if with_color and os.getenv("eink_screen") == "1":
        with_color = False
    format_str: str = "%(asctime)s %(levelname)s {%(processName)s} [%(filename)s => %(lineno)d] : %(message)s"
    if with_color:
        formatter: logging.Formatter = ColoredFormatter(
            "%(log_color)s" + format_str,
            log_colors={
                "DEBUG": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
            style="%",
        )
    else:
        formatter = logging.Formatter(
            format_str,
            style="%",
        )

    handler.setFormatter(formatter)


def __worker(
    qu: Queue, logger: logging.Logger, logger_lock: threading._RLock | None
) -> None:
    while True:
        try:
            record = qu.get()
            if record is None:
                return
            with logger_lock if logger_lock is not None else contextlib.nullcontext():
                logger.handle(record)
        except ValueError:
            return
        except EOFError:
            return
        except OSError:
            return


__logger_lock: threading._RLock | None = None

if not getattr(process.current_process(), "_inheriting", False):
    __logger_lock = Manager().RLock()

__colored_logger: logging.Logger = logging.getLogger("colored_logger")
if not __colored_logger.handlers:
    __colored_logger.setLevel(logging.DEBUG)
    __handler = logging.StreamHandler()
    __set_default_formatter(__handler, with_color=True)
    __colored_logger.addHandler(__handler)
    __colored_logger.propagate = False

__stub_colored_logger = logging.getLogger("colored_multiprocess_logger")
if not __stub_colored_logger.handlers:
    __stub_colored_logger.setLevel(logging.INFO)
    q: Queue = Queue()
    __stub_colored_logger.addHandler(logging.handlers.QueueHandler(q))
    __stub_colored_logger.propagate = False
    __background_thd = threading.Thread(
        target=__worker, args=(q, __colored_logger, __logger_lock), daemon=True
    )
    __background_thd.start()


def add_file_handler(filename: str) -> logging.Handler:
    filename = os.path.normpath(os.path.abspath(filename))
    with __logger_lock if __logger_lock is not None else contextlib.nullcontext():
        for handler in __colored_logger.handlers:
            if (
                isinstance(handler, logging.FileHandler)
                and handler.baseFilename == filename
            ):
                return handler
        log_dir = os.path.dirname(filename)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        handler = logging.FileHandler(filename, mode="wt", encoding="utf8", delay=True)
        __colored_logger.addHandler(handler)
        __set_default_formatter(handler, with_color=False)
        return handler


def reset_file_handler(filename: str) -> logging.Handler:
    filename = os.path.normpath(os.path.abspath(filename))
    with __logger_lock if __logger_lock is not None else contextlib.nullcontext():
        for handler in copy.copy(__colored_logger.handlers):
            if isinstance(handler, logging.FileHandler):
                __colored_logger.removeHandler(handler)
    return add_file_handler(filename=filename)


def set_level(level: Any) -> None:
    with __logger_lock if __logger_lock is not None else contextlib.nullcontext():
        __stub_colored_logger.setLevel(level)


# def set_formatter(formatter) -> None:
#     with __logger_lock:
#         for handler in __colored_logger.handlers:
#             handler.setFormatter(formatter)


def get_logger_setting() -> dict:
    setting: dict = {}
    with __logger_lock if __logger_lock is not None else contextlib.nullcontext():
        setting["level"] = __stub_colored_logger.level
        setting["lock"] = __logger_lock
        setting["handlers"] = []
        for handler in __colored_logger.handlers:
            handler_dict: dict = {"formatter": handler.formatter}
            if isinstance(handler, logging.FileHandler):
                handler_dict["type"] = "file"
                handler_dict["filename"] = handler.baseFilename
            elif isinstance(handler, logging.StreamHandler):
                handler_dict["type"] = "stream"
            else:
                raise NotImplementedError()
            setting["handlers"].append(handler_dict)
    return setting


def apply_logger_setting(setting: dict) -> None:
    global __logger_lock
    __logger_lock = setting["lock"]
    with __logger_lock if __logger_lock is not None else contextlib.nullcontext():
        set_level(setting["level"])
        for handler_info in setting["handlers"]:
            if handler_info["type"] == "stream":
                handler = __colored_logger.handlers[0]
                assert isinstance(handler, logging.StreamHandler)
            elif handler_info["type"] == "file":
                handler = add_file_handler(handler_info["filename"])
            else:
                raise NotImplementedError()
            handler.setFormatter(handler_info["formatter"])


def get_logger() -> logging.Logger:
    return __stub_colored_logger


def log_info(*args, **kwargs) -> None:
    get_logger().info(*args, **kwargs, stacklevel=2)


def log_debug(*args, **kwargs) -> None:
    get_logger().debug(*args, **kwargs, stacklevel=2)


def log_warning(*args, **kwargs) -> None:
    get_logger().warning(*args, **kwargs, stacklevel=2)


def log_error(*args, **kwargs) -> None:
    get_logger().error(*args, **kwargs, stacklevel=2)