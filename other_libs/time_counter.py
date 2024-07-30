import time
import traceback

from other_libs.log import get_logger


class TimeCounter:
    def __init__(self, debug_logging: bool = True, log_prefix: str | None = None):
        self.__debug_logging = debug_logging
        self.__log_prefix = log_prefix
        self.__start_ns: int | None = None
        self.reset_start_time()

    def reset_start_time(self) -> None:
        self.__start_ns = time.monotonic_ns()

    def elapsed_milliseconds(self) -> float:
        assert self.__start_ns is not None
        return (time.monotonic_ns() - self.__start_ns) / 1000000

    def __enter__(self):
        self.reset_start_time()
        return self

    def __exit__(self, exc_type, exc_value, real_traceback):
        if real_traceback:
            return
        used_ms = self.elapsed_milliseconds()
        with_block = traceback.extract_stack(limit=2)[0]

        if self.__debug_logging:
            logging_func = get_logger().debug
        else:
            logging_func = get_logger().info
        logging_func(
            "%s [%s => %d] uses %s ms",
            self.__log_prefix if self.__log_prefix is not None else "",
            with_block.filename,
            with_block.lineno,
            used_ms,
        )
