import concurrent.futures

from other_libs.log import get_logger_setting

from .executor_pool import ExecutorPool
from .process_context import ProcessContext
from .process_initialization import default_initializer, reinitialize_logger


class ProcessPool(ExecutorPool):
    def __init__(
        self, initializer=None, initargs=None, use_logger: bool = True, **kwargs
    ) -> None:
        real_initarg: dict = {}
        real_initarg["initializers"] = [initializer]
        real_initarg["initargs_list"] = [{} if initargs is None else initargs]
        if use_logger:
            real_initarg["initializers"].insert(0, reinitialize_logger)
            real_initarg["initargs_list"].insert(
                0, {"logger_setting": get_logger_setting()}
            )
        if "mp_context" not in kwargs:
            kwargs["mp_context"] = ProcessContext().get_ctx()

        super().__init__(
            concurrent.futures.ProcessPoolExecutor(
                initializer=default_initializer, initargs=(real_initarg,), **kwargs
            )
        )
