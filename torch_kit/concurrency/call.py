import asyncio
import inspect
import traceback
from typing import Any, Callable

from other_libs.log import get_logger


def exception_aware_call(fn: Callable, *args: Any, **kwargs: Any) -> Any:
    try:
        if inspect.iscoroutinefunction(fn):
            return asyncio.run(fn(*args, **kwargs))
        return fn(*args, **kwargs)
    # pylint: disable=broad-exception-caught
    except Exception as e:
        get_logger().error("catch exception:%s", e)
        get_logger().error("traceback:%s", traceback.format_exc())
    return None
