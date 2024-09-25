import time
import traceback
from typing import Callable, List, Tuple

from other_libs.log import get_logger


def retry_operation(
    operation: Callable, retry_times: int, retry_sleep_time: float, *args, **kwargs
) -> Tuple[bool, None]:
    """
        Attempt a specific operation until it succeeds or a max nbr of retries is reached
    """
    for _ in range(retry_times):
        try:
            res = operation(*args, **kwargs)
            if res[0]:
                return res
        except Exception as e:
            get_logger().error("catch exception:%s", e)
            get_logger().error("traceback:%s", traceback.format_exc())
        get_logger().error("operation failed,retry after %s seconds", retry_sleep_time)
        time.sleep(retry_sleep_time)
    return False, None


def readlines(file_path: str) -> List[str]:
    """
        Read all lines form a file and return them as a list of strings
    """
    lines = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines += f.readlines()
    except Exception:
        with open(file_path, "r", encoding="utf-8-sig") as f:
            lines += f.readlines()
    return lines
