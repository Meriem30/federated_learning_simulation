from typing import Any, Iterable

from .task_queue import TaskQueue


def batch_process(queue: TaskQueue, tasks: Iterable[Any]) -> dict:
    assert not queue.has_data()
    result: dict = {}
    cnt = 0
    for task in tasks:
        queue.add_task(task)
        cnt += 1
        while queue.has_data():
            cnt -= 1
            data = queue.get_data()
            assert data is not None
            assert isinstance(data[0], dict)
            result |= data[0]
    while cnt > 0:
        data = queue.get_data()
        assert data is not None
        assert isinstance(data[0], dict)
        result |= data[0]
        cnt -= 1
    return result
