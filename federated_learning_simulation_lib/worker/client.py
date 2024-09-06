from typing import Any

import gevent
import torch
from other_libs.topology import ClientEndpoint

from ..executor import ExecutorContext
from .protocol import WorkerProtocol


class ClientMixin(WorkerProtocol):
    """
        provide functionalities in client-server architecture
    """

    def _send_data_to_server(self, data: Any) -> None:
        # assert correct type of endpoint
        assert isinstance(self.endpoint, ClientEndpoint)
        # use the endpoint method to send data to the server
        self.endpoint.send(data)
        # clear the memory cache to free up GPU space
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _get_data_from_server(self) -> Any:
        # assert correct type of endpoint
        assert isinstance(self.endpoint, ClientEndpoint)
        # pause the current process
        self.pause()
        # release any locks or resources held by the current executor context
        ExecutorContext.release()
        # continuously check if there is data available
        while not self.endpoint.has_data():
            gevent.sleep(0.1)
        # once data available, require the executor context
        ExecutorContext.acquire(self.name)
        # retrieve and return data from the server
        return self.endpoint.get()
