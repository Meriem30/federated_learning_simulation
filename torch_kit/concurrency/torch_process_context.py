from typing import Any

import torch.multiprocessing
from other_libs.concurrency import ProcessContext


class TorchProcessContext(ProcessContext):
    def __init__(self, **kwargs: Any) -> None:
        ctx = torch.multiprocessing
        if torch.cuda.is_available():
            ctx = torch.multiprocessing.get_context("spawn")
        super().__init__(ctx=ctx, **kwargs)