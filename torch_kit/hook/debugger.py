from typing import Any

import torch
import torch.autograd
from other_libs.log import log_warning

from ..ml_type import MachineLearningPhase
from . import Hook


class GradientSanitizer(Hook):
    def _before_batch(self, executor, batch_index, **kwargs):
        if executor.phase != MachineLearningPhase.Training:
            return
        if batch_index % 100 != 0:
            return
        # check parameters can be updated
        trainer = executor
        optimizer = trainer.get_optimizer()
        for name, parameter in trainer.model.named_parameters():
            flag = False
            for group in optimizer.param_groups:
                if flag:
                    break
                for param in group["params"]:
                    if param is parameter:
                        flag = True
                        break
            if not flag:
                raise RuntimeError("can't find parameter " + name + " in the optimizer")


class Debugger(Hook):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.gradient_sanitizer = GradientSanitizer()

    def _before_execute(self, executor, **kwargs: Any) -> None:
        torch.autograd.set_detect_anomaly(True)
        log_warning("model executor in debugging mode")

    def _after_execute(self, **kwargs: Any) -> None:
        torch.autograd.set_detect_anomaly(False)