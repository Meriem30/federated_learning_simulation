import torch

from ..hook import Hook
from ..ml_type import StopExecutingException


class LRFinder(Hook):

    def __init__(
            self,
            start_lr: float = 1e-7,
            end_lr: float = 10,
            epoch: int = 2,
            stop_div: bool = True,
    ) -> None:
        super().__init__()
        self.lr_getter = lambda idx: start_lr * (end_lr / start_lr) ** idx
        self.epoch: int = epoch
        self.stop_div: bool = stop_div
        self.best_loss: float = float("inf")
        self.losses: list[float] = []
        self.learning_rates: list[float] = []
        self.batch_index: int = 0
        self.total_batch_num: int = 0
        self.suggested_learning_rate: float = 0

    def _before_execute(self, **kwargs) -> None:
        trainer = kwargs["executor"]
        trainer.remove_optimizer()
        trainer.remove_lr_scheduler()
        trainer.hyper_parameter.epoch = self.epoch
        trainer.hyper_parameter.learning_rate = 1
        self.total_batch_num = self.epoch * (
                (trainer.dataset_size + trainer.hyper_parameter.batch_size - 1)
                // trainer.hyper_parameter.batch_size
        )

    def _before_batch(self, **kwargs) -> None:
        trainer = kwargs["executor"]
        learning_rate = self.lr_getter(self.batch_index / (self.total_batch_num - 1))
        self.learning_rates.append(learning_rate)
        optimizer = trainer.get_optimizer()
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

    def _after_batch(self, **kwargs):
        batch_loss = kwargs["result"]["loss"]
        if self.losses:
            batch_loss = batch_loss + 0.98 * (self.losses[-1] - batch_loss)
        self.losses.append(batch_loss)

        self.best_loss = min(batch_loss, self.best_loss)

        stop_training = False
        if batch_loss > 10 * self.best_loss and kwargs["epoch"] > 1 and self.stop_div:
            stop_training = True
        self.batch_index += 1
        if self.batch_index == self.total_batch_num:
            stop_training = True

        if stop_training:
            self.learning_rates = self.learning_rates[self.total_batch_num // 10:]
            self.losses = self.losses[self.total_batch_num // 10:]
            self.suggested_learning_rate = (
                    self.learning_rates[torch.tensor(self.losses).argmin()] / 10.0
            )
            raise StopExecutingException()
