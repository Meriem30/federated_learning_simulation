import json
import os
from typing import Any

from other_libs.log import log_info

from ..message import ParameterMessage
from .protocol import AggregationServerProtocol


class PerformanceMixin(AggregationServerProtocol):
    """
        extend the AggregationServerProtocol
        include recording and analyzing performance statistics
    """
    def __init__(self) -> None:
        super().__init__()
        # initialize the performance tracking attributes
        self.__stat: dict = {}
        # to track round indices
        self.__keys: list = []
        # counter to track how many rounds the model performance had plateaued
        self.__plateau: int = 0
        # the acc difference threshold to determine improvement
        self.__acc_diff = 0.001
        # the maw num of plateau rounds before considering the model to be converged
        self.__max_plateau: int = 5

    @property
    def performance_stat(self) -> dict:
        return self.__stat

    def _set_plateau_limit(self, max_plateau: int) -> None:
        self.__max_plateau = max_plateau

    def _set_accurary_difference(self, acc_diff: float) -> None:
        self.__acc_diff = acc_diff

    def _get_stat_key(self, message: ParameterMessage) -> Any:
        # determine the key for storing statics
        if message.is_initial:
            return 0
        return self.round_index

    def _set_stat(self, key: str, value: Any, message: ParameterMessage) -> None:
        # set a static value (round nbr) for a specific key (metric)
        stat_key = self._get_stat_key(message=message)
        if stat_key not in self.__keys:
            self.__keys.append(stat_key)
        # add the value of the corresponding key in a specific round
        self.__stat[stat_key][key] = value

    def record_performance_statistics(
        self,
        message: ParameterMessage,
    ) -> None:
        """
            record the performance statics for a given message (model params)
        """
        # retrieve the performance metrics
        metric = self.get_metric(
            message.parameter, log_performance_metric=(not message.is_initial)
        )
        # store the metrics in a dict prefixed with "test"
        round_stat = {f"test_{k}": v for k, v in metric.items()}
        # add the stat for the corresponding round nbr
        key = self._get_stat_key(message)
        assert key not in self.__stat
        self.__keys.append(key)
        self.__stat[key] = round_stat
        # save the stat to a json file
        round_record_path = os.path.join(self.save_dir, "round_record.json")
        os.makedirs(os.path.dirname(round_record_path), exist_ok=True)
        with open(
            round_record_path,
            "wt",
            encoding="utf8",
        ) as f:
            json.dump(self.__stat, f)

    def get_test_accuracies(self) -> list[float]:
        """
            retrieve a list of test accuracies from the performance statistics
        """
        return [self.performance_stat[k]["test_accuracy"] for k in self.__keys]

    def convergent(self) -> bool:
        """
            determine if the model training has converged
            based on test accuracies
        """
        if len(self.performance_stat) < 2:
            return False
        test_accuracies = self.get_test_accuracies()
        # retrieve the historical max acc (excluding the most recent round)
        historical_max_acc = max(test_accuracies[0:-1])
        # return false if the latest test acc exceeds the historical data by more than the threshold
        if test_accuracies[-1] > historical_max_acc + self.__acc_diff:
            self.__plateau = 0
            return False
        # if not, increment __plateau
        self.__plateau += 1
        # log
        log_info(
            "historical_max_acc is %s diff is %s, plateau is %s",
            historical_max_acc,
            historical_max_acc - test_accuracies[-1],
            self.__plateau,
        )
        # return true if the __plateau is equal the max threshold 
        return self.__plateau >= self.__max_plateau
