from typing import Any, Mapping, MutableMapping

import torch
from torch_kit import ModelParameter

from ..config import DistributedTrainingConfig
from ..message import Message, ParameterMessage


class AggregationAlgorithm:
    def __init__(self) -> None:
        self._all_worker_data: MutableMapping[int, Message] = {}
        # track workers whose data was not received
        self.__skipped_workers: set[int] = set()
        # previous model parameters
        self._old_parameter: ModelParameter | None = None
        # hold the distributed config
        self._config: DistributedTrainingConfig | None = None

    def set_old_parameter(self, old_parameter: ModelParameter) -> None:
        self._old_parameter = old_parameter

    @property
    def config(self) -> DistributedTrainingConfig:
        assert self._config is not None
        return self._config

    def set_config(self, config: DistributedTrainingConfig) -> None:
        self._config = config

    @classmethod
    def get_total_weight(cls, data_dict: Mapping[int, Message]) -> float:
        """
            calculate the total weight from a dict of worker Messages
        """
        total_weight: float = 0
        for v in data_dict.values():
            assert v.aggregation_weight is not None
            total_weight += v.aggregation_weight
        return total_weight

    @classmethod
    def get_ratios(cls, data_dict: Mapping[int, Message]) -> dict[int, float]:
        """
            compute the ratio of each worker's aggregation weight
            relative to the total weight
            return a dict
        """
        total_weight: float = float(cls.get_total_weight(data_dict=data_dict))
        ratios = {}
        for k, v in data_dict.items():
            assert v.aggregation_weight is not None
            ratios[k] = float(v.aggregation_weight) / total_weight
        return ratios

    @classmethod
    def weighted_avg(
        cls,
        data_dict: Mapping[int, ParameterMessage],
        weights: dict[int, float] | float,
    ) -> ModelParameter:
        """
            calculate the weighted average of model parameters from workers
        """
        assert data_dict
        # initialize
        avg_data: ModelParameter = {}
        # iterate over the ParameterMessage objs of workers
        for worker_id, v in data_dict.items():
            # retrieve the corresponding worker weight
            if isinstance(weights, dict):
                weight = weights[worker_id]
            else:
                weight = weights
            assert 0 <= weight <= 1
            # type check for the value
            assert isinstance(v, ParameterMessage)
            # new dict, each param is scaled by the worker weight
            d = {
                k2: v2.to(dtype=torch.float64) * weight
                for (k2, v2) in v.parameter.items()
            }
            # if avg_data is empty, initialize it with d
            if not avg_data:
                avg_data = d
            # if not, add the worker weighted model to the existing avg model
            else:
                for k in avg_data:
                    avg_data[k] += d[k]
        # ensure that there are no NaN value in the average model
        for p in avg_data.values():
            assert not p.isnan().any().cpu()
        # return the dict of averaged model params
        return avg_data

    @classmethod
    def weighted_avg_for_scalar(
        cls,
        data_dict: MutableMapping[int, Message],
        weights: dict[int, float] | float,
        scalar_key: str,
    ) -> float:
        """
            calculate the weighted average of scaler value (metrics) from workers
        """
        assert data_dict
        result: float = 0
        for worker_id, v in data_dict.items():
            if isinstance(weights, dict):
                weight = weights[worker_id]
            else:
                weight = weights
            assert 0 <= weight <= 1
            # add the weighted scalar value to the result
            result += v.other_data[scalar_key] * weight
        return result

    def process_worker_data(
        self,
        worker_id: int,
        worker_data: Message | None,
    ) -> bool:
        if worker_data is None:
            # if no data from a worker, skip its data
            self.__skipped_workers.add(worker_id)
            return True
        # store valid data
        self._all_worker_data[worker_id] = worker_data
        return True

    def aggregate_worker_data(self) -> Any:
        raise NotImplementedError()

    def clear_worker_data(self) -> None:
        self._all_worker_data.clear()
        self.__skipped_workers.clear()

    def exit(self) -> None:
        pass

