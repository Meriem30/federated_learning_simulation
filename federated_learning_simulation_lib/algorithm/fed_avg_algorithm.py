from typing import Any, MutableMapping

import torch
from other_libs.log import log_error
from torch_kit import ModelParameter

from ..message import Message, ParameterMessage
from .aggregation_algorithm import AggregationAlgorithm


class FedAVGAlgorithm(AggregationAlgorithm):
    """
        inherit AggregationAlgorithm and implement the FedAvg algo
    """
    def __init__(self) -> None:
        super().__init__()
        # whether to accumulate the updates from the workers or compute the average directly
        self.accumulate: bool = True
        # whether to aggregate loss information
        self.aggregate_loss: bool = False
        # dict to store total weights for each model
        self.__total_weights: dict[str, float] = {}
        # store accumulated parameter  updates
        self.__parameter: ModelParameter = {}

    def process_worker_data(
        self,
        worker_id: int,
        worker_data: Message | None,
    ) -> bool:
        """
            add more functionalities to the parent method
        """
        # initialize the parent method for processing data
        res = super().process_worker_data(worker_id=worker_id, worker_data=worker_data)
        if not res:
            return False
        if not self.accumulate:
            # if accumulate is not required, the method returns early
            return True
        worker_data = self._all_worker_data.get(worker_id, None)
        if worker_data is None:
            return True
        if not isinstance(worker_data, ParameterMessage):
            return True
        # if the all worker data is not None
        for k, v in worker_data.parameter.items():
            # iterate over each parameter update from the worker and apply the weight
            assert not v.isnan().any().cpu()
            # retrieve the weight associated with the param v
            weight = self._get_weight(worker_data, name=k, parameter=v)
            # store the weighted parameter
            tmp = v.to(dtype=torch.float64) * weight
            if k not in self.__parameter:
                self.__parameter[k] = tmp
            else:
                # if the param key already exists, add the weighted update value
                self.__parameter[k] += tmp
            #
            if k not in self.__total_weights:
                # similarly, add or initialize the weight for the param key
                self.__total_weights[k] = weight
            else:
                self.__total_weights[k] += weight
        # release to reduce memory pressure
        worker_data.parameter = {}
        return True

    def _get_weight(
        self, worker_data: ParameterMessage, name: str, parameter: Any
    ) -> Any:
        return worker_data.aggregation_weight

    def _apply_total_weight(
        self, name: str, parameter: torch.Tensor, total_weight: Any
    ) -> torch.Tensor:
        """
            device the parameter tensor by the total weight to compute average
        """
        return parameter / total_weight

    def aggregate_worker_data(self) -> ParameterMessage:
        if not self.accumulate:
            # if not required, directly aggregate
            parameter = self.aggregate_parameter(self._all_worker_data)
        else:
            # if required, process accumulated parameter & normalize them using total weight
            assert self.__parameter
            parameter = self.__parameter
            self.__parameter = {}
            for k, v in parameter.items():
                assert not v.isnan().any().cpu()
                parameter[k] = self._apply_total_weight(
                    name=k, parameter=v, total_weight=self.__total_weights[k]
                )
                assert not parameter[k].isnan().any().cpu()
            self.__total_weights = {}
        other_data: dict[str, Any] = {}
        if self.aggregate_loss:
            # if true, compute aggregated loss values
            other_data |= self.__aggregate_loss(self._all_worker_data)
        # ensure consistency
        other_data |= self.__check_and_reduce_other_data(self._all_worker_data)
        # return the aggregated params and other data
        return ParameterMessage(
            parameter=parameter,
            end_training=next(iter(self._all_worker_data.values())).end_training,
            in_round=next(iter(self._all_worker_data.values())).in_round,
            other_data=other_data,
        )

    @classmethod
    def aggregate_parameter(
        cls, all_worker_data: MutableMapping[int, Any]
    ) -> ModelParameter:
        """
            aggregate (compute the weighted average) parameters from all worker
            use weighted_avg parent method
        """
        assert all_worker_data
        assert all(
            isinstance(parameter, ParameterMessage)
            for parameter in all_worker_data.values()
        )
        parameter = AggregationAlgorithm.weighted_avg(
            all_worker_data,
            AggregationAlgorithm.get_ratios(all_worker_data),
        )
        assert parameter
        return parameter

    @classmethod
    def __aggregate_loss(cls, all_worker_data: MutableMapping[int, Message]) -> dict:
        """
            compute the weighted average for training and validation loss values
            from worker data
        """
        assert all_worker_data
        loss_dict = {}
        for worker_data in all_worker_data.values():
            for loss_type in ("training_loss", "validation_loss"):
                if loss_type in worker_data.other_data:
                    loss_dict[loss_type] = AggregationAlgorithm.weighted_avg_for_scalar(
                        all_worker_data,
                        AggregationAlgorithm.get_ratios(all_worker_data),
                        scalar_key=loss_type,
                    )
            break
        assert loss_dict
        for worker_data in all_worker_data.values():
            for loss_type in ("training_loss", "validation_loss"):
                # remove loss data from other_data
                worker_data.other_data.pop(loss_type, None)
        return loss_dict

    @classmethod
    def __check_and_reduce_other_data(
        cls, all_worker_data: MutableMapping[int, Message]
    ) -> dict:
        """
            check other_data fields across all worker data are consistent (no discrepancies)
        """
        result: dict = {}
        for worker_data in all_worker_data.values():
            for k, v in worker_data.other_data.items():
                if k not in result:
                    result[k] = v
                    continue
                if v != result[k]:
                    log_error("different values on key %s", k)
                    raise RuntimeError(f"different values on key {k}")
        return result
