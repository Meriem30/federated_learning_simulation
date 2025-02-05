from other_libs.log import log_info

from ..ml_type import MachineLearningPhase
from .metric_visualizer import MetricVisualizer


class PerformanceMetricLogger(MetricVisualizer):
    def _after_epoch(self, executor, epoch, **kwargs) -> None:
        performance_metric = executor.performance_metric

        epoch_metrics = performance_metric.get_epoch_metrics(epoch)

        if not epoch_metrics:
            return
        metric_str: str = ""
        for k, value in epoch_metrics.items():
            if "F1" in k or "AUROC" in k:
                metric_str = metric_str + f"{k}:{value:.4f}, "
            elif "accuracy" in k:
                metric_str = metric_str + f"{k}:{value:.2%}, "
            elif "loss" in k:
                metric_str = metric_str + f"{k}:{value:.5f}, "
            elif k == "duration":
                metric_str = metric_str + f"in {value:.3f} seconds, "
            elif k == "data_waiting_time":
                metric_str = metric_str + f"data loader uses {value:.3f} seconds, "
            else:
                metric_str = metric_str + f"{k}:{value}, "
        metric_str = metric_str[:-2]
        if executor.phase == MachineLearningPhase.Training:
            log_info(
                "%sepoch: %s, %s %s",
                self.prefix + " " if self.prefix else "",
                epoch,
                str(executor.phase),
                metric_str,
            )
        else:
            assert epoch == 1
            log_info(
                "%s%s %s",
                self.prefix + " " if self.prefix else "",
                str(executor.phase),
                metric_str,
            )

