import json
import os
import re

import torch
import tempfile
import shutil
import portalocker
import uuid

from ..ml_type import MachineLearningPhase
from .metric_visualizer import MetricVisualizer
from other_libs.log import log_warning


class PerformanceMetricRecorder(MetricVisualizer):
    def _after_epoch(self, executor, epoch, **kwargs) -> None:

        prefix = re.sub(r"[: ,]+$", "", self._prefix)
        prefix = re.sub(r"[: ,]+", "_", prefix)

        assert self._data_dir is not None
        #print(f"[DEBUG] self._data_dir = {self._data_dir}")
        #print(f"[DEBUG] prefix = {prefix}")
        #print(f"[DEBUG] phase = {executor.phase}")

        # Build full file path
        json_filename = os.path.join(
            self._data_dir, prefix, str(executor.phase), "performance_metric.json"
        )
        if os.name == 'nt':
            json_filename = '\\\\?\\' + os.path.abspath(json_filename)

        json_dir = os.path.dirname(json_filename)
        os.makedirs(json_dir, exist_ok=True)
        assert os.path.isdir(json_dir), f"Directory creation failed: {json_dir}"

        #print(f"[DEBUG] Will save metrics to: {json_filename}")

        # Try loading previous record
        json_record = {}
        if os.path.isfile(json_filename):
            try:
                with open(json_filename, "rt", encoding="utf8") as f:
                    content = f.read().strip()
                    if content:
                        json_record = json.loads(content)
            except json.JSONDecodeError as e:
                log_warning(f"JSON decode failed for '{json_filename}': {e}. Starting fresh.")
                shutil.copy(json_filename, json_filename + ".corrupted")
                json_record = {}

        epoch_metrics = executor.performance_metric.get_epoch_metrics(epoch)
        if not epoch_metrics and executor.phase != MachineLearningPhase.Training:
            epoch_metrics = executor.performance_metric.get_epoch_metrics(epoch=1)
        if not epoch_metrics:
            return
        for k, value in epoch_metrics.items():
            if k not in json_record:
                json_record[k] = {}
            if value is None:
                continue
            if isinstance(value, torch.Tensor):
                value = value.item()
            json_record[k][epoch] = value

        #with open(json_filename, "w", encoding="utf8") as f:
        #    json.dump(json_record, f)

        # Save JSON atomically to avoid corruption
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf8") as tmp_f:
            json.dump(json_record, tmp_f)
            temp_name = tmp_f.name

        shutil.move(temp_name, json_filename)