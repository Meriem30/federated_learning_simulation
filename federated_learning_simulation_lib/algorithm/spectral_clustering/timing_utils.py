"""
timing_utils.py
===============
Standalone timing utility for GRAIL-FL runtime analysis.

Collects per-round timing for every measurable phase and appends one
JSON object per round to  <save_dir>/timing_record.json   the same
directory where performance stats are written.

Usage (server side):
    from .timing_utils import TimingRecorder, phase_timer

    # in __init__:
    self._timing_recorder = TimingRecorder(self.config.save_dir, self.config.worker_number)

    # measure a block:
    with phase_timer() as t:
        self._update_network()
    graph_update_ms = t.elapsed_ms

    # write at end of round:
    self._timing_recorder.record(round_index, record_dict)

No dependency on any FL framework module  safe to import anywhere.
"""

import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, asdict, field
from typing import Optional
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Context manager for inline timing
# ─────────────────────────────────────────────────────────────────────────────

class _Timer:
    """Holds the elapsed time after __exit__. Access via .elapsed_ms."""
    def __init__(self):
        self.elapsed_ms: float = 0.0
        self._start: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1_000.0


@contextmanager
def phase_timer():
    """
    Inline context-manager timer.

    Example:
        with phase_timer() as t:
            do_something()
        print(t.elapsed_ms)
    """
    t = _Timer()
    with t:
        yield t


# ─────────────────────────────────────────────────────────────────────────────
# Per-round record dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ClientTimingSummary:
    """Summary statistics for a per-client timing metric across the round."""
    mean_ms: float = 0.0
    std_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    # worker_id (int, as str in JSON) → elapsed ms
    per_worker: dict = field(default_factory=dict)

    @classmethod
    def from_worker_dict(cls, timing_dict: dict[int, float]) -> "ClientTimingSummary":
        """
        Build summary from {worker_id: ms} dict.
        Workers with value <= 0 (not measured this round) are excluded
        from the statistics but kept in per_worker for completeness.
        """
        measured = {k: v for k, v in timing_dict.items() if v > 0.0}
        if not measured:
            return cls(per_worker={str(k): v for k, v in timing_dict.items()})
        vals = list(measured.values())
        return cls(
            mean_ms=float(np.mean(vals)),
            std_ms=float(np.std(vals)),
            min_ms=float(np.min(vals)),
            max_ms=float(np.max(vals)),
            per_worker={str(k): round(v, 4) for k, v in timing_dict.items()},
        )


@dataclass
class RoundTimingRecord:
    """
    Complete timing record for one FL round.

    Server-side phases
    ------------------
    mi_matrix_build_ms    : extracting MI values into the numpy client-state
                            matrix (_create_workers_matrix)
    spectral_clustering_ms: eigenvector computation + k-means on U
                            (_perform_clustering)
    aggregation_ms        : FedAvg weighted parameter accumulation +
                            normalisation (aggregate_worker_data minus
                            clustering)
    graph_update_ms       : rebuilding NetworkX graph edges from the new
                            adjacency matrix (_update_network)
    total_server_overhead_ms: sum of the four above (excludes training,
                              which runs on clients)

    Client-side phases (aggregated across selected workers)
    -------------------------------------------------------
    training              : ClientTimingSummary  local SGD
    mi_computation        : ClientTimingSummary  MI inference loop
    """
    round: int = 0
    n_workers_total: int = 0
    n_workers_selected: int = 0
    selected_worker_ids: list = field(default_factory=list)
    variant: str = "grail_fl"   # grail_fl | ablation_no_clustering | ablation_random_selection

    # server-side (ms)
    mi_matrix_build_ms: float = 0.0
    spectral_clustering_ms: float = 0.0
    aggregation_ms: float = 0.0
    graph_update_ms: float = 0.0
    total_server_overhead_ms: float = 0.0

    # client-side
    training: ClientTimingSummary = field(default_factory=ClientTimingSummary)
    mi_computation: ClientTimingSummary = field(default_factory=ClientTimingSummary)

    def finalise(self) -> None:
        """Compute derived total after all fields are set."""
        self.total_server_overhead_ms = round(
            self.mi_matrix_build_ms
            + self.spectral_clustering_ms
            + self.aggregation_ms
            + self.graph_update_ms,
            4,
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        # Round floats for readability
        for key in (
            "mi_matrix_build_ms", "spectral_clustering_ms",
            "aggregation_ms", "graph_update_ms", "total_server_overhead_ms",
        ):
            d[key] = round(d[key], 4)
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Recorder  handles JSON persistence
# ─────────────────────────────────────────────────────────────────────────────

class TimingRecorder:
    """
    Appends one JSON object per round to  <save_dir>/timing_record.json.

    The file is a JSON array: [ {round1}, {round2}, ... ]
    It is read on startup (if exists) so that a resumed experiment appends
    rather than overwrites previous rounds.

    Thread/process safety: each server process writes its own save_dir,
    so no locking is needed in the standard FL simulation setup.
    """

    FILENAME = "timing_record.json"

    def __init__(self, save_dir: str, n_workers_total: int):
        self._save_dir = save_dir
        self._n_workers_total = n_workers_total
        self._records: list[dict] = []
        self._path = os.path.join(save_dir, self.FILENAME)
        self._load_existing()

    # ------------------------------------------------------------------
    def _load_existing(self) -> None:
        if os.path.exists(self._path):
            try:
                with open(self._path, "r") as f:
                    self._records = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._records = []

    # ------------------------------------------------------------------
    def record(self, record: RoundTimingRecord) -> None:
        """
        Finalise the record, append to in-memory list, and write JSON.
        Called once per round at the end of _after_send_result().
        """
        record.n_workers_total = self._n_workers_total
        record.finalise()
        self._records.append(record.to_dict())
        self._write()

    # ------------------------------------------------------------------
    def _write(self) -> None:
        os.makedirs(self._save_dir, exist_ok=True)
        tmp = self._path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self._records, f, indent=2)
        os.replace(tmp, self._path)   # atomic on POSIX

    # ------------------------------------------------------------------
    @property
    def path(self) -> str:
        return self._path