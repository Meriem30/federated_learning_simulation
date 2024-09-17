import functools
import json
import os
import pickle

import dill


def get_worker_stat(session_dir: str) -> dict:
    """
        take a directory path
        return worker statics in a dict
    """
    worker_data: dict = {}
    for root, dirs, __ in os.walk(os.path.join(session_dir, "..")):
        for name in dirs:
            # search for worker directory
            if name.startswith("worker"):
                # initialize worker_data empty dict
                worker_data[name] = {}
                # open and read
                with open(
                    os.path.join(root, name, "hyper_parameter.pk"),
                    "rb",
                ) as f:
                    worker_data[name]["hyper_parameter"] = dill.load(f)
                if os.path.isfile(os.path.join(root, name, "graph_worker_stat.json")):
                    with open(
                        os.path.join(root, name, "graph_worker_stat.json"),
                        "rt",
                        encoding="utf8",
                    ) as f:
                        worker_data[name] = json.load(f)
    return worker_data


class Session:
    """
        encapsulate the session data
    """
    def __init__(self, session_dir: str) -> None:
        """
            initialize a session obj with a given session directory
        """
        assert session_dir
        # open and read
        with open(
            os.path.join(session_dir, "round_record.json"), "rt", encoding="utf8"
        ) as f:
            # convert the JSON data to a dict
            self.round_record = json.load(f)
        # convert the keys into integers
        self.round_record = {int(k): v for k, v in self.round_record.items()}

        with open(os.path.join(session_dir, "config.pkl"), "rb") as f:
            # convert back the pickled conf obj to the original python obj
            self.config = pickle.load(f)

        # get worker stats to populate with worker_data
        self.worker_data: dict = get_worker_stat(session_dir)
        if not self.worker_data:
            raise RuntimeError(os.path.join(session_dir, ".."))

    @functools.cached_property
    def rounds(self) -> list:
        """
            return a sorted list of round numbers
        """
        return sorted(self.round_record.keys())

    @functools.cached_property
    def last_round(self) -> int:
        """
            return the last round number
        """
        return self.rounds[-1]

    @functools.cached_property
    def last_test_acc(self) -> float:
        """
            return the test accuracy for the last round
        """
        return self.round_record[self.last_round]["test_accuracy"]

    @functools.cached_property
    def mean_test_acc(self) -> float:
        """
            return the mean test accuracy
            for all rounds
        """
        total_acc = 0
        for r in self.round_record.values():
            total_acc += r["test_accuracy"]
        return total_acc / len(self.round_record)
