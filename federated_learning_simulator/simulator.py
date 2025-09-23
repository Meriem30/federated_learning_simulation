import os
import sys
import hydra

# Add parent directories to Python path
currentdir = os.path.dirname(__file__)
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, os.path.abspath(parentdir))
sys.path.insert(0, os.path.abspath("."))

from federated_learning_simulation_lib.config import DistributedTrainingConfig
from federated_learning_simulation_lib.config import load_config as __load_config
from federated_learning_simulation_lib.training import train
import method  # noqa: F401

global_config: DistributedTrainingConfig = DistributedTrainingConfig()

@hydra.main(config_path="conf", version_base=None)
def load_config(conf) -> None:
    global global_config
    global_config = __load_config(
        conf, os.path.join(os.path.dirname(__file__), "conf", "global.yaml")
    )


if __name__ == "__main__":
    # call the load function
    load_config()
    # start training
    train(config=global_config)
