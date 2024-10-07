import os
import sys
import hydra

currentdir = os.path.dirname(__file__)
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, os.path.abspath(parentdir))

from federated_learning_simulation_lib.config import DistributedTrainingConfig, import_dependencies
from federated_learning_simulation_lib.config import load_config as __load_config
from federated_learning_simulation_lib.training import train

# modify the python path to include the current directory
sys.path.insert(0, os.path.abspath("."))

import method  # noqa: F401

# initialize a global configuration
global_config: DistributedTrainingConfig = DistributedTrainingConfig()
# import_dependencies()


@hydra.main(config_path="./conf", config_name="fed_avg/mnist.yaml", version_base=None)
def load_config(conf) -> None:
    global global_config
    global_config = __load_config(
        conf, os.path.join(os.path.dirname(__file__), "conf", "global.yaml")
    )


if __name__ == "__main__":
    # call the load function
    load_config()
    print("this is the conf from simulator", global_config.dc_config.dataset_name)
    # start training
    train(config=global_config)
