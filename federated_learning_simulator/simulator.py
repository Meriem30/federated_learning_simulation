import os
import sys

import hydra
from federated_learning_simulation_lib.config import DistributedTrainingConfig  # import_dependencies
from federated_learning_simulation_lib.config import load_config as __load_config
from federated_learning_simulation_lib.training import train

# modify the python path to include the current directory
sys.path.insert(0, os.path.abspath("."))
import method  # noqa: F401

# initialize a global configuration
global_config: DistributedTrainingConfig = DistributedTrainingConfig()
# import_dependencies()


# execute this as the main function specifying the config path location
@hydra.main(config_path="./conf", version_base=None)
def load_config(conf) -> None:
    global global_config
    # load a conf file specified by hydra
    # update the global_config with the loaded conf (param) and
    # a global YAML conf file located inn the constructed path
    global_config = __load_config(
        conf, os.path.join(os.path.dirname(__file__), "conf", "global.yaml")
    )


if __name__ == "__main__":
    # call the load function
    load_config()
    # start training
    train(config=global_config)
