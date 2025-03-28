import os
import sys

os.environ["TQDM_DISABLE"] = "1"
os.environ["LEAST_REQUIRED_DEVICE_MEMORY_IN_GB"] = "10"

from cyy_naive_lib.log import redirect_stdout_to_logger
from distributed_learning_simulation import (
    load_config,
    train,
)

project_path = os.path.join(os.path.dirname(__file__))
sys.path.insert(0, project_path)
import src.method  # noqa: F401

if __name__ == "__main__":
    config_path = os.path.join(project_path, "conf")
    with redirect_stdout_to_logger(logger_names=["transformers"]):
        config = load_config(
            config_path=config_path,
            global_conf_path=os.path.join(config_path, "global.yaml"),
            import_libs=False,
        )
        train(config=config, single_task=True)
