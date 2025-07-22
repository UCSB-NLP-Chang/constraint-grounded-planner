from .runner import SequentialRunner
from .data_utils import DataDict
from .config_store import RunConfig

from simulation.simenv.box1 import Box1Env


# TODO (jiabao): implement planner class
class Astar_Box_Planner(SequentialRunner):
    def __init__(
        self,
        configs,
        rundir,
        dataset,
        batch_size,
        model,
        tokenizer,
        eval_fn,
        sampling_params,
        **kwargs,
    ):
        super().__init__(
            configs,
            rundir,
            dataset,
            batch_size,
            model,
            tokenizer,
            eval_fn,
            sampling_params,
            **kwargs,
        )

    def consume_batch(self, batch: DataDict):
        for env_json in batch.to_list():  # Assume each batch is a list of environments
            env = Box1Env.load(env_json)

            # Run A* planner

            pass
