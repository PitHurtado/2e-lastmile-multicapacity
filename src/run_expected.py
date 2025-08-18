"""Main module for the SAA application."""
import sys

from src.app.main_extended_saa import Main
from src.instance.experiment import Experiment
from src.utils import LOGGER as logger

if __name__ == "__main__":
    # (0) Set meta-parameters:
    """
    Notes:
        - Check Line 187 of Experiments: Ensure you have the filename in BC folder
    """

    # (0) Set meta-parameters:
    run_supercloud = False
    N = 1
    M = 1

    if run_supercloud:
        ROOT_PATH = "."
    else:
        # ROOT_PATH = "/Users/silvestri/Documents/code_multi_period"
        ROOT_PATH = "/Users/juanpina/Dropbox (MIT)/01 Postdoctoral Research/02 Work in Progress/code_multi_period"

    FOLDER_PATH = ROOT_PATH + "/results/expected/"

    # (1) Generate instance:
    logger.info("[RUN EXPECTED] Generating instances")
    experiments = Experiment(N=N, M=M, folder_path=FOLDER_PATH, evaluation=False, expected=True)
    combinations_list = list(experiments.get_combinations({}))

    # (2) Get subset of instances to be solved:
    logger.info("[RUN EXPECTED] Getting subset of instances to be solved")
    if run_supercloud:
        my_task_id = int(sys.argv[1])
        num_tasks = int(sys.argv[2])
        combinations_to_solve = combinations_list[my_task_id: len(combinations_list): num_tasks]
    else:
        combinations_to_solve = combinations_list

    logger.info(f"[RUN EXPECTED] Total combinations to be solved: {len(combinations_to_solve)}")

    # (3) Solve the instances:
    for combination in combinations_to_solve:
        try:
            print(f"[RUN EXPECTED] Solving instance with combination: {combination}")

            main = Main(FOLDER_PATH,
                        combination,
                        scenario_evaluation_id=319,
                        is_evaluation=True,
                        expected=True,
                        max_run_time=60,
                        )

            logger.info(f"[RUN EXPECTED] Starting the solve: {combination}")
            main.solve()
            logger.info(f"[RUN EXPECTED] Done: Starting the solve: {combination}")

        except Exception as e:
            logger.info(f"[RUN EXPECTED] Exception occurred: {e} - Solving {combination}")
