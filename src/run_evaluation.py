"""Main module for the SAA application."""
import sys

from src.app.main_evaluation import Main  
from src.instance.experiment import Experiment
from src.utils import LOGGER as logger

if __name__ == "__main__":

    # (0) Set meta-parameters:
    run_supercloud = False
    N = 20
    M = 20

    evaluation_options = [False] # [True, False]
    for evaluation_branch_and_cut in evaluation_options:
        expected = not evaluation_branch_and_cut

        if run_supercloud:
            ROOT_PATH = "."
        else:
            # ROOT_PATH = "/Users/silvestri/Documents/code_multi_period"
            ROOT_PATH = "/Users/juanpina/Dropbox (MIT)/01 Postdoctoral Research/02 Work in Progress/code_multi_period"

        if evaluation_branch_and_cut:
            FOLDER_PATH = ROOT_PATH + "/results/bc/"
        else:
            FOLDER_PATH = ROOT_PATH + "/results/expected/"

        # (1) Generate instance:
        logger.info("[RUN EVALUATION] Generating instances")
        experiments = Experiment(N=N, M=M, folder_path=FOLDER_PATH, evaluation=True, expected=expected)
        combinations_list = list(experiments.get_combinations({}))

        # (2) Get subset of instances to be solved:
        logger.info("[RUN EVALUATION] Getting subset of instances to be solved")
        if run_supercloud:
            my_task_id = int(sys.argv[1])
            num_tasks = int(sys.argv[2])
            combinations_to_solve = combinations_list[my_task_id: len(combinations_list): num_tasks]
        else:
            combinations_to_solve = combinations_list

        logger.info(f"[RUN EVALUATION] Total combinations to be solved: {len(combinations_to_solve)}")

        # (3) Solve the instances:
        for combination in combinations_to_solve:
            try:
                logger.info(f"[RUN EVALUATION] Solving instance with combination: {combination}")
                main = Main(FOLDER_PATH,
                            combination,
                            evaluation_branch_and_cut=evaluation_branch_and_cut,
                            )
                main.solve()
            except Exception as e:
                logger.info(f"[RUN EVALUATION] Exception occurred: {e}")
