"""Module for the Sample Average Approximation (SAA) model."""

import json
from typing import Any, Dict, List

from src.constants import PATH_BEST_SOLUTION_SAA
from src.instance.instance import Instance
from src.model.branch_and_cut import Branch_and_Cut
from src.utils import LOGGER as logger


class SampleAverageApproximation:
    """Class to define the Sample Average Approximation (SAA) model"""

    def __init__(self, experiment: Dict[str, Any], id_experiment: int):
        self.experiment = experiment
        self.id_experiment = id_experiment

    def __create_branch_and_cut(self, instance: Instance) -> Branch_and_Cut:
        """Return a Branch_and_Cut object."""
        return Branch_and_Cut(instance)

    def run(self) -> None:
        """
        Run one experiment of the SAA model determinate best solution.
        -----
        Params:
            - experiment: Dict[str, Any]
                contains instances train and evaluation
        """
        instances_train: Dict[str, Instance] = self.experiment["instances_train"]
        instances_evaluation: Instance = self.experiment["instance_evaluation"]

        # (1) Train the model
        best_solutions: List[Dict[Any, float]] = []
        for n, instance in instances_train.items():
            print(
                f"[SAA] ID experiment {self.id_experiment} -  Train model for instance {n}"
            )
            branch_and_cut: Branch_and_Cut = self.__create_branch_and_cut(instance)
            branch_and_cut.solve(max_run_time=3600, warm_start=False)
            current_solution: Dict[
                str, float
            ] = branch_and_cut.get_best_solution_allocation()

            print(
                f"[SAA ---------] ID experiment {self.id_experiment} - current solution: {current_solution} for instances {instance.id_instance}"
            )

            # check if the current solution is not in the best solutions
            # if not any(
            #     key_current_solution in best_solution
            #     for best_solution in best_solutions
            #     for key_current_solution in current_solution.keys()
            # ):
            #     best_solutions.append(current_solution)
            best_solutions.append(current_solution)

        # (2) Evaluate the model
        logger.info(
            f"[SAA] ID experiment {self.id_experiment} - Evaluate model for instance \n {instances_evaluation}"
        )
        solutions_evaluation: List[Dict[str, float]] = []
        for i, solution in enumerate(best_solutions):
            bc_evaluation: Branch_and_Cut = self.__create_branch_and_cut(
                instances_evaluation
            )
            current_solution["objective_value"] = bc_evaluation.solve_evaluation(
                solution
            )
            current_solution["id_experiment"] = self.id_experiment
            current_solution["id_solution"] = i
            current_solution["solution"] = solution
            solutions_evaluation.append(current_solution)
            logger.info(
                f"[SAA] ID experiment {self.id_experiment} - Evaluate model for instance \n {instances_evaluation} - Solution {i} - Objective value {current_solution['objective_value']}"
            )

        # (3) select the best solution
        best_solution: Dict[Any, float] = min(
            solutions_evaluation, key=lambda x: x["objective_value"]
        )
        logger.info(
            f"[SAA] ID experiment {self.id_experiment} - Best solution: {best_solution}"
        )

        # (4) Save the best solution
        path_json: str = (
            PATH_BEST_SOLUTION_SAA + f"best_solution_{self.id_experiment}.json"
        )
        print(best_solution)
        # with open(path_json, "w") as file:
        #     json.dump(best_solution, file)
        logger.info(
            f"[SAA] ID experiment {self.id_experiment} - Save best solution in {path_json}"
        )
