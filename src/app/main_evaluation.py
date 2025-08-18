"""Module of class Main Evaluation."""
import json
import logging
import numpy as np

from src.instance.instance import Instance
from src.model.sub_problem import SubProblem
from src.utils import LOGGER as logger


class Main:
    def __init__(self, folder_path: str, combination: list, evaluation_branch_and_cut: bool):
        self.folder_path = folder_path
        logger.info(f"[MAIN EVALUATION] Reading instance: {combination}")
        (
            ID,
            N,
            M,
            capacity_satellites,
            is_continuous_x,
            type_of_flexibility,
            type_of_cost_serving,
            alpha,
            split,
            warm_start_MP,
            warm_start_SP,
            reformulated,
            valid_inequalities,
            Y_solution,
            scenario_evaluation_id,
        ) = combination

        self.instance = Instance(
            id_instance=ID,
            capacity_satellites=capacity_satellites,
            is_continuous_x=is_continuous_x,
            alpha=alpha,
            type_of_flexibility=type_of_flexibility,
            type_of_cost_serving=type_of_cost_serving,
            periods=12,
            N=N,
            is_evaluation=True,
            scenario_evaluation_id=scenario_evaluation_id,
        )

        self.combination = combination
        self.ID = ID
        self.Y_solution = Y_solution
        self.scenario_evaluation_id = scenario_evaluation_id
        self.scenario = self.instance.scenarios[str(self.scenario_evaluation_id)]
        self.split = split
        self.periods = 12
        self.evaluation_branch_and_cut = evaluation_branch_and_cut
        if self.evaluation_branch_and_cut:
            self.folder_path = folder_path.replace("bc", "") + "evaluation/"
        else:
            self.folder_path = folder_path.replace("expected", "") + "evaluation/"

        logger.info(f"[MAIN EVALUATION] Instance read: {combination}")

    def solve(self):
        logger.info(f"[MAIN EVALUATION] Solving started")

        subproblem = SubProblem(self.instance, self.periods, self.scenario, self.split, False)
        subproblem_runtime, subproblem_total_costs, X_values, W_values, Z_values = subproblem.solve_model(fixed_y=self.Y_solution)
        results = {"Y": str(self.Y_solution)}
        results.update({"Z": str(Z_values)})
        results.update({"X": str(X_values)})
        results.update({"W": str(W_values)})
        results.update(subproblem.get_evaluation_metrics(self.Y_solution, X_values, W_values, Z_values))
        results.update(self.get_information())

        total_cost = np.sum(
            [
                satellite.cost_fixed[q] * self.Y_solution[(s, q)]
                for s, satellite in self.instance.satellites.items()
                for q, capacity in satellite.capacity.items() if capacity > 0
            ]
        ) + np.sum([subproblem_total_costs[t] for t in range(self.periods)])

        other_metrics = {'run_time': subproblem_runtime,
                         'total_cost': total_cost}

        results.update(other_metrics)

        if self.evaluation_branch_and_cut:
            file_name = self.folder_path + f"test_solution_branch_and_cut_{self.instance.id_instance}_evaluated_on_scenario_{self.scenario_evaluation_id}.json"
        else:
            file_name = self.folder_path + f"test_solution_expected_{self.instance.id_instance}_evaluated_on_scenario_{self.scenario_evaluation_id}.json"

        results_to_save = {str(key): value for key, value in results.items()}

        with open(file_name, "w") as json_file:
            json.dump(results_to_save, json_file, indent = 4)

        logger.info(f"[MAIN EVALUATION] Results saved in {file_name}")

    def get_information(self):
        """Return a dictionary with info combination."""
        (
            ID,
            N,
            M,
            capacity_satellites,
            is_continuous_x,
            type_of_flexibility,
            type_of_cost_serving,
            alpha,
            split,
            warm_start_MP,
            warm_start_SP,
            reformulated,
            valid_inequalities,
            Y_solution,
            scenario_evaluation_id,
        ) = self.combination

        info_combination = {
            "ID": ID,
            "m": M,
            "N": N,
            "capacity_satellites": capacity_satellites,
            "is_continuous_x": is_continuous_x,
            "type_of_flexibility": type_of_flexibility,
            "type_of_cost_serving": type_of_cost_serving,
            "alpha": alpha,
            "periods": 12,
            "warm_start_MP": warm_start_MP,
            "split": split,
            "warm_start_SP": warm_start_SP,
            "valid_inequalities": valid_inequalities,
            "reformulated": reformulated,
            "scenario_evaluation_ID": scenario_evaluation_id,
        }

        return info_combination

