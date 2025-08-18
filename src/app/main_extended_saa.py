"""Module of class Main Extended SAA Model."""
import json
import logging

from src.instance.instance import Instance
from src.model.extended_saa_model import ExtendedSAAModel
from src.utils import LOGGER as logger


class Main:
    def __init__(self, folder_path: str, combination: list, scenario_evaluation_id: int, is_evaluation: bool, expected: bool, max_run_time: int):
        self.folder_path = folder_path
        self.expected = expected
        self.is_evaluation = is_evaluation
        self.scenario_evaluation_id = scenario_evaluation_id
        self.combination = combination
        self.max_run_time = max_run_time

        logger.info(f"[MAIN EXTENDED] Reading instance: {combination}")
        if self.expected:
            (
                ID,
                N,
                M,
                capacity_satellites,
                is_continuous_x,
                type_of_flexibility,
                type_of_cost_serving,
                alpha,
                Y,
            ) = combination
        else:
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
            is_evaluation=is_evaluation,
            scenario_evaluation_id=scenario_evaluation_id,
        )
        logger.info(f"[MAIN EXTENDED] Instance read: {combination}")

    def solve(self):
        logger.info(f"[MAIN EXTENDED] Solving started")

        # (1) Create model:
        solver = ExtendedSAAModel(self.instance)
        solver.build()
        params_config = {"TimeLimit": self.max_run_time,
                         # "MIPGap": 0.0005
                         }
        solver.set_params(params_config)

        # (2) Solve model:
        logger.disabled = True
        logging.disable(logging.CRITICAL)
        solver_metrics = solver.solve()
        logger.disabled = False
        logging.disable(logging.NOTSET)
        logger.info(f"[MAIN EXTENDED] Solving ended")

        # (3) Save results:
        results = {"Y": str({keys: value.X for keys, value in solver.model._Y.items()}),
                   "objective": solver.objective.getValue(),
                   "cost_installation_satellites": solver.cost_installation_satellites.getValue(),
                   "cost_operating_satellites": solver.cost_operating_satellites.getValue(),
                   "cost_served_from_satellite": solver.cost_served_from_satellite.getValue(),
                   "cost_served_from_dc": solver.cost_served_from_dc.getValue(),
                   "scenarios": self.instance.scenarios_ids,
                   # "Instance information": self.get_information(run_time), # MUST BE UNPACKED
                   "Solver information": solver_metrics}

        results.update(self.get_information())

        if self.expected:
            path_file_output = self.folder_path + f"solution_expected_{self.instance.id_instance}.json"
        else:
            path_file_output = self.folder_path + f"solution_extended_saa_model_{self.instance.id_instance}.json"

        with open(path_file_output, "w") as file:
            file.write(json.dumps(results, indent=4))

        logger.info(f"[MAIN EXTENDED] Results saved in {path_file_output}")

    def get_information(self):
        """Return a dictionary with info combination."""
        if self.expected:
            (
                ID,
                N,
                M,
                capacity_satellites,
                is_continuous_x,
                type_of_flexibility,
                type_of_cost_serving,
                alpha,
                Y,
            ) = self.combination

            info_combination = {
                "ID": ID,
                "m": M,
                "N": N,
                "capacity_satellites": capacity_satellites,
                "is_continuous_x": is_continuous_x,
                "type_of_flexibility": type_of_flexibility,
                "type_of_cost_serving": type_of_cost_serving,
                "periods": 12,
                "max_run_time": self.max_run_time,
            }

        else:
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
                "max_run_time": self.max_run_time,
                "warm_start_MP": warm_start_MP,
                "split": split,
                "warm_start_SP": warm_start_SP,
                "valid_inequalities": valid_inequalities,
                "reformulated": reformulated,
            }

        return info_combination
