"""Module to solve the Branch and Cut algorithm"""
import time
from typing import Any, Dict

import numpy as np
import io
from matplotlib import pyplot as plt
import logging
import pstats
import cProfile


from src.classes import Satellite
from src.instance.instance import Instance
from src.instance.scenario import Scenario
from src.model.cuts import Cuts
from src.model.master_problem import MasterProblem
from src.model.warm_start import WarmStart
from src.model.sub_problem import SubProblem
from src.utils import LOGGER as logger


class Branch_and_Cut:
    """Class to define the Branch and Cut algorithm"""

    def __init__(self,
                 instance: Instance,
                 warm_start: bool,
                 valid_inequalities: bool,
                 maximum_run_time: int,
                 profiler: bool,
                 split: int,
                 warm_start_subproblems: bool,
                 reformulated: bool):

        # Solvers
        self.MP = MasterProblem(instance, reformulated, split, warm_start_subproblems)
        self.Cuts = Cuts(instance, self.MP.LB, valid_inequalities, split, warm_start_subproblems, reformulated)
        if warm_start:
            self.WS = WarmStart(instance, reformulated, split, warm_start_subproblems)

        # Params
        self.instance: Instance = instance
        self.satellites: Dict[str, Satellite] = instance.satellites
        self.scenarios: Dict[str, Scenario] = instance.scenarios
        self.warm_start = warm_start
        self.periods = instance.periods
        self.valid_inequalities = valid_inequalities
        self.maximum_run_time = maximum_run_time
        self.profiler = profiler
        self.reformulated = reformulated

        # Config params
        self.objective_value = 0
        self.initial_upper_bound = 0
        self.run_time = 0
        self.optimality_gap = 0
        self.best_bound_value = 0
        self.y_solution = {}

    def solve(self) -> None:
        """Solve the Branch and Cut algorithm"""
        logger.info("[BRANCH AND CUT] Start Branch and Cut algorithm - ID instance: %s", self.instance.id_instance)

        # (1) Create master problem:
        self.MP.build()

        # (2) Calculate warm start for master problem
        if self.warm_start:
            y_values, θ_values, x_values, w_values = self.__solve_warm_start()
            logger.info(f"[BRANCH AND CUT] Setting start values to master problem")
            self.MP.set_values(y_values, θ_values, x_values, w_values)

        # (3) Define Gurobi parameters and optimize:
        params_config = {"TimeLimit": self.maximum_run_time,
                         "MIPGap": 0.001,
                         'Threads': 12,
                         # "Heuristics": 0,
                         # "Cuts": 0,
                         }

        self.MP.set_params(params_config)
        self.MP.model.Params.lazyConstraints = 1

        start_time = time.time()
        self.MP.set_start_time(start_time)
        self.Cuts.set_start_time(start_time)

        logger.info("[BRANCH AND CUT] Starting optimizing")

        logger.disabled = True
        logging.disable(logging.CRITICAL)

        if self.profiler:
            pr = cProfile.Profile()
            pr.enable()
            self.MP.model.optimize(Cuts.add_cuts)

            pr.disable()
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s)
            ps.strip_dirs().sort_stats("cumulative").print_stats(10)
            print(f"Profile: {s.getvalue()}")
        else:
            self.MP.model.optimize(Cuts.add_cuts)

        logger.disabled = False
        logging.disable(logging.NOTSET)
        logger.info("[BRANCH AND CUT] End Branch and Cut algorithm")

        # (4) Save metrics:
        logger.info("[BRANCH AND CUT] Save metrics")
        try:
            self.run_time = round(time.time() - start_time, 3)
            self.optimality_gap = round(100 * self.MP.model.MIPGap, 3)
            self.objective_value = round(self.MP.get_objective_value(), 3)
            self.best_bound_value = round(self.MP.get_best_bound_value(), 3)
            self.y_solution = self.MP.get_y_solution()
            self.MP.model.dispose()

        except AttributeError:
            logger.error(
                "[BRANCH AND CUT] Error while saving metrics - id instance: %s",
                self.instance.id_instance,
            )

    def get_best_solution_allocation(self):
        """Get the best solution allocation"""
        return self.Cuts.best_solution

    def get_y_solution(self):
        return self.y_solution

    def get_metrics_evaluation(self):
        """Get metrics of the evaluation"""
        metrics = {
            "id_instance": self.instance.id_instance,
            "cost_installed_satellites": self.MP.model._cost_installation_satellites.getValue(),
            "run_time": self.run_time,
            "optimality_gap": self.optimality_gap,
            "objective_value": self.objective_value,
            "best_bound_value": self.best_bound_value,
            "solution": {
                str(key): value for key, value in self.Cuts.best_solution.items()
            },
            "cost_second_echeleon": self.MP.model._cost_second_stage.getValue(),
        }
        return metrics

    def __solve_subproblem(
        self, scenario: Scenario, t: int, solution: Dict[Any, float]
    ) -> float:
        """Solve the subproblem and return the total cost of the solution"""
        # (1) create subproblem
        sub_problem = SubProblem(self.instance, t, scenario)
        sp_run_time, sp_total_costs, sp_x_values, sp_w_values, sp_z_values = sub_problem.solve_model(
            solution, True
        )
        return sp_total_costs["sp_total_cost"]

    def solve_evaluation(self, solution: Dict[Any, float]) -> float:
        """Solve the subproblem for the evaluation"""
        # (1) compute cost installed satellites
        cost_installed_satellites = np.sum(
            [
                satellite.cost_fixed[q] * solution[(s, q)]
                for s, satellite in self.satellites.items()
                for q in satellite.capacity.keys()
            ]
        )

        # (2) create subproblem N * T times and solve
        cost_second_echeleon = 0
        for scenario in self.scenarios.values():
            for t in range(self.periods):
                cost_second_echeleon += self.__solve_subproblem(scenario, t, solution)

        # (3) compute total cost of the evaluation
        total_cost = (
            cost_installed_satellites + (1 / len(self.scenarios)) * cost_second_echeleon
        )
        return total_cost

    def get_final_metrics(self):
        """Get final metrics"""
        metrics = {
            "actual_run_time": self.run_time,
            "optimality_gap": self.optimality_gap,
            "objective_value": self.objective_value,
            "best_bound_value": self.best_bound_value,
        }
        return metrics

    def __solve_warm_start(self):
        # (1) Create warm start problem:
        self.WS.build()

        # (2) Define Gurobi parameters and optimize:
        params_config = {"TimeLimit": self.maximum_run_time,
                         "MIPGap": 0.001,
                         # "Heuristics": 0.2,
                         }

        self.WS.set_params(params_config)

        logger.info("[BRANCH AND CUT] Starting optimizing warm start model")
        logger.disabled = True
        logging.disable(logging.CRITICAL)
        self.WS.model.optimize()
        logger.disabled = False
        logging.disable(logging.NOTSET)
        logger.info("[BRANCH AND CUT] End optimizing warm start")

        # (3) Save values:
        logger.info("[BRANCH AND CUT] Save warm start values")
        try:
            y_values = self.WS.get_y_solution()
            x_values = self.WS.get_x_solution() # TODO: These variables are relaxed in the warm start
            w_values = self.WS.get_w_solution()
            θ_values = self.WS.get_θ_solution()
            self.WS.model.dispose()

        except AttributeError:
            logger.error("[BRANCH AND CUT] Error while saving warm start values")

        return y_values, θ_values, x_values, w_values
