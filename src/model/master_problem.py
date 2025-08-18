"""Module for the master problem of the stochastic model."""
from typing import Any, Dict, Tuple

import gurobipy as gp
import numpy as np
from gurobipy import GRB, quicksum

from src.classes import Satellite
from src.instance.instance import Instance
from src.instance.scenario import Scenario
from src.model.sub_problem import SubProblem
from src.utils import LOGGER as logger


class MasterProblem:
    """Class for the master problem of the stochastic model."""

    def __init__(self, instance: Instance, reformulated: bool, split: int, warm_start_subproblems: bool) -> None:
        """Initialize the master problem."""
        self.model: gp.Model = gp.Model(name="MasterProblem")

        # Instance
        self.instance = instance
        self.periods: int = instance.periods
        self.satellites: Dict[str, Satellite] = instance.satellites
        self.scenarios: Dict[str, Scenario] = instance.scenarios
        self.type_of_flexibility: int = instance.type_of_flexibility
        self.reformulated = reformulated
        self.split = split

        # Variables
        self.Y = {}
        self.θ = {}

        if self.reformulated:
            self.X = {}
            self.W = {}

        # Compute global lower bounds
        self.LB = self.__compute_lower_bound()

        # objective
        self.objective = None
        self.cost_installation_satellites = None
        self.cost_second_stage = None

    def build(self) -> None:
        """Build the master problem."""
        logger.info("[MASTER PROBLEM] Building master problem")
        self.__add_variables(self.satellites, self.scenarios)
        self.__add_objective(self.satellites, self.scenarios)
        self.__add_constraints(self.satellites)

        self.model._start_time = 0
        self.model.update()
        logger.info("[MASTER PROBLEM] Master problem built")

    def __add_variables(
            self, satellites: Dict[str, Satellite], scenarios: Dict[str, Scenario]
    ) -> None:
        """Add variables to the model."""
        logger.info("[MASTER PROBLEM] Adding variables to master problem")
        self.Y = dict(
            [
                (
                    (s, q),
                    self.model.addVar(vtype=GRB.BINARY, name=f"Y_s{s}_q{q}"),
                )
                for s, satellite in satellites.items()
                for q in satellite.capacity.keys()
            ]
        )
        logger.info(f"[Master Problem] Number of variables Y: {len(self.Y)}")

        self.θ = dict(
            [
                (
                    (n, t),
                    self.model.addVar(vtype=GRB.CONTINUOUS,
                                      lb=self.LB[(n, t)],
                                      name=f"θ_n{n}_t{t}"),
                )
                for n in scenarios.keys()
                for t in range(self.periods)
            ]
        )
        logger.info(f"[Master Problem] Number of variables θ: {len(self.θ)}")

        if self.reformulated:
            self.X = dict(
                [
                    (
                        (s, k, n, t),
                        self.model.addVar(
                            vtype=GRB.CONTINUOUS, name=f"X_s{s}_k{k}_n{n}_t{t}", lb=0, ub=1
                        ),
                    )
                    for s in satellites.keys()
                    for n, scenario in scenarios.items()
                    for k in scenario.pixels.keys()
                    for t in range(self.periods)
                ]
            )
            logger.info(f"[Master Problem] Number of variables X: {len(self.X)}")

            self.W = dict(
                [
                    (
                        (k, n, t),
                        self.model.addVar(
                            vtype=GRB.CONTINUOUS, name=f"W_k{k}_n{n}_t{t}", lb=0, ub=1
                        ),
                    )
                    for n, scenario in scenarios.items()
                    for k in scenario.pixels.keys()
                    for t in range(self.periods)
                ]
            )
            logger.info(f"[Master Problem] Number of variables W: {len(self.W)}")

        self.model._Y = self.Y
        self.model._θ = self.θ

        if self.reformulated:
            self.model._X = self.X
            self.model._W = self.W

    def __compute_lower_bound_old(self):
        """Compute the lower bound for the second stage cost."""
        LB = {}
        logger.info("[MASTER PROBLEM] Computing global lower bounds")
        for n, scenario in self.scenarios.items():
            for t in range(self.periods):
                LB[(n, t)] = np.sum(
                    [
                        np.min(
                            [
                                scenario.get_cost_serving("satellite")[(s, k, t)]["total"]
                                for s in self.satellites.keys()
                            ]
                            + [scenario.get_cost_serving("dc")[(k, t)]["total"]]
                        )
                        for k in scenario.pixels.keys()
                    ]
                )

        logger.info(f"[MASTER PROBLEM] Lower bounds computed")
        return LB

    def __compute_lower_bound(self):
        """Compute the lower bound for the second stage cost."""
        LB = {}
        print("[MASTER PROBLEM] Computing global lower bounds")

        Y = {
            (s, q_key): int(q_value == max(satellite.capacity.values()))
            for s, satellite in self.satellites.items()
            for q_key, q_value in satellite.capacity.items()
        }

        for n in self.instance.scenarios.keys():
            subproblem = SubProblem(self.instance, self.periods, self.instance.scenarios[n], self.split, True)
            subproblem_runtime, subproblem_total_costs, X_values, W_values, Z_values = subproblem.solve_model(fixed_y=Y)

            for t in range(self.periods):
                LB[(n, t)] = subproblem.total_costs[t]

        print(f"[MASTER PROBLEM] Global lower bounds computed")
        return LB

    def __add_objective(
            self, satellites: Dict[str, Satellite], scenarios: Dict[str, Scenario]
    ) -> None:
        """Add objective function to the model."""
        logger.info("[MASTER PROBLEM] Adding objective function to master problem")

        # Fixed cost:
        if self.type_of_flexibility >= 2:
            cost_installation_satellites = quicksum(
                [
                    round(satellite.cost_fixed[q_key], 0) * self.Y[(s, q_key)]
                    for s, satellite in satellites.items()
                    for q_key, q_values in satellite.capacity.items()
                    if q_values > 0
                ]
            )
        else:
            cost_installation_satellites = quicksum(
                [
                    (round(satellite.cost_fixed[q_key], 0) + sum(round(satellite.cost_operation[q_key][t], 0) for t in range(self.periods)))
                    * self.Y[(s, q_key)]
                    for s, satellite in satellites.items()
                    for q_key, q_values in satellite.capacity.items()
                    if q_values > 0
                ]
            )

        # Second stage cost:
        cost_second_stage = (
            1 / (len(scenarios)) * quicksum([self.θ[(n, t)]
                                             for n in scenarios.keys()
                                             for t in range(self.periods)]
                                            )
        )

        total_cost = cost_installation_satellites + cost_second_stage
        self.model.setObjective(total_cost, GRB.MINIMIZE)

        self.objective = total_cost
        self.cost_installation_satellites = cost_installation_satellites
        self.cost_second_stage = cost_second_stage

        self.model._total_cost = total_cost
        self.model._cost_installation_satellites = cost_installation_satellites
        self.model._cost_second_stage = cost_second_stage

    def __add_constraints(self, satellites: Dict[str, Satellite]) -> None:
        """Add constraints to the model."""
        logger.info("[MASTER PROBLEM] Adding constraints to master problem")

        # Constraints (15):
        for s, satellite in satellites.items():
            nameConstraint = f"R_Open_s{s}"
            constraint_expression = quicksum([self.Y[(s, q)] for q in satellite.capacity.keys()])
            self.model.addConstr(
                constraint_expression == 1,
                name=nameConstraint,
            )

        if self.reformulated:
            # Constraints (19):
            for n, scenario in self.scenarios.items():
                for t in range(self.periods):
                    self.model.addConstr(
                        self.θ[(n, t)] >=
                        quicksum(
                            [
                                 scenario.get_cost_serving("satellite")[(s, k, t)]["total"]
                                 * self.X[(s, k, n, t)]
                                 for s in satellites.keys()
                                 for k in scenario.pixels.keys()
                            ]
                        )
                        + quicksum(
                            [
                                scenario.get_cost_serving("dc")[(k, t)]["total"] * self.W[(k, n, t)]
                                for k in scenario.pixels.keys()
                            ]
                        )
                    )

            # Constraints (20):
            for t in range(self.periods):
                for s, satellite in satellites.items():
                    for n, scenario in self.scenarios.items():
                        pixels = scenario.pixels
                        fleet_size_required = scenario.get_fleet_size_required("satellite")
                        nameConstraint = f"R_capacity_s{s}_n{n}_t{t}"

                        self.model.addConstr(
                            quicksum(
                                [
                                    self.X[(s, k, n, t)]
                                    * round(fleet_size_required[(s, k, t)]["fleet_size"], 1)
                                    for k in pixels.keys()
                                ]
                            )
                            - quicksum(
                                [
                                    self.Y[(s, q)] * capacity
                                    for q, capacity in satellite.capacity.items()
                                    if capacity > 0
                                ]
                            )
                            <= 0,
                            name=nameConstraint,
                        )

            # Constraints (21):
            for n, scenario in self.scenarios.items():
                for t in range(self.periods):
                    for k in scenario.pixels.keys():
                        nameConstraint = f"R_demand_k{k}_n{n}_t{t}"
                        self.model.addConstr(
                            quicksum([self.X[(s, k, n, t)] for s in satellites.keys()])
                            + quicksum([self.W[(k, n, t)]])
                            >= 1,
                            name=nameConstraint,
                        )

            # Constraints (22):
            for s, satellite in self.satellites.items():
                for n, scenario in self.scenarios.items():
                    for t in range(self.periods):
                        for k in scenario.pixels.keys():
                            nameConstraint = f"R_X_s{s}_k{k}_n{n}_t{t}"
                            self.model.addConstr(
                                self.X[(s, k, n, t)] <= 1 - self.Y[(s, '0')],
                                name=nameConstraint,
                            )

            # New valid inequalities:
            for s, satellite in self.satellites.items():
                for q, capacity in satellite.capacity.items():
                    if capacity > 0:
                        nameConstraint = f"R_X_s{s}_q{q}"
                        self.model.addConstr(self.Y[(s, q)]
                                             <=
                                             quicksum([self.X[(s, k, n, t)]
                                                      for n, scenario in self.scenarios.items()
                                                      for t in range(self.periods)
                                                      for k in scenario.pixels.keys()]
                                                      ),
                                             name=nameConstraint,
                                             )

            # for s, satellite in self.satellites.items():
            #     for q, capacity in satellite.capacity.items():
            #         self.model.addConstr((self.Y[(s, q)] <= 0.5) >> (self.Y[(s, q)] == 0))

    def get_objective_value(self):
        """Get the objective value of the model."""
        logger.info("[MASTER PROBLEM] Getting objective value")
        return self.model._total_cost.getValue()

    def get_best_bound_value(self):
        """Get the best bound value of the model."""
        logger.info("[MASTER PROBLEM] Getting best bound value")
        return self.model.ObjBound

    def set_start_time(self, start_time):
        """Set start time to model."""
        self.model._start_time = start_time

    def set_params(self, params: Dict[str, int]):
        """Set params to model."""
        logger.info(f"[MASTER PROBLEM] Set params to model {params}")
        for key, item in params.items():
            self.model.setParam(key, item)

    def __warm_start(self):
        """Warm start the model."""
        logger.info(f"[MASTER PROBLEM] Setting warm start")

        for (s, q) in self.Y.keys():
            self.Y[(s, q)].start = 0
            satellite = self.satellites[s]
            is_max_capacity = satellite.capacity[q] == max(satellite.capacity.values())
            if is_max_capacity:
                self.Y[(s, q)].start = int(is_max_capacity)

    def get_y_solution(self):
        """Get the Y solution of the model."""
        logger.info("[MASTER PROBLEM] Getting Y solution")
        return {keys: value.x for keys, value in self.Y.items()}

    def set_values(self, y_values, θ_values, x_values, w_values):
        self.model.update()
        logger.info("[MASTER PROBLEM] Setting values")

        for key in self.Y.keys():
            self.Y[key].start = y_values[key]

        for key in self.θ.keys():
            self.θ[key].start = θ_values[key]

        if self.reformulated:
            for key in self.X.keys():
                self.X[key].start = x_values[key] # TODO: Should we round this?

            for key in self.W.keys():
                self.W[key].start = w_values[key]

        self.model.update()

    # def solve(self):
    #     """Solve the Master Problem."""
    #     logger.info("[MASTER PROBLEM] Solving Master Problem")
    #     self.model.optimize()
    #     logger.info("[MASTER PROBLEM] Master Problem solved")
