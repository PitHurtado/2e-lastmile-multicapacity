"""Module for the sub problem of the stochastic model."""
import time
from typing import Any, Dict

import gurobipy as gp
import numpy as np
from gurobipy import GRB, quicksum

from src.classes import Pixel, Satellite
from src.instance.instance import Instance
from src.instance.scenario import Scenario
from src.utils import LOGGER as logger


class SubProblem:
    """Class for the subproblem of the stochastic model."""

    def __init__(self, instance: Instance, periods: int, scenario: Scenario, split: int, warm_start_subproblems: bool) -> None:
        """Initialize the sub problem."""
        self.model = gp.Model(name="SubProblem")

        # Params from instance
        self.instance: Instance = instance
        self.periods: int = periods
        self.split: int = split
        self.satellites: Dict[str, Satellite] = instance.satellites
        self.type_of_flexibility: int = instance.type_of_flexibility
        self.is_continuous_x: bool = instance.is_continuous_x
        self.alpha = instance.alpha
        self.scenario = scenario
        self.warm_start_subproblems = warm_start_subproblems

        # Creating the list of end_ranges that we want to solve separately, based on self.split value
        self.ranges = self.__create_time_periods_ranges()

        # Params from scenario
        self.pixels: Dict[str, Pixel] = scenario.pixels
        self.costs_serving: Dict[str, Dict] = scenario.get_cost_serving()
        self.fleet_size_required: Dict[str, Any] = scenario.get_fleet_size_required()

        # Variables
        self.Z = {}
        self.X = {}
        self.W = {}

        # Objective
        self.objective = None
        self.cost_operating_satellites = None
        self.cost_serving_from_satellite = None
        self.cost_serving_from_dc = None
        self.total_costs = {}

    def __create_time_periods_ranges(self) -> list:
        group_size = self.periods // self.split
        remainder = self.periods % self.split

        ranges = []
        current_period = 0

        for i in range(1, self.split + 1):
            if i <= remainder:
                current_period += group_size + 1
            else:
                current_period += group_size

            ranges.append(current_period)
        return ranges

    def __add_variables(
        self,
        satellites: Dict[str, Satellite],
        pixels: Dict[str, Pixel],
        fixed_y: Dict,
        subproblem_range: range,
    ) -> None:
        """Add variables to the model."""
        # 1. add variable Z: binary variable to decide if a satellite is operating
        if self.type_of_flexibility == 2:
            self.Z = dict(
                [
                    (
                        (s, q_lower_key, t),
                        self.model.addVar(
                            vtype=GRB.BINARY, name=f"Z_s{s}_q_{q_lower_key}_t{t}"
                        ),
                    )
                    for s, satellite in satellites.items()
                    for t in subproblem_range
                    for q_lower_key, q_lower_value in satellite.capacity.items()
                    for q_key, q_value in satellite.capacity.items()
                    if fixed_y[(s, q_key)] > 0.5 and q_value >= q_lower_value
                ]
            )
        elif self.type_of_flexibility == 3:
            self.Z = dict(
                [
                    (
                        (s, q_lower_key, t),
                        self.model.addVar(
                            vtype=GRB.BINARY, name=f"Z_s{s}_q_{q_lower_key}_t{t}"
                        ),
                    )
                    for s, satellite in satellites.items()
                    for t in subproblem_range
                    for q_lower_key, q_lower_value in satellite.capacity.items()
                    for q_key, q_value in satellite.capacity.items()
                    if fixed_y[(s, q_key)] > 0.5 and (q_value == q_lower_value or q_lower_value == 0)
                ]
            )

        # logger.info(f"[SUBPROBLEM] Number of variables Z: {len(self.Z)}")

        # 2. add variable X: binary variable to decide if a satellite is used to serve a pixel
        type_variable = GRB.CONTINUOUS if self.is_continuous_x else GRB.BINARY
        self.X = dict(
            [
                (
                    (s, k, t),
                    self.model.addVar(
                        vtype=type_variable,
                        name=f"X_s{s}_k{k}_t{t}",
                        lb=0,
                        ub=1,
                    ),
                )
                for t in subproblem_range
                for k in pixels.keys()
                for s, satellite in satellites.items()
                if any(fixed_y[(s, q)] > 0.5 for q, capacity in satellite.capacity.items() if capacity > 0)
            ]
        )
        # logger.info(f"[SUBPROBLEM] Number of variables X: {len(self.X)}")

        # 3. add variable W: binary variable to decide if a pixel is served from dc
        self.W = dict(
            [
                (
                    (k, t),
                    self.model.addVar(
                        vtype=type_variable, name=f"W_k{k}_t{t}", lb=0, ub=1
                    ),
                )
                for k in pixels.keys()
                for t in subproblem_range
            ]
        )
        # logger.info(f"[SUBPROBLEM] Number of variables W: {len(self.W)}")

    def __add_objective(
        self,
        satellites: Dict[str, Satellite],
        pixels: Dict[str, Pixel],
        costs: Dict,
        subproblem_range: range,
    ) -> None:
        """Add objective function to the model."""
        # 1. add cost operating satellites
        if self.type_of_flexibility >= 2:
            self.cost_operating_satellites = quicksum(
                [
                    round(satellite.cost_operation[q][t], 0) * self.Z[(s, q, t)]
                    for t in subproblem_range
                    for s, satellite in satellites.items()
                    for q, capacity in satellite.capacity.items()
                    if (s, q, t) in self.Z.keys() and capacity > 0
                ]
            )

        # 2. add cost served from satellite
        self.cost_serving_from_satellite = quicksum(
            [
                round(costs["satellite"][(s, k, t)]["total"], 0) * self.X[(s, k, t)]
                for t in subproblem_range
                for s in satellites.keys()
                for k in pixels.keys()
                if (s, k, t) in self.X.keys()
            ]
        )

        # 3. add cost served from dc
        self.cost_serving_from_dc = quicksum(
            [
                self.alpha * round(costs["dc"][(k, t)]["total"], 0) * self.W[(k, t)]
                for t in subproblem_range
                for k in pixels.keys()
            ]
        )

        if self.type_of_flexibility >= 2:
            self.objective = (
                self.cost_serving_from_dc
                + self.cost_serving_from_satellite
                + self.cost_operating_satellites
            )
        else:
            self.objective = (
                self.cost_serving_from_dc
                + self.cost_serving_from_satellite
            )

        self.model.setObjective(self.objective, GRB.MINIMIZE)

    def __add_constraints(
        self, satellites: Dict[str, Satellite], pixels: Dict[str, Pixel], fixed_y: Dict, subproblem_range: range,
    ) -> None:
        """Add constraints to the model."""
        # Constraints (3):
        logger.info(f"  Adding Constraints - One capacity must be selected")
        if self.type_of_flexibility >= 2:
            for s, satellite in satellites.items():
                if any(fixed_y[(s, q)] > 0.5 for q, capacity in satellite.capacity.items() if capacity > 0):
                    for t in subproblem_range:
                        self.model.addConstr(
                            quicksum([
                                self.Z[(s, q, t)]
                                for q, capacity in satellite.capacity.items()
                                if (s, q, t) in self.Z.keys()
                            ])
                            == 1
                        )

        # Constraints (5)
        logger.info(f"  Adding Constraints - Respect capacity limit")
        if self.type_of_flexibility == 1:
            for s, satellite in satellites.items():
                for q, capacity in satellite.capacity.items():
                    if fixed_y[(s, q)] > 0.5 and capacity > 0:
                        for t in subproblem_range:
                            nameConstraint = f"R_capacity_s{s}_t{t}"
                            self.model.addConstr(
                                    quicksum(
                                        [
                                            self.X[(s, k, t)]
                                            * round(self.fleet_size_required["satellite"][(s, k, t)]["fleet_size"], 1)
                                            for k in pixels.keys()
                                        ]
                                    )
                                    <= capacity,
                                    name=nameConstraint,
                            )
        else:
            for s, satellite in self.satellites.items():
                if any(fixed_y[(s, q)] > 0.5 for q, capacity in satellite.capacity.items() if capacity > 0):
                    for t in subproblem_range:
                        nameConstraint = f"R_capacity_s{s}_t{t}"
                        self.model.addConstr(
                            quicksum(
                                [
                                    self.X[(s, k, t)]
                                    * round(self.fleet_size_required["satellite"][(s, k, t)]["fleet_size"], 1)
                                    for k in self.pixels.keys()
                                ]
                            )
                            - quicksum(
                                [
                                    self.Z[(s, q, t)] * capacity
                                    for q, capacity in satellite.capacity.items()
                                    if (s, q, t) in self.Z.keys() if capacity > 0
                                ]
                            )
                            <= 0,
                            name=nameConstraint,
                        )

        logger.info(f"  Adding Constraints - Satisfy demand")
        for t in subproblem_range:
            for k in self.pixels.keys():
                nameConstraint = f"R_demand_k{k}_t{t}"
                self.model.addConstr(
                    quicksum(
                        [
                            self.X[(s, k, t)]
                            for s in self.satellites.keys()
                            if (s, k, t) in self.X.keys()
                        ]
                    )
                    + quicksum([self.W[(k, t)]])
                    >= 1,
                    name=nameConstraint,
                )

        # Valid inequalities:
        logger.info(f"  Adding Constraints - First valid inequalities")
        if self.type_of_flexibility >= 2:
            for s, satellite in self.satellites.items():
                if any(fixed_y[(s, q)] > 0.5 for q, capacity in satellite.capacity.items() if capacity > 0):
                    for t in subproblem_range:
                        for k in self.pixels.keys():
                            nameConstraint = f"R_valid_inequality_s{s}_t{t}"
                            self.model.addConstr(
                                self.X[(s, k, t)] <= 1 - self.Z[(s, '0', t)],
                                name=nameConstraint,
                            )

            logger.info(f"  Adding Constraints - Second valid inequalities")
            for s, satellite in self.satellites.items():
                if any(fixed_y[(s, q)] > 0.5 for q, capacity in satellite.capacity.items() if capacity > 0):
                    for t in subproblem_range:
                        for q, capacity in satellite.capacity.items():
                            if (s, q, t) in self.Z.keys():
                                if capacity > 0:
                                    nameConstraint = f"R_valid_inequality_2_s{s}_t{t}"
                                    self.model.addConstr(
                                        self.Z[(s, q, t)]
                                        <= quicksum(
                                            [
                                                self.X[(s, k, t)]
                                                for k in self.pixels.keys()
                                            ]
                                        ),
                                        name=nameConstraint,
                                    )

            # # New constraints
            # for s, satellite in self.satellites.items():
            #     if any(fixed_y[(s, q)] > 0.5 for q, capacity in satellite.capacity.items() if capacity > 0):
            #         previous_capacity = None
            #         for q, capacity in satellite.capacity.items():
            #             if capacity > 0:
            #                 if previous_capacity is not None:
            #                     for t in subproblem_range:
            #                         if (s, q, t) in self.Z.keys():
            #                             nameConstraint = f"Needed_capacity_s{s}_t{t}"
            #                             self.model.addConstr(
            #                                 quicksum(
            #                                     [
            #                                         self.X[(s, k, t)]
            #                                         * round(self.fleet_size_required["satellite"][(s, k, t)]["fleet_size"], 1)
            #                                         for k in self.pixels.keys()
            #                                     ]
            #                                 )
            #                                 >= self.Z[(s, q, t)] * (previous_capacity + 0.9),
            #                                 name=nameConstraint,
            #                             )
            #                 previous_capacity = capacity

    def solve_model(self, fixed_y: Dict[Any, float]) -> None:
        """Solve |self.ranges| models of the sub problem considering the fixed y."""
        start_period = 0
        all_t_run_time = 0
        all_t_total_cost = 0
        X_values = {}
        W_values = {}
        Z_values = {}

        for end_period in self.ranges:
            subproblem_range = range(start_period, end_period)

            # Create model
            self.model = gp.Model(name=f"SubProblem_{subproblem_range}")

            logger.info(f"[SUBPROBLEM] Creating model {subproblem_range}")
            logger.info(f"[SUBPROBLEM] Adding variables {subproblem_range}")
            self.__add_variables(self.satellites, self.pixels, fixed_y, subproblem_range)
            logger.info(f"[SUBPROBLEM] Adding objective {subproblem_range}")
            self.__add_objective(self.satellites, self.pixels, self.costs_serving, subproblem_range)
            logger.info(f"[SUBPROBLEM] Adding constraints {subproblem_range}")
            self.__add_constraints(self.satellites, self.pixels, fixed_y, subproblem_range)
            if self.warm_start_subproblems:
                self.model.update()
                self.__warm_start_subproblem(fixed_y, subproblem_range)

            logger.info(f"[SUBPROBLEM] Model created for {subproblem_range}")

            # Update model:
            self.model.Params.LogToConsole = 0
            self.model._total_cost = self.objective
            self.model.setParam("MIPGap", 0.001)
            self.model.setParam("MIPGapAbs", 1)
            # self.model.setParam("Heuristics", 0)
            # self.model.setParam("Cuts", 0)
            # self.model.setParam("MIPFocus", 2)
            # self.model.setParam('NumericFocus', 3)
            self.model.update()

            logger.info(f"[SUBPROBLEM] Solving subproblem for {subproblem_range}")
            start_time = time.time()
            self.model.optimize()
            logger.info(f"[SUBPROBLEM] Done solving subproblem for {subproblem_range}")
            run_time = round(time.time() - start_time, 3)
            all_t_run_time += run_time
            total_cost = self.model._total_cost.getValue()
            all_t_total_cost += total_cost

            # Saving total costs
            self.__compute_total_cost(subproblem_range)

            # Saving the values of X and W
            for t in subproblem_range:
                for k in self.pixels.keys():
                    for s, satellite in self.satellites.items():
                        if any(fixed_y[(s, q)] > 0.5 for q, capacity in satellite.capacity.items() if capacity > 0):
                            X_values[(s, k, t)] = self.X[(s, k, t)].x
                        else:
                            X_values[(s, k, t)] = 0

                    W_values[(k,t)] = self.W[(k,t)].x

            if self.type_of_flexibility >= 2:
                for s, satellite in self.satellites.items():
                    for t in subproblem_range:
                        for q_lower_key, q_lower_value in satellite.capacity.items():
                            if (s, q_lower_key, t) in self.Z:
                                Z_values[(s, q_lower_key, t)] = self.Z[(s, q_lower_key, t)].x
                            else:
                                Z_values[(s, q_lower_key, t)] = 0

            logger.info(
                f"[SUBPROBLEM] Subproblem in range {subproblem_range} is solved - Run time: {run_time} - Total cost: {total_cost}")
            self.model.dispose()
            start_period = end_period

        logger.info(
            f"[SUBPROBLEM] All subproblems in range are solved - Run time: {all_t_run_time} - Total cost: {all_t_total_cost}")

        return all_t_run_time, self.total_costs, X_values, W_values, Z_values

    def __compute_total_cost(self, subproblem_range):

        self.model._x = self.X
        self.model._z = self.Z
        self.model._w = self.W

        for t in subproblem_range:
            # 1. add cost for operating satellites
            cost_operating_satellites = 0
            if self.type_of_flexibility >= 2:
                cost_operating_satellites = np.sum(
                    [
                        satellite.cost_operation[q][t] * self.model._z[(s, q, t)].x
                        for s, satellite in self.satellites.items()
                        for q, capacity in satellite.capacity.items()
                        if (s, q, t) in self.Z.keys() if capacity > 0
                    ]
                )

            # 2. add cost served from satellite
            cost_serving_from_satellite = np.sum(
                [
                    self.costs_serving["satellite"][(s, k, t)]["total"] * self.model._x[(s, k, t)].x
                    for s in self.satellites.keys()
                    for k in self.pixels.keys()
                    if (s, k, t) in self.X.keys()
                ]
            )

            # 3. add cost served from dc
            cost_serving_from_dc = np.sum(
                [
                    self.costs_serving["dc"][(k, t)]["total"] * self.model._w[(k, t)].x
                    for k in self.pixels.keys()
                ]
            )

            self.total_costs[t] = cost_operating_satellites + cost_serving_from_satellite + cost_serving_from_dc

    def __warm_start_subproblem(self, fixed_y, subproblem_range: range):
        """Warm start the model."""
        logger.info(f"[SUBPROBLEM] Setting warm start")
        for (s, k, t) in self.X.keys():
            self.X[(s, k, t)].start = 0

        for (k, t) in self.W.keys():
            self.W[(k, t)].start = 0

        for t in subproblem_range:
            for k in self.pixels.keys():
                cost_list = [
                    self.scenario.get_cost_serving("satellite")[(s, k, t)]["total"]
                    for s, satellite in self.satellites.items()
                    if any(fixed_y[(s, q)] > 0.5 for q, capacity in satellite.capacity.items() if capacity > 0)
                ]

                min_index = np.argmin(cost_list) if cost_list else -1

                if min_index != -1:
                    valid_satellites = [
                        s for s, satellite in self.satellites.items()
                        if any(fixed_y[(s, q)] > 0.5 for q, capacity in satellite.capacity.items() if capacity > 0)
                    ]
                    selected_satellite = valid_satellites[min_index]
                else:
                    selected_satellite = None

                if selected_satellite is None or self.scenario.get_cost_serving("dc")[(k, t)]["total"] < \
                        self.scenario.get_cost_serving("satellite")[(selected_satellite, k, t)]["total"]:
                    self.W[(k, t)].start = 1
                else:
                    self.X[(selected_satellite, k, t)].start = 1

    def get_evaluation_metrics(self, fixed_y, X_values, W_values, Z_values):
        Z_operating_periods = {s: 0 for s in self.satellites.keys()}
        Z_total_capacity_limit = {s: 0 for s in self.satellites.keys()}
        Z_total_capacity_used = {s: 0 for s in self.satellites.keys()}
        Z_average_capacity_limit = {s: 0 for s in self.satellites.keys()}
        Z_average_capacity_used = {s: 0 for s in self.satellites.keys()}

        # (1) Satellite utilization:
        for s, satellite in self.satellites.items():
            if any(fixed_y[(s, q)] > 0.5 for q, capacity in satellite.capacity.items() if capacity > 0):
                for t in range(self.periods):
                    satellite_operates = False
                    for q, capacity in satellite.capacity.items():
                        if capacity > 0 and Z_values[(s, q, t)] > 0.5 and not satellite_operates:
                            # Increase number of operating periods and the total capacity limit:
                            satellite_operates = True
                            Z_operating_periods[s] += 1
                            Z_total_capacity_limit[s] += capacity

                    if satellite_operates:
                        # Compute total capacity used:
                        Z_total_capacity_used[s] += sum(
                            [
                                X_values[(s, k, t)]
                                * round(self.fleet_size_required["satellite"][(s, k, t)]["fleet_size"], 1)
                                for k in self.pixels.keys()
                            ]
                        )

                if satellite_operates:
                    Z_average_capacity_limit[s] = Z_total_capacity_limit[s] / Z_operating_periods[s]
                    Z_average_capacity_used[s] = Z_total_capacity_used[s] / Z_operating_periods[s]

        # (2) DC utilization:
        average_DC_fleet_used = sum(
            [
                W_values[(k, t)]
                * round(self.fleet_size_required["dc"][(k, t)]["fleet_size"], 1) # TODO: Check in Line 184 of Instance the format of fleet_size_required
                for k in self.pixels.keys()
                for t in range(self.periods)
            ]
        ) / self.periods

        """
        Notes:
            - We can compute metrics at a network level. For that, we could use the total capacity limit and the total capacity used.
            - In our paper, the 'demand' is the number of vehicles required, and that number depends from where pixels are served.
            - Do you have any idea on how to compute the percentage of the total 'demand' served from the satellites and the DC?
        """

        evaluation_metrics = {
            "Satellite_operating_periods": str(Z_operating_periods),
            "Satellite_total_capacity_limit": str(Z_total_capacity_limit),
            "Satellite_total_capacity_used": str(Z_total_capacity_used),
            "Satellite_average_capacity_limit": str(Z_average_capacity_limit),
            "Satellite_average_capacity_used": str(Z_average_capacity_used),
            "DC_average_fleet_used": str(average_DC_fleet_used),
        }

        return evaluation_metrics
