"""Module for the warm start of the stochastic model."""
from typing import Any, Dict, Tuple

import gurobipy as gp
import numpy as np
from gurobipy import GRB, quicksum

from src.classes import Satellite
from src.instance.instance import Instance
from src.instance.scenario import Scenario
from src.model.sub_problem import SubProblem
from src.utils import LOGGER as logger


class WarmStart:
    """Class for the warm start problem of the master model."""

    def __init__(self, instance: Instance, reformulated: bool, split: int, warm_start_subproblems: bool) -> None:
        """Initialize the master problem."""
        self.model: gp.Model = gp.Model(name="WarmStart")

        # Instance
        self.instance = instance
        self.periods: int = instance.periods
        self.satellites: Dict[str, Satellite] = instance.satellites
        self.scenarios: Dict[str, Scenario] = instance.scenarios
        self.type_of_flexibility: int = instance.type_of_flexibility

        self.reformulated = reformulated
        self.split = split
        self.warm_start_subproblems = warm_start_subproblems

        # Variables
        self.Y = {}

        if self.reformulated:
            self.X = {}
            self.W = {}
            if self.type_of_flexibility >= 2:
                self.Z = {}

        # Objective
        self.objective = None
        self.cost_installation_satellites = None
        self.cost_operating_satellites = None
        self.cost_served_from_satellite = None
        self.cost_served_from_dc = None
        self.cost_second_stage = None

    def build(self) -> None:
        """Build the warm start problem."""
        logger.info("[WARM START] Building warm start problem")
        self.__add_variables(self.satellites, self.scenarios)
        self.__add_objective(self.satellites, self.scenarios)
        self.__add_constraints(self.satellites, self.scenarios)
        self.model.update()

        self.model._start_time = 0
        self.model.update()
        logger.info("[WARM START] Warm start problem built")

    def __add_variables(
            self, satellites: Dict[str, Satellite], scenarios: Dict[str, Scenario]
    ) -> None:
        """Add variables to the model."""
        logger.info("[WARM START] Adding variables to warm start problem")
        # 1. add variable Z: binary variable to decide if a satellite is operating in a period with a given capacity
        if self.instance.type_of_flexibility >= 2:
            self.Z = dict(
                [
                    (
                        (s, q, n, t),
                        self.model.addVar(
                            vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f"Z_s{s}_q{q}_n{n}_t{t}"
                        ),
                    )
                    for s, satellite in satellites.items()
                    for q in satellite.capacity.keys()
                    for t in range(self.periods)
                    for n in scenarios.keys()
                ]
            )
        # logger.info(f"[WARM START] Number of variables Z: {len(self.Z)}")

        # 2. add variable X: binary variable to decide if a satellite is used to serve a pixel
        self.X = dict(
            [
                (
                    (s, k, n, t),
                    self.model.addVar(
                        vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f"X_s{s}_k{k}_n{n}_t{t}",
                    ),
                )
                for s in satellites.keys()
                for n, scenario in scenarios.items()
                for k in scenario.pixels.keys()
                for t in range(self.periods)
            ]
        )
        # logger.info(f"[WARM START] Number of variables X: {len(self.X)}")

        # 3. add variable W: binary variable to decide if a pixel is served from dc
        self.W = dict(
            [
                (
                    (k, n, t),
                    self.model.addVar(
                        vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f"W_k{k}_n{n}_t{t}",
                    ),
                )
                for n, scenario in scenarios.items()
                for k in scenario.pixels.keys()
                for t in range(self.periods)
            ]
        )
        # logger.info(f"[WARM START] Number of variables W: {len(self.W)}")

        # 4. add variable Y: binary variable to decide if a satellite is open or not
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
        # logger.info(f"[WARM START] Number of variables Y: {len(self.Y)}")

        self.model._Y = self.Y

        if self.reformulated:
            self.model._X = self.X
            self.model._W = self.W
            if self.type_of_flexibility >= 2:
                self.model._Z = self.Z

    def __add_objective(
            self, satellites: Dict[str, Satellite], scenarios: Dict[str, Scenario]
    ) -> None:
        """Add objective function to the model."""
        logger.info("[WARM START] Adding objective function to warm start model")
        # 1. add cost installation satellites
        self.cost_installation_satellites = quicksum(
            [
                round(satellite.cost_fixed[q], 0) * self.Y[(s, q)]
                for s, satellite in satellites.items()
                for q, capacity in satellite.capacity.items()
                if capacity > 0
            ]
        )

        # 2. add cost operating satellites
        if self.instance.type_of_flexibility == 1:
            self.cost_operating_satellites = quicksum(
                [
                    round(satellite.cost_operation[q][t], 0) * self.Y[(s, q)]
                    for s, satellite in satellites.items()
                    for q, capacity in satellite.capacity.items()
                    if capacity > 0
                    for t in range(self.periods)
                    for n in scenarios.keys()
                ]
            )
        else:
            self.cost_operating_satellites = quicksum(
                [
                    round(satellite.cost_operation[q][t], 0) * self.Z[(s, q, n, t)]
                    for s, satellite in satellites.items()
                    for q, capacity in satellite.capacity.items()
                    if capacity > 0
                    for t in range(self.periods)
                    for n in scenarios.keys()
                ]
            )

        # 3. add cost served from satellite
        self.cost_served_from_satellite = quicksum(
            [
                round(scenario.get_cost_serving("satellite")[(s, k, t)]["total"], 0)
                * self.X[(s, k, n, t)]
                for s in satellites.keys()
                for n, scenario in scenarios.items()
                for k in scenario.pixels.keys()
                for t in range(self.periods)
            ]
        )

        # 4. add cost served from dc
        self.cost_served_from_dc = quicksum(
            [
                round(scenario.get_cost_serving("dc")[(k, t)]["total"], 0) * self.W[(k, n, t)]
                for n, scenario in scenarios.items()
                for k in scenario.pixels.keys()
                for t in range(self.periods)
            ]
        )

        self.objective = self.cost_installation_satellites + (1 / len(scenarios)) * (
                self.cost_operating_satellites
                + self.cost_served_from_dc
                + self.cost_served_from_satellite
        )
        self.model.setObjective(self.objective, GRB.MINIMIZE)

        self.cost_second_stage = self.cost_operating_satellites + self.cost_served_from_dc + self.cost_served_from_satellite

        self.model._total_cost = self.objective
        self.model._cost_installation_satellites = self.cost_installation_satellites
        self.model._cost_second_stage = self.cost_second_stage

    def __add_constraints(
            self,
            satellites: Dict[str, Satellite],
            scenarios: Dict[str, Scenario],
    ) -> None:
        """Add constraints to model."""
        logger.info("[WARM START] Adding constraints:")
        self.__add_constr_A_1(satellites)

        if self.type_of_flexibility >= 2:
            self.__add_constr_A_2(satellites, scenarios)
            self.__add_constr_A_3(satellites, scenarios)
            if self.type_of_flexibility == 3:
                self.__add_constr_deactivate_z(satellites, scenarios)

        self.__add_constr_A_4(satellites, scenarios)
        self.__add_constr_A_5(satellites, scenarios)
        self.__add_valid_inequalities(satellites, scenarios)

    def __add_constr_A_1(
            self, satellites: Dict[str, Satellite]
    ) -> None:
        logger.info("   Adding constraints A.1 - Installing satellites")
        for s, satellite in satellites.items():
            nameConstraint = f"R_Open_s{s}"
            self.model.addConstr(
                quicksum([self.Y[(s, q)] for q in satellite.capacity.keys()]) == 1,
                name=nameConstraint,
                )

    def __add_constr_A_2(
            self, satellites: Dict[str, Satellite], scenarios: Dict[str, Scenario]
    ) -> None:
        logger.info("   Adding constraints A.2 - Operating satellites")
        for s, satellite in satellites.items():
            for n in scenarios.keys():
                for t in range(self.periods):
                    nameConstraint = f"R_activation_s{s}_n{n}_t{t}"
                    self.model.addConstr(
                        quicksum([self.Z[(s, q, n, t)] for q, capacity in satellite.capacity.items()])
                        == 1,
                        name=nameConstraint,
                        )

    def __add_constr_A_3(
            self, satellites: Dict[str, Satellite], scenarios: Dict[str, Scenario]
    ) -> None:
        """Add Constraints A.3."""
        logger.info("[WARM START] Add constraints maximum operating satellite")
        for t in range(self.periods):
            for n in scenarios.keys():
                for s, satellite in satellites.items():
                    max_capacity = max(satellite.capacity.values())
                    for q, q_value in satellite.capacity.items():
                        if q_value < max_capacity:
                            nameConstraint = f"R_Operating_s{s}_q{q}_n{n}_t{t}"
                            q_higher_values = [
                                q_higher
                                for q_higher, q_higher_value in satellite.capacity.items()
                                if q_higher_value > q_value
                            ]
                            self.model.addConstr(
                                quicksum(
                                    [
                                        self.Z[(s, q_higher, n, t)]
                                        for q_higher in q_higher_values
                                    ]
                                )
                                <= 1 - self.Y[(s, q)],
                                name=nameConstraint,
                                )

    def __add_constr_deactivate_z(
            self, satellites: Dict[str, Satellite], scenarios: Dict[str, Scenario]
    ) -> None:
        """Add Constraints Deactivate z"""
        logger.info("   Add constraints A.3.1 - Deactivate Z variables")
        for t in range(self.periods):
            for n in scenarios.keys():
                for s, satellite in satellites.items():
                    for q_y_key, q_y_value in satellite.capacity.items():
                        if q_y_value > 0:
                            for q_z_key, q_z_value in satellite.capacity.items():
                                if q_y_value > q_z_value > 0:
                                    nameConstraint = f"Z_Deactivated_s{s}_q{q_z_key}"
                                    self.model.addConstr(self.Z[(s, q_z_key, n, t)] <= 1 - self.Y[(s, q_y_key)],
                                                         name=nameConstraint,
                                                         )

    def __add_constr_A_4(
            self,
            satellites: Dict[str, Satellite],
            scenarios: Dict[str, Scenario],
    ) -> None:
        logger.info("   Adding constraints A.4 - Capacity limit")
        for t in range(self.periods):
            for s, satellite in satellites.items():
                for n, scenario in scenarios.items():
                    pixels = scenario.pixels
                    fleet_size_required = scenario.get_fleet_size_required("satellite")
                    nameConstraint = f"R_capacity_s{s}_n{n}_t{t}"
                    if self.type_of_flexibility >= 2:
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
                                    self.Z[(s, q, n, t)] * capacity
                                    for q, capacity in satellite.capacity.items() if capacity > 0
                                ]
                            )
                            <= 0,
                            name=nameConstraint,
                            )
                    else:
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
                                    for q, capacity in satellite.capacity.items() if capacity > 0
                                ]
                            )
                            <= 0,
                            name=nameConstraint,
                            )

    def __add_constr_A_5(
            self, satellites: Dict[str, Satellite], scenarios: Dict[str, Scenario]
    ):
        logger.info("   Adding constraints A.5 - Demand satisfaction")
        for n, scenario in scenarios.items():
            for t in range(self.periods):
                for k in scenario.pixels.keys():
                    nameConstraint = f"R_demand_k{k}_n{n}_t{t}"
                    self.model.addConstr(
                        quicksum([self.X[(s, k, n, t)] for s in satellites.keys()])
                        + quicksum([self.W[(k, n, t)]])
                        >= 1,
                        name=nameConstraint,
                        )

    def __add_valid_inequalities(
            self, satellites: Dict[str, Satellite], scenarios: Dict[str, Scenario]
    ):
        logger.info("   Adding valid inequalities")
        for n, scenario in scenarios.items():
            for t in range(self.periods):
                for s, satellite in satellites.items():
                    for k in scenario.pixels.keys():
                        nameConstraint = f"R_valid_inequality_s{s}_t{t}"
                        self.model.addConstr(
                            self.X[(s, k, n, t)]
                            <= quicksum(
                                [
                                    self.Y[(s, q)]
                                    for q, capacity in satellite.capacity.items()
                                    if capacity > 0
                                ]
                            ),
                            name=nameConstraint,
                            )

    def set_params(self, params: Dict[str, int]):
        """Set params to model."""
        logger.info(f"[WARM START] Set params to model {params}")
        for key, item in params.items():
            self.model.setParam(key, item)

    def __warm_start(self):
        """Warm start the warm start model."""
        logger.info(f"[WARM START] Setting warm start")
        for (s, q) in self.Y.keys():
            self.Y[(s, q)].start = 0
            # Get the satellite object corresponding to 's'
            satellite = self.satellites[s]

            # Determine if the current q value corresponds to the maximum capacity for satellite s
            is_max_capacity = satellite.capacity[q] == max(satellite.capacity.values())

            # Set the start value accordingly
            if is_max_capacity:
                self.Y[(s, q)].start = int(is_max_capacity)

        # for (s, k, n, t) in self.X.keys():
        #     self.X[(s, k, n, t)].start = 0

        for (k, n, t) in self.W.keys():
            self.W[(k, n, t)].start = 0

        for n, scenario in self.scenarios.items():
            for t in range(self.periods):
                for k in scenario.pixels.keys():
                    cost_list = [
                        scenario.get_cost_serving("satellite")[(s, k, t)]["total"]
                        for s, satellite in self.satellites.items()
                    ]
                    min_index = np.argmin(cost_list) if cost_list else -1

                    if min_index != -1:
                        valid_satellites = [
                            s for s, satellite in self.satellites.items()
                        ]
                        selected_satellite = valid_satellites[min_index]
                    else:
                        selected_satellite = None

                    if selected_satellite is None or scenario.get_cost_serving("dc")[(k, t)]["total"] < \
                            scenario.get_cost_serving("satellite")[(selected_satellite, k, t)]["total"]:
                        self.W[(k, n, t)].start = 1.0
                    # else:
                    #     self.X[(selected_satellite, k, n, t)].start = 1.0
                    #     self.W[(k, n, t)].start = 1.0

    def get_objective_value(self):
        """Get the objective value of the model."""
        logger.info("[MASTER PROBLEM] Getting objective value")
        return self.model._total_cost.getValue()

    def get_y_solution(self):
        """Get the Y solution of the model."""
        logger.info("[WARM START] Getting Y solution")
        return {keys: value.X for keys, value in self.Y.items()}

    def get_x_solution(self):
        """Get the X solution of the model."""
        logger.info("[WARM START] Getting X solution")
        return {keys: value.X for keys, value in self.X.items()}

    def get_w_solution(self):
        """Get the W solution of the model."""
        logger.info("[WARM START] Getting W solution")
        return {keys: value.X for keys, value in self.W.items()}

    def get_θ_solution(self):
        """Get the θ solution of the model."""
        logger.info("[WARM START] Getting θ solution")
        θ_value = {}
        for n, scenario in self.scenarios.items():
            for t in range(self.periods):
                θ_value[(n,t)] = sum(
                    [
                        scenario.get_cost_serving("satellite")[(s, k, t)]["total"]
                        * self.X[(s, k, n, t)].X
                        for s in self.satellites.keys()
                        for k in scenario.pixels.keys()
                    ]
                ) + sum(
                    [
                        scenario.get_cost_serving("dc")[(k, t)]["total"] * self.W[(k, n, t)].X
                        for k in scenario.pixels.keys()
                    ]
                )

                if self.type_of_flexibility >= 2:

                    θ_value[(n,t)] += sum(
                        [
                            satellite.cost_operation[q][t] * self.Z[(s, q, n, t)].X
                            for s, satellite in self.satellites.items()
                            for q, capacity in satellite.capacity.items()
                            if capacity > 0
                        ]
                    )

        return θ_value
