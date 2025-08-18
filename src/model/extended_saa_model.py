"""Module of class Extended SAA Model."""
from typing import Dict
import time

import gurobipy as gb
from gurobipy import GRB, quicksum

from src.classes import Satellite
from src.instance.instance import Instance
from src.instance.scenario import Scenario
from src.utils import LOGGER as logger


class ExtendedSAAModel:
    """Extended SAA Model."""

    def __init__(self, instance: Instance):
        self.model = gb.Model(name="Extended")

        # Params from instance and scenarios
        self.instance: Instance = instance
        self.periods: int = instance.periods
        self.satellites: Dict[str, Satellite] = instance.satellites
        self.scenarios: Dict[str, Scenario] = instance.scenarios

        # Params from instance
        self.type_of_flexibility: int = instance.type_of_flexibility
        self.is_continuous_x: bool = instance.is_continuous_x
        self.alpha: float = instance.alpha

        # self.__calculate_basecase_alpha()

        # Variables
        self.Z = {}
        self.Y = {}
        self.W = {} # Serving from the DC
        self.X = {} # Serving from satellites

        # Objective
        self.objective = None
        self.cost_installation_satellites = None
        self.cost_served_from_dc = None
        self.cost_served_from_satellite = None
        self.cost_operating_satellites = None

    def __calculate_basecase_alpha(self) -> None:
        """Calculate basecase alpha."""
        logger.info("[EXTENDED SAA] Calculate basecase alpha")
        # Initialize total cost and total count of pixels
        total_dc_serving_costs = 0
        total_pixels_dc = 0

        current_dc_serving_costs = 0

        # Iterate over all scenarios
        for n, scenario in self.scenarios.items():
            num_pixels = len(scenario.pixels)
            total_pixels_dc += num_pixels  # Accumulate the total number of pixels across all scenarios

            # Sum up the costs for each pixel and time period
            for t in range(self.periods):
                total_dc_serving_costs = 0
                for k in scenario.pixels.keys():
                    total_dc_serving_costs += round(scenario.get_cost_serving("dc")[(k, t)]["total"], 0)

                current_dc_serving_costs += total_dc_serving_costs/num_pixels

        current_dc_serving_costs = current_dc_serving_costs/ (len(self.scenarios) * self.periods)

        # Calculate the average cost over all scenarios, time periods, and pixels
        # The total pixels need to be multiplied by the number of periods only during the final averaging step
        # SELLY: current_dc_serving_costs = total_dc_serving_costs / (len(self.scenarios) * self.periods * total_pixels_dc)

        # Initialize total cost and total count of pixels
        total_satellites_serving_costs = 0
        total_pixels_s = 0  # This will track the total number of pixel-period-satellite combinations

        # Iterate over all scenarios
        current_satellites_serving_costs = 0
        for n, scenario in self.scenarios.items():
            num_pixels = len(scenario.pixels)
            total_pixels_s += num_pixels  # Accumulate the total number of pixels across all scenarios

            # Sum up the costs for each satellite, pixel, and time period
            for t in range(self.periods):
                total_satellites_serving_costs = 0
                for s in self.satellites.keys():
                    for k in scenario.pixels.keys():
                        total_satellites_serving_costs += round(
                            scenario.get_cost_serving("satellite")[(s, k, t)]["total"], 0)

                current_satellites_serving_costs += total_satellites_serving_costs/(len(self.satellites) * num_pixels)

        current_satellites_serving_costs = current_satellites_serving_costs /(len(self.scenarios)* self.periods)

        # Calculate the average cost over all scenarios, satellites, time periods, and pixels
        # SELLY: current_satellites_serving_costs = total_satellites_serving_costs / (
                # len(self.scenarios) * len(self.satellites) * self.periods * total_pixels_s)

        logger.info(f"[EXTENDED SAA] avg cost to serve 1 pixel in 1 time period from dc is {current_dc_serving_costs}")
        logger.info(f"[EXTENDED SAA] avg cost to serve 1 pixel in 1 time period from a satellite is {current_satellites_serving_costs}")
        basecase_alpha = current_dc_serving_costs / current_satellites_serving_costs
        logger.info(f"[EXTENDED SAA] basecase_alpha {basecase_alpha}")

    def build(self) -> None:
        """Build the model."""
        logger.info("[EXTENDED SAA] Build model")
        self.__add_variables(self.satellites, self.scenarios)
        self.__add_objective(self.satellites, self.scenarios)
        self.__add_constraints(self.satellites, self.scenarios)

        self.model.update()
        logger.info("[EXTENDED SAA] Model built")

    def __add_variables(
        self, satellites: Dict[str, Satellite], scenarios: Dict[str, Scenario]
    ) -> None:
        """Add variables to model."""
        type_variable = GRB.CONTINUOUS if self.is_continuous_x else GRB.BINARY

        # 1. add variable Z: binary variable to decide if a satellite is operating in a period with a given capacity
        if self.type_of_flexibility >= 2:
            self.Z = dict(
                [
                    (
                        (s, q, n, t),
                        self.model.addVar(
                            vtype=GRB.BINARY, name=f"Z_s{s}_q{q}_n{n}_t{t}", lb=0, ub=1
                        ),
                    )
                    for s, satellite in satellites.items()
                    for q, capacity in satellite.capacity.items()
                    for t in range(self.periods)
                    for n in scenarios.keys()
                ]
            )
        logger.info(f"[EXTENDED SAA] Number of variables Z: {len(self.Z)}")

        # 2. add variable X: binary variable to decide if a satellite is used to serve a pixel
        self.X = dict(
            [
                (
                    (s, k, n, t),
                    self.model.addVar(
                        vtype=type_variable, name=f"X_s{s}_k{k}_n{n}_t{t}", lb=0, ub=1
                    ),
                )
                for s in satellites.keys()
                for n, scenario in scenarios.items()
                for k in scenario.pixels.keys()
                for t in range(self.periods)
            ]
        )
        logger.info(f"[EXTENDED SAA] Number of variables X: {len(self.X)}")

        # 3. add variable W: binary variable to decide if a pixel is served from dc
        self.W = dict(
            [
                (
                    (k, n, t),
                    self.model.addVar(
                        vtype=type_variable, name=f"W_k{k}_n{n}_t{t}", lb=0, ub=1
                    ),
                )
                for n, scenario in scenarios.items()
                for k in scenario.pixels.keys()
                for t in range(self.periods)
            ]
        )
        logger.info(f"[EXTENDED SAA] Number of variables W: {len(self.W)}")

        # 4. add variable Y: binary variable to decide if a satellite is open or not
        self.Y = dict(
            [
                (
                    (s, q),
                    self.model.addVar(vtype=GRB.BINARY, name=f"Y_s{s}_q{q}"),
                )
                for s, satellite in satellites.items()
                for q, capacity in satellite.capacity.items()
            ]
        )
        logger.info(f"[EXTENDED SAA] Number of variables Y: {len(self.Y)}")

        self.model._X = self.X
        self.model._Y = self.Y
        self.model._W = self.W
        self.model._Z = self.Z

    def __add_objective(
        self,
        satellites: Dict[str, Satellite],
        scenarios: Dict[str, Scenario],
    ) -> None:
        """Add objective to model."""
        # 1. add cost installation satellites
        logger.info("[EXTENDED SAA] Add objective")
        self.cost_installation_satellites = quicksum(
            [
                round(satellite.cost_fixed[q], 0) * self.Y[(s, q)]
                for s, satellite in satellites.items()
                for q, capacity in satellite.capacity.items() if capacity > 0
            ]
        )

        # 2. add cost operating satellites
        if self.type_of_flexibility == 1:
            self.cost_operating_satellites = quicksum(
                [
                    round(satellite.cost_operation[q][t], 0) * self.Y[(s, q)]
                    for s, satellite in satellites.items()
                    for q, capacity in satellite.capacity.items() if capacity > 0
                    for t in range(self.periods)
                    for n in scenarios.keys()
                ]
            )
        else:
            self.cost_operating_satellites = quicksum(
                [
                    round(satellite.cost_operation[q][t], 0) * self.Z[(s, q, n, t)]
                    for s, satellite in satellites.items()
                    for q, capacity in satellite.capacity.items() if capacity > 0
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
                self.alpha * round(scenario.get_cost_serving("dc")[(k, t)]["total"], 0) * self.W[(k, n, t)]
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
        self.model._total_cost = self.objective
        self.model.setObjective(self.objective, GRB.MINIMIZE)

    def __add_constraints(
        self,
        satellites: Dict[str, Satellite],
        scenarios: Dict[str, Scenario],
    ) -> None:
        """Add constraints to model."""
        logger.info("[EXTENDED SAA] Adding constraints:")
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
        logger.info("[EXTENDED SAA] Add constraints maximum operating satellite")
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
                                if q_y_key > q_z_key > 0:
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


    def solve(self):
        """Solve the model."""
        logger.info("[EXTENDED SAA] Solving model")
        start_time = time.time()
        self.model.optimize()
        logger.info("[EXTENDED SAA] Model solved")
        results = {
            "actual_run_time": round(time.time() - start_time, 3),
            "optimality_gap": round(100 * self.model.MIPGap, 3),
            "objective_value": round(self.model._total_cost.getValue(), 3),
            "best_bound_value": round(self.model.ObjBound, 3),
        }
        return results

    def set_params(self, params: Dict[str, int]) -> None:
        """Set parameters to model."""
        for key, item in params.items():
            self.model.setParam(key, item)
