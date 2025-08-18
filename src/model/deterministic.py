"""Model Multiperiod Flexible Operation."""
from typing import Any, Dict

import gurobipy as gb
from gurobipy import GRB, quicksum

from src.classes import Pixel, Satellite
from src.instance.instance import Instance
from src.utils import LOGGER as logger


class FlexibilityModel:
    """Model Multiperiod Flexible Operation."""

    def __init__(self, instance: Instance, id_scenario: str = "expected"):
        self.model = gb.Model(name="Deterministic")

        # Params from instance and one scenario
        self.instance: Instance = instance
        self.periods: int = instance.periods
        self.satellites: Dict[str, Satellite] = instance.satellites
        self.pixels: Dict[str, Pixel] = instance.scenarios[id_scenario].pixels
        self.cost_serving: Dict[str, Dict] = instance.scenarios[
            id_scenario
        ].get_cost_serving()
        self.fleet_size_required: Dict[str, Dict] = instance.scenarios[
            id_scenario
        ].get_fleet_size_required()

        # Params from instance
        self.type_of_flexibility: int = instance.type_of_flexibility
        self.is_continuous_x: bool = instance.is_continuous_x

        # variables
        self.Z = {}
        self.Y = {}
        self.W = {}
        self.X = {}

        # objective
        self.objective = None
        self.cost_installation_satellites = None
        self.cost_served_from_dc = None
        self.cost_served_from_satellite = None
        self.cost_operating_satellites = None

    def build(self) -> None:
        """Build the model."""
        logger.info("[DETERMINISTIC] Build model")
        self.__add_variables(self.satellites, self.pixels)
        self.__add_objective(self.satellites, self.pixels, self.cost_serving)
        self.__add_constraints(
            self.satellites,
            self.pixels,
            self.fleet_size_required,
        )

        self.model.update()
        logger.info("[DETERMINISTIC] Model built")

    def __add_variables(
        self, satellites: Dict[str, Satellite], pixels: Dict[str, Pixel]
    ) -> None:
        """Add variables to model."""
        type_variable = GRB.CONTINUOUS if self.is_continuous_x else GRB.BINARY

        # 1. add variable Z: binary variable to decide if a satellite is operating in a period with a capacity
        if self.type_of_flexibility == 2:
            self.Z = dict(
                [
                    (
                        (s, q, t),
                        self.model.addVar(vtype=GRB.BINARY, name=f"Z_s{s}_q{q}_t{t}"),
                    )
                    for s, satellite in satellites.items()
                    for q in satellite.capacity.keys()
                    for t in range(self.periods)
                ]
            )
        logger.info(f"[DETERMINISTIC] Number of variables Z: {len(self.Z)}")

        # 2. add variable X: binary variable to decide if a satellite is used to serve a pixel
        self.X = dict(
            [
                (
                    (s, k, t),
                    self.model.addVar(
                        vtype=type_variable, name=f"X_s{s}_k{k}_t{t}", lb=0, ub=1
                    ),
                )
                for s in satellites.keys()
                for k in pixels.keys()
                for t in range(self.periods)
            ]
        )
        logger.info(f"[DETERMINISTIC] Number of variables X: {len(self.X)}")

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
                for t in range(self.periods)
            ]
        )
        logger.info(f"[DETERMINISTIC] Number of variables W: {len(self.W)}")

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
        logger.info(f"[DETERMINISTIC] Number of variables Y: {len(self.Y)}")

        self.model._X = self.Z
        self.model._Y = self.Y
        self.model._W = self.W
        self.model._Z = self.X

    def __add_objective(
        self,
        satellites: Dict[str, Satellite],
        pixels: Dict[str, Pixel],
        costs: Dict[str, Any],
    ) -> None:
        """Add objective to model."""
        # 1. add cost installation satellites
        logger.info("[DETERMINISTIC] Add objective")
        self.cost_installation_satellites = quicksum(
            [
                satellite.cost_fixed[q] * self.Y[(s, q)]
                for s, satellite in satellites.items()
                for q in satellite.capacity.keys()
            ]
        )

        # 2. add cost operating satellites
        if self.type_of_flexibility == 1:
            self.cost_operating_satellites = quicksum(
                [
                    satellite.cost_operation[q][t] * self.Y[(s, q)]
                    for s, satellite in satellites.items()
                    for q in satellite.capacity.keys()
                    for t in range(self.periods)
                ]
            )
        else:
            self.cost_operating_satellites = quicksum(
                [
                    satellite.cost_operation[q][t] * self.Z[(s, q, t)]
                    for s, satellite in satellites.items()
                    for q in satellite.capacity.keys()
                    for t in range(self.periods)
                ]
            )

        # 3. add cost served from satellite
        self.cost_served_from_satellite = quicksum(
            [
                costs["satellite"][(s, k, t)]["total"] * self.X[(s, k, t)]
                for s in satellites.keys()
                for k in pixels.keys()
                for t in range(self.periods)
            ]
        )

        # 4. add cost served from dc
        self.cost_served_from_dc = quicksum(
            [
                costs["dc"][(k, t)]["total"] * self.W[(k, t)]
                for k in pixels.keys()
                for t in range(self.periods)
            ]
        )

        self.objective = (
            self.cost_installation_satellites
            + self.cost_served_from_dc
            + self.cost_served_from_satellite
            + self.cost_operating_satellites
        )
        self.model.setObjective(self.objective, GRB.MINIMIZE)

    def __add_constraints(
        self,
        satellites: Dict[str, Satellite],
        pixels: Dict[str, Pixel],
        fleet_size_required: Dict[str, Any],
    ) -> None:
        """Add constraints to model."""

        if self.type_of_flexibility == 2:
            self.__add_constr_activation_satellite(satellites)
            self.__add_constr_operating_capacity_satellite(satellites)

        self.__add_constr_installation_satellite(satellites)
        self.__add_constr_capacity_satellite(satellites, pixels, fleet_size_required)
        self.__add_constr_demand_satified(satellites, pixels)

        # dummi constraints
        # self.__add_constr_dummi(satellites)
        # self.__add_constr_fixed_w(pixels, 0)

    def __add_constr_fixed_w(self, pixels: Dict[str, Pixel], value: int) -> None:
        """Add constraint fixed W."""
        logger.info("[DETERMINISTIC] DUMMI Add constraint fixed W")
        for k in pixels.keys():
            for t in range(self.periods):
                nameConstraint = f"R_fixed_w_k{k}_t{t}"
                self.model.addConstr(
                    self.W[(k, t)] == value,
                    name=nameConstraint,
                )

    def __add_constr_installation_satellite(
        self, satellites: Dict[str, Satellite]
    ) -> None:
        """Add constraint installation satellite."""
        logger.info("[DETERMINISTIC] Add constraint installation satellite")
        for s, satellite in satellites.items():
            nameConstraint = f"R_Open_s{s}"
            self.model.addConstr(
                quicksum([self.Y[(s, q)] for q in satellite.capacity.keys()]) <= 1,
                name=nameConstraint,
            )

    def __add_constr_activation_satellite(
        self, satellites: Dict[str, Satellite]
    ) -> None:
        """Add constraint activation satellite."""
        logger.info("[DETERMINISTIC] Add constraint activation satellite")
        for s, satellite in satellites.items():
            for t in range(self.periods):
                nameConstraint = f"R_activation_s{s}_t{t}"
                self.model.addConstr(
                    quicksum([self.Z[(s, q, t)] for q in satellite.capacity.keys()])
                    <= quicksum([self.Y[(s, q)] for q in satellite.capacity.keys()]),
                    name=nameConstraint,
                )

    def __add_constr_operating_capacity_satellite(
        self, satellites: Dict[str, Satellite]
    ) -> None:
        """Add constraint operating satellite."""
        logger.info("[DETERMINISTIC] Add constraint operating satellite")
        for t in range(self.periods):
            for s, satellite in satellites.items():
                max_capacity = max(satellite.capacity.values())
                for q, q_value in satellite.capacity.items():
                    if q_value < max_capacity:
                        nameConstraint = f"R_Operating_s{s}_q{q}_t{t}"
                        q_higher_values = [
                            q_higher
                            for q_higher, q_higher_value in satellite.capacity.items()
                            if q_higher_value > q_value
                        ]
                        self.model.addConstr(
                            quicksum(
                                [
                                    self.Z[(s, q_higher, t)]
                                    for q_higher in q_higher_values
                                ]
                            )
                            <= 1 - self.Y[(s, q)],
                            name=nameConstraint,
                        )

    def __add_constr_capacity_satellite(
        self,
        satellites: Dict[str, Satellite],
        pixels: Dict[str, Pixel],
        fleet_size_required: Dict[str, Any],
    ) -> None:
        """Add constraint capacity satellite."""
        logger.info("[DETERMINISTIC] Add constraint capacity satellite")
        for t in range(self.periods):
            for s, satellite in satellites.items():
                nameConstraint = f"R_capacity_s{s}_t{t}"
                if self.type_of_flexibility == 2:
                    self.model.addConstr(
                        quicksum(
                            [
                                self.X[(s, k, t)]
                                * fleet_size_required["satellite"][(s, k, t)][
                                    "fleet_size"
                                ]
                                for k in pixels.keys()
                            ]
                        )
                        - quicksum(
                            [
                                self.Z[(s, q, t)] * capacity
                                for q, capacity in satellite.capacity.items()
                            ]
                        )
                        <= 0,
                        name=nameConstraint,
                    )
                else:
                    self.model.addConstr(
                        quicksum(
                            [
                                self.X[(s, k, t)]
                                * fleet_size_required["satellite"][(s, k, t)][
                                    "fleet_size"
                                ]
                                for k in pixels.keys()
                            ]
                        )
                        - quicksum(
                            [
                                self.Y[(s, q)] * capacity
                                for q, capacity in satellite.capacity.items()
                            ]
                        )
                        <= 0,
                        name=nameConstraint,
                    )

    def __add_constr_demand_satified(
        self, satellites: Dict[str, Satellite], pixels: Dict[str, Pixel]
    ):
        """Add constraint demand satisfied."""
        logger.info("[DETERMINISTIC] Add constraint demand satisfied")
        for t in range(self.periods):
            for k in pixels.keys():
                nameConstraint = f"R_demand_k{k}_t{t}"
                self.model.addConstr(
                    quicksum([self.X[(s, k, t)] for s in satellites.keys()])
                    + quicksum([self.W[(k, t)]])
                    == 1,
                    name=nameConstraint,
                )

    def __add_constr_dummi(self, satellites: Dict[str, Satellite]) -> None:
        """Add constraint dumming."""
        logger.info("[DETERMINISTIC] Add constraint dummi")
        nameConstraint = "R_Dumming"
        self.model.addConstr(
            quicksum(
                [
                    self.Y[(s, q)]
                    for s, satellite in satellites.items()
                    for q in satellite.capacity.keys()
                ]
            )
            == 1,
            name=nameConstraint,
        )

    def solve(self):
        """Solve the model."""
        logger.info("[DETERMINISTIC] Solve model")
        self.model.optimize()
        logger.info("[DETERMINISTIC] Model solved")

    def set_params(self, params: Dict[str, int]) -> None:
        """Set parameters to model."""
        for key, item in params.items():
            self.model.setParam(key, item)
