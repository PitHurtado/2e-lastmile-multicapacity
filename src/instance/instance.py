"""Module of class Instance."""
import json
import os
import random
from typing import Dict, List

from src.classes import Pixel, Satellite, Vehicle
from src.constants import PATH_SAMPLING_SCENARIO
from src.continuous_approximation.continuous_approximation import (
    ContinuousApproximation,
)
from src.continuous_approximation.fleet_size import (
    ContinuousApproximationConfig,
    get_cost_from_continuous_approximation,
)
from src.etl import Data
from src.instance.scenario import Scenario
from src.utils import LOGGER as logger


class Instance:
    """Class to define Instance"""

    def __init__(
        self,
        id_instance: str,
        capacity_satellites: Dict[str, int],
        is_continuous_x: bool,
        alpha: float,
        type_of_flexibility: int,
        periods: int,
        N: int,
        is_evaluation: bool,
        type_of_cost_serving: int,
        scenario_evaluation_id: int,
    ):  # pylint: disable=too-many-arguments

        # Parameters
        self.id_instance = id_instance
        self.capacity_satellites = capacity_satellites
        self.is_continuous_x = is_continuous_x
        self.alpha = alpha
        self.type_of_flexibility = type_of_flexibility
        self.type_of_cost_serving = type_of_cost_serving
        self.periods = periods
        self.N = N
        self.is_evaluation = is_evaluation
        self.matrix_satellite_pixels = dict()
        self.matrix_dc_pixels = dict()
        self.scenario_evaluation_id = scenario_evaluation_id

        # Read vehicles and satellites
        self.vehicles: Dict[str, Vehicle] = self.__read_vehicles()
        self.satellites: Dict[str, Satellite] = self.__read_satellites()

        # Create object for loading CA costs
        self.computer_fleet_size = (
            ContinuousApproximationConfig(
                periods=periods,
                small_vehicle=self.vehicles["small"],
                large_vehicle=self.vehicles["large"],
            )
            if type_of_cost_serving == 1
            else ContinuousApproximation(
                periods=periods,
                satellites=self.satellites,
                matrixes={
                    "dc": Data.load_matrix_from_dc()["distance"],
                    "satellite": Data.load_matrix_from_satellite()["distance"],
                },
                vehicles=self.vehicles,
            )
        )

        # Read the demand of each scenario
        self.scenarios: Dict[str, Scenario] = self.__compute_scenarios()
        self.scenarios_ids = list(key for key in self.scenarios)
        self.__update_satellites()

    def __str__(self):
        return (
            f"---- Instance ----\n"
            f"ID of the instance: {self.id_instance}\n"
            f"Capacity of satellites: {self.capacity_satellites}\n"
            f"Is continuous X: {self.is_continuous_x}\n"
            f"Alpha: {self.alpha}\n"
            f"Beta: {self.beta}\n"
            f"Type of flexibility: {self.type_of_flexibility}\n"
            f"Periods: {self.periods}\n"
            f"N: {self.N}\n"
            f"Is evaluation: {self.is_evaluation}\n"
            f"Quantity of satellites: {len(self.satellites)}\n"
            f"Quantity of vehicles: {len(self.vehicles)}\n"
            f"Quantity of scenarios: {len(self.scenarios)}\n"
            f"-----------------"
        )

    def get_info(self) -> Dict:
        """Get the information of the instance."""
        return {
            "id_instance": self.id_instance,
            "capacity_satellites": self.capacity_satellites,
            "is_continuous_x": self.is_continuous_x,
            "alpha": self.alpha,
            "beta": self.beta,
            "type_of_flexibility": self.type_of_flexibility,
            "periods": self.periods,
            "N": self.N,
            "is_evaluation": self.is_evaluation,
            "quantity_satellites": len(self.satellites),
            "quantity_vehicles": len(self.vehicles),
            "quantity_scenarios": len(self.scenarios),
        }

    def __update_satellites(self) -> None:
        """Update the satellites."""
        for satellite in self.satellites.values():
            satellite.capacity = self.capacity_satellites

    def __read_satellites(self) -> Dict[str, Satellite]:
        """Reads the satellites from the file."""
        try:
            satellites = Data.load_satellites()
        except FileNotFoundError as error:
            logger.error(f"[read satellites] File not found: {error}")
            raise error
        return satellites

    def __read_vehicles(self) -> Dict[str, Vehicle]:
        """Reads the vehicles from the file."""
        try:
            vehicles = Data.load_vehicles()
        except FileNotFoundError as error:
            logger.error(f"[read vehicles] File not found: {error}")
            raise error
        return vehicles

    def __read_pixels(self, id_scenario: int) -> Dict[str, Pixel]:
        """Reads the pixels from the file."""
        try:
            pixels = Data.load_scenario(id_scenario=id_scenario)
        except FileNotFoundError as error:
            logger.error(f"[read pixels] File not found: {error}")
            raise error
        return pixels

    def __calculate_fleet_size_required(self, pixels: Pixel) -> Dict[str, Dict]:
        """Calculates the fleet size required for the instance."""
        try:
            self.matrix_satellite_pixels = Data.load_matrix_from_satellite()
            self.matrix_dc_pixels = Data.load_matrix_from_dc()

            # Compute fleet size required from satellite to pixel
            if self.type_of_cost_serving == 1:
                fleet_size_from_satellites = (
                    self.computer_fleet_size.calculate_avg_fleet_size_from_satellites(
                        pixels=pixels,
                        distances_line_haul=self.matrix_satellite_pixels["distance"],
                        satellites=self.satellites,
                    )
                )
                fleet_size_from_dc = (
                    self.computer_fleet_size.calculate_avg_fleet_size_from_dc(
                        pixels=pixels,
                        distances_line_haul=self.matrix_dc_pixels["distance"],
                    )
                )
            else:
                fleet_size_from_satellites = (
                    self.computer_fleet_size.get_average_fleet_size(pixels, "satellite")
                )
                fleet_size_from_dc = self.computer_fleet_size.get_average_fleet_size(
                    pixels, "dc"
                )
        except FileNotFoundError as error:
            logger.error(f"[calculate fleet size required] File not found: {error}")
            raise error
        fleet_size_required = {
            "satellite": fleet_size_from_satellites,
            "dc": fleet_size_from_dc,
        }
        return fleet_size_required

    def __get_scenarios_sample(self) -> List[int]:
        """Get the scenarios for sample."""
        id_scenarios_sample = []
        if self.is_evaluation:
            path_json = PATH_SAMPLING_SCENARIO + "evaluation.json"
            if os.path.exists(path_json):
                with open(path_json, "r") as file:
                    data = json.load(file)
                    id_scenarios_sample = data["id_scenarios_sample"]
        else:
            id_scenarios_sample = range(1, self.N + 1) # random.sample(range(500), self.N)

        return id_scenarios_sample

    def __calculate_costs(
        self, pixels: Pixel, fleet_size_required: Dict[str, Dict]
    ) -> Dict[str, Dict]:
        """Calculates the costs of the instance."""
        try:
            if self.type_of_cost_serving == 1:
                costs = get_cost_from_continuous_approximation(
                    pixels=pixels,
                    satellites=self.satellites,
                    vehicles=self.vehicles,
                    periods=self.periods,
                    fleet_size_required=fleet_size_required,
                    distance_line_haul={
                        "satellite": self.matrix_satellite_pixels["distance"],
                        "dc": self.matrix_dc_pixels["distance"],
                    },
                )
            else:
                costs = {
                    "dc": self.computer_fleet_size.get_cost_serve_pixel(pixels, "dc"),
                    "satellite": self.computer_fleet_size.get_cost_serve_pixel(
                        pixels, "satellite"
                    ),
                }
        except Exception as error:
            logger.error(f"[calculate costs] File not found: {error}")
            raise error

        return costs

    def __compute_scenarios(self) -> Dict[str, Scenario]:
        """Computes the scenarios."""
        if not self.is_evaluation:
            id_scenarios_sample = self.__get_scenarios_sample()
        else:
            id_scenarios_sample = [self.scenario_evaluation_id]

        logger.info(f"[INSTANCE] Scenarios sample: {id_scenarios_sample}")
        scenarios = {}
        for id_scenario in id_scenarios_sample:
            logger.info(f"  [INSTANCE] Reading scenario: {id_scenario}")
            pixels = self.__read_pixels(id_scenario)
            fleet_size_required = self.__calculate_fleet_size_required(pixels)
            costs_from_ca = self.__calculate_costs(pixels, fleet_size_required)
            scenario = Scenario(
                id_scenario=id_scenario,
                pixels=pixels,
                costs=costs_from_ca,
                fleet_size_required=fleet_size_required,
                periods=self.periods,
            )
            scenarios[str(id_scenario)] = scenario
        return scenarios

    def get_scenario_expected(self) -> Scenario:
        """Get the expected scenario."""
        id_scenario = "expected"
        pixels = self.__read_pixels(id_scenario)
        fleet_size_required = self.__calculate_fleet_size_required(pixels)
        costs_from_ca = self.__calculate_costs(pixels, fleet_size_required)
        scenario = Scenario(
            id_scenario=id_scenario,
            pixels=pixels,
            costs=costs_from_ca,
            fleet_size_required=fleet_size_required,
            periods=self.periods,
        )
        return scenario
