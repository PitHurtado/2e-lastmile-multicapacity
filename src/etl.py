"""Module to get data from csv file"""
import json
import os
import sys
from typing import Dict

import pandas as pd

from src.classes import Pixel, Satellite, Vehicle
from src.config import LARGE_CONFIG, SMALL_CONFIG
from src.constants import (
    PATH_DATA_DISTANCES_FROM_DC,
    PATH_DATA_DISTANCES_FROM_SATELLITES,
    PATH_DATA_PIXEL,
    PATH_DATA_SATELLITE,
    PATH_ROOT_SCENARIO,
)
from src.utils import LOGGER as logger


class Data:
    """Class to get data from csv file"""

    @staticmethod
    def load_satellites(
        show_data: bool = False,
    ) -> Dict[str, Satellite]:
        """Load data from csv file and create a dictionary of satellites"""
        satellites = {}
        if not os.path.isfile(PATH_DATA_SATELLITE):
            logger.error(f"[ETL] File {PATH_DATA_SATELLITE} not found")
            sys.exit(1)
        df = pd.read_excel(PATH_DATA_SATELLITE)
        if df.empty:
            logger.error(f"[ETL] File {PATH_DATA_SATELLITE} is empty")
            sys.exit(1)
        for _, row in df.iterrows():
            id_satellite = str(row["id_satellite"])

            cost_fixed = Data.__round_dict_values(json.loads(row["cost_fixed"]))
            cost_operation = {
                key: Data.__round_list_values(value_list)
                for key, value_list in json.loads(row["cost_operation"]).items()
            }

            new_satellite = Satellite(
                id_satellite=id_satellite,
                lon=row["lon"],
                lat=row["lat"],
                distance_from_dc=row["distance"],
                travel_time_from_dc=row["travel_time_from_dc"],
                travel_time_in_traffic_from_dc=row["travel_time_in_traffic_from_dc"],
                capacity=json.loads(row["capacity"]),
                cost_fixed=cost_fixed,
                cost_operation=cost_operation,
                cost_sourcing=row["cost_sourcing"],
            )
            satellites[id_satellite] = new_satellite
        if show_data:
            for s in satellites.values():
                logger.info(
                    "-" * 50 + "\n" + json.dumps(s.__dict__, indent=2, default=str)
                )
        # logger.info(f"[ETL] Count of SATELLITES loaded: {len(satellites)}")
        return satellites

    @staticmethod
    def load_pixels(show_data: bool = False) -> Dict[str, Pixel]:
        """Load data from csv file and create a dictionary of pixels"""
        pixels = {}
        if not os.path.isfile(PATH_DATA_PIXEL):
            logger.error(f"[ETL] File {PATH_DATA_PIXEL} not found")
            sys.exit(1)
        df = pd.read_excel(PATH_DATA_PIXEL)
        if df.empty:
            logger.error(f"[ETL] File {PATH_DATA_PIXEL} is empty")
            sys.exit(1)
        for _, row in df.iterrows():
            id_pixel = str(row["id_pixel"])
            # create a new pixel
            new_pixel = Pixel(
                id_pixel=id_pixel,
                lon=row["lon"],
                lat=row["lat"],
                area_surface=row["area_surface"],
                speed_intra_stop=json.loads(row["speed_intra_stop"]),
            )
            pixels[id_pixel] = new_pixel
        if show_data:
            for p in pixels.values():
                logger.info(
                    "-" * 50 + "\n" + json.dumps(p.__dict__, indent=2, default=str)
                )
        # logger.info(f"[ETL] Count of PIXELS loaded: {len(pixels)}")
        return pixels

    @staticmethod
    def load_matrix_from_satellite() -> Dict[str, Dict]:
        """Load data from csv file and create a dictionary of distances and durations"""
        if not os.path.isfile(PATH_DATA_DISTANCES_FROM_SATELLITES):
            logger.error(f"[ETL] File {PATH_DATA_DISTANCES_FROM_SATELLITES} not found")
            sys.exit(1)
        df = pd.read_excel(PATH_DATA_DISTANCES_FROM_SATELLITES)
        if df.empty:
            logger.error(f"[ETL] File {PATH_DATA_DISTANCES_FROM_SATELLITES} is empty")
            sys.exit(1)
        distance = {
            (row["id_satellite"], row["id_pixel"]): row["distance"]
            for _, row in df.iterrows()
        }
        travel_time = {
            (row["id_satellite"], row["id_pixel"]): row["travel_time"]
            for _, row in df.iterrows()
        }
        travel_time_in_traffic = {
            (row["id_satellite"], row["id_pixel"]): row["travel_time_in_traffic"]
            for _, row in df.iterrows()
        }
        matrix = {
            "travel_time": travel_time,
            "distance": distance,
            "travel_time_in_traffic": travel_time_in_traffic,
        }
        return matrix

    @staticmethod
    def load_matrix_from_dc() -> Dict[str, Dict]:
        """Load data from csv file and create a dictionary of distances and durations"""
        if not os.path.isfile(PATH_DATA_DISTANCES_FROM_DC):
            logger.error(f"[ETL] File {PATH_DATA_DISTANCES_FROM_DC} not found")
            sys.exit(1)
        df = pd.read_excel(PATH_DATA_DISTANCES_FROM_DC)
        if df.empty:
            logger.error(f"[ETL] File {PATH_DATA_DISTANCES_FROM_DC} is empty")
            sys.exit(1)
        distance = {(row["id_pixel"]): row["distance"] for _, row in df.iterrows()}
        travel_time = {
            (row["id_pixel"]): row["travel_time"] for _, row in df.iterrows()
        }
        travel_time_in_traffic = {
            (row["id_pixel"]): row["travel_time_in_traffic"] for _, row in df.iterrows()
        }
        matrix = {
            "travel_time": travel_time,
            "distance": distance,
            "travel_time_in_traffic": travel_time_in_traffic,
        }
        return matrix

    @staticmethod
    def load_scenario(id_scenario: str, show_data: bool = False) -> Dict[str, Pixel]:
        """Load data from scenario from csv file and create a dictionary of pixels"""
        base_pixels = Data.load_pixels(show_data)
        scenario_path = PATH_ROOT_SCENARIO + f"scenario_{id_scenario}.xlsx"
        if not os.path.isfile(scenario_path):
            logger.error(f"[ETL] File {scenario_path} not found")
            sys.exit(1)
        df = pd.read_excel(scenario_path)
        if df.empty:
            logger.error(f"[ETL] File {scenario_path} is empty")
            sys.exit(1)

        pixels = {}
        for _, row in df.iterrows():
            id_pixel = str(row["id_pixel"])
            demand_by_period = json.loads(row["demand_by_period"])
            drop_by_period = json.loads(row["drop_by_period"])
            stop_by_period = json.loads(row["stop_by_period"])
            pixel = base_pixels.get(id_pixel, None)
            if not pixel is None:
                # update the demand by period
                pixel.demand_by_period = demand_by_period
                pixel.stop_by_period = stop_by_period
                pixel.drop_by_period = drop_by_period
                pixels[id_pixel] = pixel
        if show_data:
            for p in pixels.values():
                logger.info(
                    "-" * 50 + "\n" + json.dumps(p.__dict__, indent=2, default=str)
                )  # pylint: disable=no-member
        # logger.info(
        #     f"[ETL] Count of PIXELS loaded from instances: {len(pixels)} - scenario {id_scenario}" # pylint: disable=line-too-long
        # )
        return pixels

    @staticmethod
    def load_vehicles() -> Dict[str, Vehicle]:
        """Load data from csv file and create a dictionary of distances and durations"""
        vehicle_large = Vehicle(
            **LARGE_CONFIG,
        )
        vehicle_small = Vehicle(
            **SMALL_CONFIG,
        )
        # logger.info(
        #     f"[ETL] Quantity of vehicles loaded: {len([vehicle_small, vehicle_large])}" # pylint: disable=line-too-long
        # )
        return {"small": vehicle_small, "large": vehicle_large}

    @staticmethod
    def __round_dict_values(dict_to_round: Dict) -> Dict:
        """Round the values of a dictionary"""
        return {k: round(v, 0) for k, v in dict_to_round.items()}

    @staticmethod
    def __round_list_values(list_to_round: list) -> list:
        """Round the values of a list"""
        return [round(v, 0) for v in list_to_round]
