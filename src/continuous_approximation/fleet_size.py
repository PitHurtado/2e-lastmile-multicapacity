"""Module to define the configuration of the CA"""
import math
import sys
from typing import Any, Dict, Tuple

from src.classes import Pixel, Satellite, Vehicle
from src.constants import FEE_COST_FROM_DC, FEE_COST_FROM_SATELLITE
from src.utils import LOGGER as logger


class ContinuousApproximationConfig:
    """Class to define the configuration of the CA"""

    def __init__(
        self,
        periods: int,
        small_vehicle: Vehicle,
        large_vehicle: Vehicle,
    ) -> None:
        self.periods: int = periods
        self.small_vehicle: Vehicle = small_vehicle
        self.large_vehicle: Vehicle = large_vehicle

    def __avg_fleet_size(
        self, pixel: Pixel, vehicle: Vehicle, t: int, distance: float
    ) -> Dict[str, float]:
        """Calculate the average fleet size for a pixel in a period of time"""
        if (
            pixel.drop_by_period[t] <= 0
            or pixel.stop_by_period[t] <= 0
            or pixel.demand_by_period[t] <= 0
        ):
            logger.warning(
                f"[CA] Pixel {pixel.id_pixel} in period {t} "
                f"has no demand or no stops or no drops"
            )
            return {
                "fleet_size": 0,
                "avg_tour_time": 0,
                "fully_loaded_tours": 0,
                "effective_capacity": 0,
                "demand_served": pixel.demand_by_period[t],
                "avg_drop": pixel.drop_by_period[t],
                "avg_stop": pixel.stop_by_period[t],
                "avg_time": 0,
                "avg_time_dispatch": 0,
                "avg_time_line_haul": 0,
            }

        # effective vehicle capacity
        effective_vehicle_capacity = vehicle.capacity / pixel.drop_by_period[t]

        # time services
        time_services = (
            vehicle.time_set_up + vehicle.time_service * pixel.drop_by_period[t]
        )

        # time intra stop
        time_intra_stop = (vehicle.k * pixel.k) / (
            pixel.speed_intra_stop[vehicle.type_vehicle]
            * math.sqrt(pixel.drop_by_period[t] / pixel.area_surface)
        )

        # average tour time
        avg_tour_time = effective_vehicle_capacity * (time_services + time_intra_stop)

        # time preparing
        time_preparing_dispatch = (
            vehicle.time_prep
            + effective_vehicle_capacity
            * pixel.drop_by_period[t]
            * vehicle.time_loading_per_item
        )

        # time line_haul
        time_line_haul = 2 * (distance * vehicle.k / vehicle.speed_linehaul)

        # number of fully loaded tours
        beta = vehicle.t_max / (
            avg_tour_time + time_preparing_dispatch + time_line_haul
        )
        avg_time = avg_tour_time + time_preparing_dispatch + time_line_haul

        # average fleet size
        numerador = pixel.stop_by_period[t]
        denominador = beta * effective_vehicle_capacity
        v = (numerador / denominador) if denominador > 0 else 0.0

        return {
            "fleet_size": v,
            "avg_tour_time": avg_tour_time,
            "fully_loaded_tours": beta,
            "effective_capacity": effective_vehicle_capacity,
            "demand_served": pixel.demand_by_period[t],
            "avg_drop": pixel.drop_by_period[t],
            "avg_stop": pixel.stop_by_period[t],
            "avg_time": avg_time,
            "avg_time_dispatch": time_preparing_dispatch,
            "avg_time_line_haul": time_line_haul,
        }

    def calculate_avg_fleet_size_from_satellites(
        self,
        pixels: Dict[str, Pixel],
        distances_line_haul: Dict[Any, float],
        satellites: Dict[str, Satellite],
    ) -> Dict[Any, float]:
        """Calculate the average fleet size for a pixel in a period of time"""
        # logger.info("[CA] Estimation of fleet size running for satellites")
        fleet_size = dict(
            [
                (
                    (s, k, t),
                    self.__avg_fleet_size(
                        pixel, self.small_vehicle, t, distances_line_haul[(s, k)]
                    ),
                )
                for t in range(self.periods)
                for s in satellites.keys()
                for k, pixel in pixels.items()
            ]
        )
        return fleet_size

    def calculate_avg_fleet_size_from_dc(
        self,
        pixels: Dict[str, Pixel],
        distances_line_haul: Dict[str, float],
    ) -> Dict[Any, float]:
        """Calculate the average fleet size for a pixel in a period of time"""
        # logger.info("[CA] Estimation of fleet size running for DC")
        fleet_size = dict(
            [
                (
                    (k, t),
                    self.__avg_fleet_size(
                        pixel, self.large_vehicle, t, distances_line_haul[k]
                    ),
                )
                for t in range(self.periods)
                for k, pixel in pixels.items()
            ]
        )
        return fleet_size


def cost_satellite_to_pixel(
    satellites: Dict[str, Satellite],
    pixels: Dict[str, Pixel],
    vehicle: Vehicle,
    periods: int,
    fleet_size_required: Dict[Tuple[str, str], float],
    distance_line_haul: Dict[Tuple[str, str], float],
) -> Dict[Tuple[str, str, int], float]:  # pylint: disable=too-many-arguments
    """Calculate the cost from satellite to pixel by period"""
    # cost of shipping from satellite to pixel
    distance_average_from_satellites = {}
    for s in satellites.keys():
        min_distances, max_distances = sys.maxsize, -sys.maxsize - 1
        for k in pixels.keys():
            min_distances = (
                min_distances
                if distance_line_haul[(s, k)] > min_distances
                else distance_line_haul[(s, k)]
            )
            max_distances = (
                max_distances
                if distance_line_haul[(s, k)] < max_distances
                else distance_line_haul[(s, k)]
            )
        distance_average_from_satellites[s] = {
            "min": min_distances,
            "max": max_distances,
            "interval": max_distances - min_distances,
            "cost": 0.421 - 0.335,
        }  # TODO: validate this cost
    cost_shipping_from_satellites = dict(
        [
            (
                (s, k),
                distance_average_from_satellites[(s)]["cost"]
                / distance_average_from_satellites[(s)]["interval"]
                * (
                    distance_line_haul[(s, k)]
                    - distance_average_from_satellites[(s)]["min"]
                )
                + FEE_COST_FROM_SATELLITE,
            )
            for s in satellites.keys()
            for k in pixels.keys()
        ]
    )
    # compute the total cost
    costs = {}
    for t in range(periods):
        for k, pixel in pixels.items():
            for s, satellite in satellites.items():
                cost_first_level = satellite.cost_sourcing * pixel.demand_by_period[t]
                cost_shipping = (
                    cost_shipping_from_satellites[(s, k)] * pixel.demand_by_period[t]
                )
                cost_vehicles = (
                    vehicle.cost_fixed * fleet_size_required[(s, k, t)]["fleet_size"]
                )

                total_cost = (
                    round(cost_first_level, 0)
                    + round(cost_shipping, 0)
                    + round(cost_vehicles, 1)
                )
                costs[(s, k, t)] = {
                    "total": total_cost,
                    "first_level": round(cost_first_level, 0),
                    "shipping": round(cost_shipping, 0),
                    "vehicles": round(cost_vehicles, 1),
                }
    return costs


def cost_dc_to_pixel(
    pixels: Dict[str, Pixel],
    periods: int,
    vehicle: Vehicle,
    fleet_size_required: Dict[str, dict],
    distance_line_haul: Dict[str, float],
) -> Dict[Any, float]:
    """Calculate the cost from DC to pixel by period"""
    # cost of shipping from DC to pixel
    distance_average_from_dc = {}
    min_distances, max_distances = sys.maxsize, -sys.maxsize - 1
    for k in pixels.keys():
        min_distances = (
            min_distances
            if distance_line_haul[(k)] > min_distances
            else distance_line_haul[(k)]
        )
        max_distances = (
            max_distances
            if distance_line_haul[(k)] < max_distances
            else distance_line_haul[(k)]
        )
    distance_average_from_dc = {
        "min": min_distances,
        "max": max_distances,
        "interval": max_distances - min_distances,
        "cost": 0.389 - 0.264,
    }  # TODO: validate this cost
    cost_shipping_from_dc = dict(
        [
            (
                k,
                distance_average_from_dc["cost"]
                / distance_average_from_dc["interval"]
                * (distance_line_haul[(k)] - distance_average_from_dc["min"])
                + FEE_COST_FROM_DC,
            )
            for k in pixels.keys()
        ]
    )
    # compute the total cost
    costs = {}
    for t in range(periods):
        for k, pixel in pixels.items():
            cost_shipping = cost_shipping_from_dc[k] * pixel.demand_by_period[t]
            cost_vehicles = (
                vehicle.cost_fixed * fleet_size_required[(k, t)]["fleet_size"]
            )

            total_cost = round(cost_shipping, 0) + round(cost_vehicles, 1)
            costs[(k, t)] = {
                "total": total_cost,
                "shipping": round(cost_shipping, 0),
                "vehicles": round(cost_vehicles, 1),
            }
    return costs


def get_cost_from_continuous_approximation(
    pixels: Dict[str, Pixel],
    satellites: Dict[str, Satellite],
    vehicles: Dict[str, Vehicle],
    fleet_size_required: Dict[Any, Dict],
    distance_line_haul: Dict[str, Dict],
    periods: int,
) -> Dict[str, Dict]:  # pylint: disable=too-many-arguments
    """
    Calculate the cost of serving a pixel by period
    ----
    returns:
    - Dict[str, Dict]
        the cost of serving a pixel by period, where the key is the echelon dc or satellite
    """
    cost_satellite_to_pixel_period = cost_satellite_to_pixel(
        satellites=satellites,
        pixels=pixels,
        vehicle=vehicles["small"],
        periods=periods,
        fleet_size_required=fleet_size_required["satellite"],
        distance_line_haul=distance_line_haul["satellite"],
    )
    cost_dc_to_pixel_period = cost_dc_to_pixel(
        pixels=pixels,
        periods=periods,
        vehicle=vehicles["large"],
        fleet_size_required=fleet_size_required["dc"],
        distance_line_haul=distance_line_haul["dc"],
    )
    return {
        "dc": cost_dc_to_pixel_period,
        "satellite": cost_satellite_to_pixel_period,
    }
