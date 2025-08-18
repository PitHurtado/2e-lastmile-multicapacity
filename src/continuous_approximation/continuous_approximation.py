"""Module for continuous approximation of discrete data."""
import math
from typing import Any, Dict

from src.classes import Pixel, Satellite, Vehicle
from src.utils import LOGGER as logger


class ContinuousApproximation:
    """
    Class for continuous approximation from dc and satellite to pixel usign the continuous approximation model. # pylint: disable=line-too-long
    ----------------
    Parameters:
    - satellites: Dict[str, Satellite]
    - vehicles: Dict[str, Vehicle]
    - t: int
        number of periods
    - matrixes: Dict[str, Any]
        distance matrixes for "dc" and "satellite" to pixel
    ----------------
    Attributes:
    - satellites: Dict[str, Satellite]
    - vehicles: Dict[str, Vehicle]
    - periods: int
    - distance_matrixes: Dict[str, Any]
    """

    def __init__(
        self,
        satellites: Dict[str, Satellite],
        vehicles: Dict[str, Vehicle],
        periods: int,
        matrixes: Dict[str, Any],
    ):
        self.satellites: Dict[str, Satellite] = satellites
        self.vehicles: Dict[str, Vehicle] = vehicles
        self.periods: int = periods
        self.distance_matrixes: Dict[str, Any] = matrixes

    def __effective_vehicle_capacity(self, vehicle: Vehicle, pixel: Pixel, t: int) -> float:
        """Calculate the effective vehicle capacity for a pixel and a vehicle at a given period."""  # pylint: disable=line-too-long
        return vehicle.capacity / pixel.drop_by_period[t]

    def __time_intra_stop(self, pixel: Pixel, vehicle: Vehicle, t: int) -> float:
        """Calculate the time intra stop for a pixel and a vehicle at a given period."""
        return (vehicle.k * pixel.k) / (
            vehicle.speed_interstop
            * math.sqrt(pixel.stop_by_period[t])
        )

    def __distance_intra_stop(self, pixel: Pixel, vehicle: Vehicle, t: int) -> float:
        """Calculate the distance intra stop for a pixel and a vehicle at a given period."""  # pylint: disable=line-too-long
        return (vehicle.k * pixel.k) / math.sqrt(
            pixel.stop_by_period[t]
        )

    def __time_linehaul(self, vehicle: Vehicle, distance: float) -> float:
        """Calculate the time line haul for a pixel and a vehicle at a given period."""
        return 2 * (distance * vehicle.k / vehicle.speed_linehaul)

    def __distance_linehaul(self, vehicle: Vehicle, distance: float) -> float:
        """Calculate the distance line haul for a pixel and a vehicle at a given period."""  # pylint: disable=line-too-long
        return 2 * distance * vehicle.k

    def __time_average_tour(self, pixel: Pixel, vehicle: Vehicle, t: int) -> float:
        """Calculate the average tour time for a pixel and a vehicle at a given period."""  # pylint: disable=line-too-long
        return (
            vehicle.time_set_up
            + vehicle.time_service * pixel.drop_by_period[t]
            + self.__time_intra_stop(pixel, vehicle, t)
        )

    def __distance_tour(self, pixel: Pixel, vehicle: Vehicle, t: int, distance: float) -> float:
        """Calculate the distance tour for a pixel and a vehicle at a given period."""
        return self.__num_customers_per_route(
            pixel, vehicle, t, distance
        ) * self.__distance_intra_stop(pixel, vehicle, t)

    def __time_set_up_fully_loaded_tours(self, pixel: Pixel, vehicle: Vehicle, t: int, distance: float) -> float:
        """Calculate the time set up fully loaded tours for a pixel and a vehicle at a given period."""  # pylint: disable=line-too-long
        return vehicle.time_prep + (
            vehicle.time_loading_per_item
            * self.__effective_vehicle_capacity(vehicle, pixel, t)
            * pixel.drop_by_period[t]
            + self.__time_linehaul(vehicle, distance)
        )

    def __num_fully_loaded_tours(self, pixel: Pixel, vehicle: Vehicle, t: int, distance: float) -> float:
        """Calculate the number of fully loaded tours for a pixel and a vehicle at a given period."""  # pylint: disable=line-too-long
        return vehicle.t_max / (
            self.__effective_vehicle_capacity(vehicle, pixel, t)
            * self.__time_average_tour(pixel, vehicle, t)
            + self.__time_set_up_fully_loaded_tours(pixel, vehicle, t, distance)
        )

    def __num_customers_per_route(self, pixel: Pixel, vehicle: Vehicle, t: int, distance: float) -> float:
        """Calculate the number of customers per route for a pixel and a vehicle at a given period."""  # pylint: disable=line-too-long
        return self.__effective_vehicle_capacity(vehicle, pixel, t) * min(
            1, self.__num_fully_loaded_tours(pixel, vehicle, t, distance)
        )

    def __num_tours(self, pixel: Pixel, vehicle: Vehicle, t: int, distance: float) -> float:
        """Calculate the number of tours for a pixel and a vehicle at a given period."""
        return max(self.__num_fully_loaded_tours(pixel, vehicle, t, distance), 1)

    def __average_fleet_size(self, pixel: Pixel, vehicle: Vehicle, t: int, distance: float) -> float:
        """Calculate the average fleet size for a pixel and a vehicle at a given period."""  # pylint: disable=line-too-long
        if (
            pixel.drop_by_period[t] <= 0
            or pixel.stop_by_period[t] <= 0
            or pixel.demand_by_period[t] <= 0
        ):
            logger.warning(
                f"[CONTINUOUS APPROXIMATION] Pixel {pixel.id_pixel} in period {t} has no demand or no stops or no drops"
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

        # (7) average fleet size
        numerador = pixel.stop_by_period[t] * pixel.area_surface
        denominador = self.__num_fully_loaded_tours(
            pixel, vehicle, t, distance
        ) * self.__effective_vehicle_capacity(vehicle, pixel, t)
        v = (numerador / denominador) if denominador > 0 else 0.0

        return {
            "fleet_size": v,
            "avg_tour_time_full_loaded": self.__time_average_tour(pixel, vehicle, t),
            "num_fully_loaded_tours": self.__num_fully_loaded_tours(
                pixel, vehicle, t, distance
            ),
            "effective_capacity": self.__effective_vehicle_capacity(vehicle, pixel, t),
            "demand_served": pixel.demand_by_period[t],
            "avg_drop": pixel.drop_by_period[t],
            "avg_stop": pixel.stop_by_period[t],
            "avg_time_tour": self.__time_average_tour(pixel, vehicle, t),
            "avg_time_line_haul": self.__time_linehaul(vehicle, distance),
        }

    def __cost_serve_pixel(
        self, pixel: Pixel, vehicle: Vehicle, t: int, distance: float
    ) -> float:
        """Calculate the cost to serve a pixel with a vehicle in a period of time."""
        if (
            pixel.drop_by_period[t] <= 0
            or pixel.stop_by_period[t] <= 0
            or pixel.demand_by_period[t] <= 0
        ):
            logger.warning(
                f"[CONTINUOUS APPROXIMATION] Pixel {pixel.id_pixel} "
                f"in period {t} has no demand or no stops or no drops"
            )
            return {
                "total": 0,
                "cost_prep": 0,
                "cost_linehaul_time": 0,
                "cost_linehaul_distance": 0,
                "cost_segment_time": 0,
                "cost_segment_distance": 0,
                "avg_fleet_size": 0,
                "num_tours": 0,
            }

        # (1) cost to set up
        cost_prep = vehicle.cost_hourly * (
            vehicle.time_prep
            + vehicle.time_loading_per_item
            * pixel.drop_by_period[t]
            * self.__num_customers_per_route(pixel, vehicle, t, distance)
        )

        # (2) cost line haul
        cost_linehaul_time = vehicle.cost_hourly * self.__time_linehaul(
            vehicle, distance
        )
        cost_linehaul_distance = vehicle.cost_km * self.__distance_linehaul(
            vehicle, distance
        )

        # (3) cost intra stop
        cost_segment_time = (
            vehicle.cost_hourly
            * self.__num_customers_per_route(pixel, vehicle, t, distance)
            * self.__time_average_tour(pixel, vehicle, t)
        )

        cost_segment_distance = vehicle.cost_km * self.__distance_tour(
            pixel, vehicle, t, distance
        )

        # (4) cost average fleet size
        avg_fleet_size = self.__average_fleet_size(pixel, vehicle, t, distance)[
            "fleet_size"
        ]
        num_tours = self.__num_tours(pixel, vehicle, t, distance)
        cost = (
            avg_fleet_size
            * num_tours
            * (
                cost_prep  # Component 1: Set-Up Costs
                + cost_linehaul_time
                + cost_linehaul_distance  # Component 2: Line Haul Time + Distance Costs
                + cost_segment_time
                + cost_segment_distance  # Component 3: Intra-Route Costs
                # + vehicle.cost_item
                # * pixel.drop_by_period[t]
                # * self.__num_customers_per_route(
                #     pixel, vehicle, t, distance
                # )  # Component 4: Parcel-Based Costs
            )
        )
        return {
            "total": cost,
            "cost_prep": cost_prep,
            "cost_linehaul_time": cost_linehaul_time,
            "cost_linehaul_distance": cost_linehaul_distance,
            "cost_segment_time": cost_segment_time,
            "cost_segment_distance": cost_segment_distance,
            "avg_fleet_size": avg_fleet_size,
            "num_tours": num_tours,
        }

    def get_average_fleet_size(
        self, pixels: Dict[str, Pixel], echelon: str
    ) -> Dict[Any, Dict[str, float]]:
        """Calculate the average fleet size for a pixel in a period of time."""
        fleet_size = {}
        if echelon == "dc":
            for t in range(self.periods):
                for k, pixel in pixels.items():
                    fleet_size[(k, t)] = self.__average_fleet_size(
                        pixel,
                        self.vehicles["large"],
                        t,
                        self.distance_matrixes[echelon][k],
                    )
        else:
            for t in range(self.periods):
                for s in self.satellites.keys():
                    for k, pixel in pixels.items():
                        fleet_size[(s, k, t)] = self.__average_fleet_size(
                            pixel,
                            self.vehicles["small"],
                            t,
                            self.distance_matrixes[echelon][(s, k)],
                        )

        return fleet_size

    def get_cost_serve_pixel(
        self, pixels: Dict[str, Pixel], echelon: str
    ) -> Dict[Any, Dict[str, float]]:
        """Calculate the cost to serve a pixel in a period of time."""
        cost = {}
        if echelon == "dc":
            for t in range(self.periods):
                for k, pixel in pixels.items():
                    cost[(k, t)] = self.__cost_serve_pixel(
                        pixel,
                        self.vehicles["large"],
                        t,
                        self.distance_matrixes[echelon][k],
                    )
        else:
            for t in range(self.periods):
                for s in self.satellites.keys():
                    for k, pixel in pixels.items():
                        cost[(s, k, t)] = self.__cost_serve_pixel(
                            pixel,
                            self.vehicles["small"],
                            t,
                            self.distance_matrixes[echelon][(s, k)],
                        )
        return cost
