"""Module for classes used in the application model."""
from typing import Dict, List


class Node:
    """Class for objects that have a location in the map."""

    # pylint: disable=too-few-public-methods

    def __init__(
        self,
        lon: float,
        lat: float,
    ):
        self.lon = lon
        self.lat = lat


class Pixel(Node):
    """Class for pixels in the map."""

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        id_pixel: str,
        lon: float,
        lat: float,
        area_surface: float,
        speed_intra_stop: Dict[str, float],
        k: float = 0.57,
    ):  # pylint: disable=too-many-arguments
        Node.__init__(
            self,
            lon,
            lat,
        )
        self.id_pixel = id_pixel
        self.area_surface = area_surface
        self.speed_intra_stop = speed_intra_stop
        self.demand_by_period = []
        self.drop_by_period = []
        self.stop_by_period = []
        self.k = k


class Satellite(Node):
    """Class for satellites in the map."""

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        id_satellite: str,
        lon: float,
        lat: float,
        distance_from_dc: float,
        travel_time_from_dc: float,
        travel_time_in_traffic_from_dc: float,
        capacity: Dict[str, float],
        cost_fixed: Dict[str, float],
        cost_operation: Dict[str, List[float]],
        cost_sourcing: float = 0.335
        / 2,  # TODO: Change this to a variable in the future
    ):  # pylint: disable=too-many-arguments
        Node.__init__(
            self,
            lon,
            lat,
        )
        self.id_satellite = id_satellite
        self.distance_from_dc = distance_from_dc
        self.travel_time_from_dc = travel_time_from_dc
        self.travel_time_in_traffic_from_dc = travel_time_in_traffic_from_dc
        self.cost_fixed = cost_fixed
        self.cost_operation = cost_operation
        self.cost_sourcing = cost_sourcing
        self.capacity = capacity


class Vehicle:
    """Class for vehicles"""

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        id_vehicle: str,
        type_vehicle: str,  # Type of Vehicle (e.g. Truck, Van, Bike) # pylint: disable=line-too-long
        capacity: float,  # Capacity of the vehicle (e.g. 1000 items) # pylint: disable=line-too-long
        cost_fixed: float,  # Fix Costs - only applicable for own # pylint: disable=line-too-long
        time_prep: float,  # Time Set-up: Fixed Set-Up Time at the facility for vehicle type v. Time to dispatch (~time prep called) # pylint: disable=line-too-long
        time_loading_per_item: float,  # Time Set-Up: Time to Load the Vehicle (Time/ package) # pylint: disable=line-too-long
        time_set_up: float,  # Intra-Stop Time: Set-up Time per vehicle (h/customer; ie parkting time) # pylint: disable=line-too-long
        time_service: float,  # Intra-Stop Time: Incremental Service Time for delivery option (h/item) # pylint: disable=line-too-long
        speed_linehaul: float,
        speed_interstop: float,
        t_max: float,
        cost_hourly: float,
        cost_km: float,
        cost_item: float,
        k: float = 0.57,
    ):
        self.id_vehicle = str(id_vehicle)
        self.type_vehicle = type_vehicle
        self.capacity = capacity
        self.cost_fixed = cost_fixed
        self.cost_hourly = cost_hourly
        self.cost_km = cost_km
        self.cost_item = cost_item

        self.time_set_up = time_set_up
        self.time_service = time_service
        self.time_prep = time_prep
        self.time_loading_per_item = time_loading_per_item
        self.t_max = t_max

        self.speed_linehaul = speed_linehaul
        self.speed_interstop = speed_interstop
        self.k = k

    # def __init__(
    #     self,
    #     id_vehicle: str,
    #     type_vehicle: str,
    #     capacity: float,
    #     cost_fixed: float,
    #     time_service: float,
    #     time_fixed: float,
    #     time_prep: float,
    #     time_loading_per_item: float,
    #     speed_line_haul: float,
    #     max_time_services: float,
    #     k: float,
    # ):  # pylint: disable=too-many-arguments
    #     self.id_vehicle = id_vehicle
    #     self.type_vehicle = type_vehicle
    #     self.capacity = capacity
    #     self.cost_fixed = cost_fixed
    #     self.time_fixed = time_fixed
    #     self.time_service = time_service
    #     self.time_prep = time_prep
    #     self.time_loading_per_item = time_loading_per_item
    #     self.speed_line_haul = speed_line_haul
    #     self.max_time_services = max_time_services
    #     self.k = k
