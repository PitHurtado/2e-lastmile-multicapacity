"""Module to get configuration from json file"""

# Configurations of the vehicles ME
SMALL_CONFIG = {
    "id_vehicle": "van",
    "type_vehicle": "small",
    "capacity": 115,
    "cost_fixed": 67,
    "time_prep": 5 / 60,
    "time_loading_per_item": 0.067 / 60,
    "time_set_up": 2 / 60,
    "time_service": 1 / 60,
    "speed_linehaul": 50,
    "speed_interstop": 35,
    "t_max": 12,
    "cost_hourly": 53.9,
    "cost_km": 0.37,
    "cost_item": 0.5,
}
LARGE_CONFIG = {
    "id_vehicle": "truck",
    "type_vehicle": "large",
    "capacity": 460,
    "cost_fixed": 268,
    "time_prep": 10 / 60,
    "time_loading_per_item": 0.05 / 60,
    "time_set_up": 2 / 60,
    "time_service": 2 / 60,
    "speed_linehaul": 35,
    "speed_interstop": 20,
    "t_max": 12,
    "cost_hourly": 50,
    "cost_km": 8.7,
    "cost_item": 0.5,
    "k": 1,
}


# Configurations of the vehicles PROVIDED
# SMALL_CONFIG = {
#     "id_vehicle": "van",
#     "type_vehicle": "small",
#     "capacity": 200,
#     "cost_fixed": 67,
#     "time_prep": 5 / 60,
#     "time_loading_per_item":  0.05 / 60,
#     "time_set_up": 0.017,
#     "time_service": 0.02,
#     "speed_linehaul": 40,
#     "speed_interstop": 20,
#     "t_max": 12,
#     "cost_hourly": 50,
#     "cost_km": 4.01,
#     "cost_item": 0.5,
#     "k": 0.57,
# }
