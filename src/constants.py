"""Constants for the application."""

# Constants
FEE_COST_FROM_SATELLITE = 0.00001333
FEE_COST_FROM_DC = 0.264


PATH_ROOT = "./data/"
PATH_ROOT = '/Users/juanpina/Dropbox (MIT)/01 Postdoctoral Research/02 Work in Progress/code_multi_period/data/'

# Paths
PATH_DATA_SATELLITE = PATH_ROOT + "input_satellites.xlsx"
PATH_DATA_PIXEL = PATH_ROOT + "input_pixels.xlsx"
PATH_DATA_DISTANCES_FROM_SATELLITES = (
    PATH_ROOT + "input_matrix_distance_satellite_to_pixels.xlsx"
)
PATH_DATA_DISTANCES_FROM_DC = PATH_ROOT + "input_matrix_distance_dc_to_pixels.xlsx"

# Path root from scenarios
PATH_ROOT_SCENARIO = PATH_ROOT + "scenarios/"

PATH_SAMPLING_SCENARIO = PATH_ROOT_SCENARIO + "sampling/"
PATH_BEST_SOLUTION_SAA = PATH_ROOT_SCENARIO + "best_solution_saa/"
