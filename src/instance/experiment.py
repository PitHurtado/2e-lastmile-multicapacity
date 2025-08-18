"""Module of class Experiment."""
import ast
import itertools
import json
import os
from typing import Any, Dict, List

from src.constants import PATH_SAMPLING_SCENARIO


class Experiment:
    """Class to generate experiments."""

    def __init__(
            self,
            N: int,
            M: int,
            folder_path: str,
            evaluation: bool,
            expected: bool,
    ):
        self.N = N
        self.M = M
        self.folder_path = folder_path
        self.combinations = []
        self.evaluation = evaluation
        self.expected = expected

    def get_combinations(self, runs:dict) -> itertools.product:
        if self.evaluation:
            return self.get_combinations_evaluation()
        else:
            if self.expected:
                return self.get_combinations_expected()
            else:
                return self.get_combinations_branch_and_cut(runs)

    def get_combinations_branch_and_cut(self, runs:dict) -> itertools.product:
        """Return a list of combinations."""
        run_test, run_split, run_valid_inequalities, run_reformulated, run_full_experiments = (
            runs.get(run, False) for run in list(runs.keys())
        )

        # TODO:
        #  - Define subset of experiments for run_valid_inequalities and run_reformulated
        #  - Define full factorial experiment design

        if run_test:
            N = [1]
            M = [1]
            capacity_satellites = [
                {"0": 0, "4": 4, "8": 8, "12": 12},
            ]
            is_continuous_x = [False]
            type_of_flexibility = [2]
            type_of_cost_serving = [2]
            alpha = [1.0]
            split = [3]
            warm_start_MP = [True]
            warm_start_SP = [False]
            reformulated = [True]
            valid_inequalities = [True]

        if run_split:
            N = [1] # [5 * i for i in range(1, self.N // 5 + 1)]
            M = [1]
            capacity_satellites = [
                {"0": 0, "4": 4, "8": 8, "12": 12},
            ]
            is_continuous_x = [False]
            type_of_flexibility = [3]
            type_of_cost_serving = [2]
            alpha = [1.0]
            split = [1, 3, 6, 12]
            warm_start_MP = [True]
            warm_start_SP = [False]
            reformulated = [True]
            valid_inequalities = [True]

        if run_full_experiments:
            N = [5 * i for i in range(1, self.N // 5 + 1)]
            M = range(1, self.M + 1)
            capacity_satellites = [ # Set keys = values
                {"0": 0, "4": 4, "8": 8, "12": 12},
                {"0": 0, "2": 2, "4": 4, "6": 6, "8": 8, "10": 10, "12": 12},
            ]
            is_continuous_x = [True, False]
            type_of_flexibility = [1, 2, 3]
            type_of_cost_serving = [2]
            alpha = [0.5, 1.0, 1.5]
            split = [6]
            warm_start_MP = [True]
            warm_start_SP = [False]
            reformulated = [True]
            valid_inequalities = [True]

        parameters_combinations = itertools.product(
            N, M, capacity_satellites, is_continuous_x, type_of_flexibility, type_of_cost_serving, alpha, split,
            warm_start_MP, warm_start_SP, reformulated, valid_inequalities
        )

        for idx, comb in enumerate(parameters_combinations):
            unique_id_parts = []
            for element in comb:
                if isinstance(element, list):
                    unique_id_parts.append(str(len(element)))
                elif isinstance(element, dict):
                    unique_id_parts.append(str(len(element.keys())))
                else:
                    unique_id_parts.append(str(element))

            unique_id = f"ID_{'_'.join(unique_id_parts)}"
            comb_with_id = (unique_id,) + comb
            self.combinations.append(comb_with_id)

        return self.combinations

    def get_combinations_evaluation(self) -> itertools.product:
        # Read evaluation set:
        scenarios_evaluation = self.get_scenarios_evaluation()

        # Read solutions to be evaluated:
        parameters_combinations = []

        if self.expected:
            folder_path = self.folder_path.replace(self.folder_path.split("/")[-2], 'expected')
        else:
            folder_path = self.folder_path.replace(self.folder_path.split("/")[-2], 'bc')

        # Iterate over all files in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith('.json'):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, 'r') as file:
                    data = json.load(file)

                    # Extract and store the value associated with the "Y" key
                    y_string = data.get('Y', '')
                    Y_solutions = ast.literal_eval(y_string)

                    # Extract and store the values of combination
                    n = data.get('N', None)
                    m = data.get('m', None)
                    capacity_satellites = data.get('capacity_satellites', None)
                    is_continuous_x = data.get('is_continuous_x', False)
                    type_of_flexibility = data.get('type_of_flexibility', False)
                    type_of_cost_serving = data.get('type_of_cost_serving', False)
                    alpha = data.get('alpha', None)
                    warm_start_MP = data.get('warm_start_MP', False)
                    split = data.get('split', None)
                    warm_start_SP = data.get('warm_start_SP', False)
                    valid_inequalities = data.get('valid_inequalities', False)
                    reformulated = data.get('reformulated', False)

                    parameters_combinations.append((
                        n, m, capacity_satellites, is_continuous_x, type_of_flexibility, type_of_cost_serving, alpha,
                        split, warm_start_MP, warm_start_SP, valid_inequalities, reformulated, Y_solutions
                    ))

        product = list(itertools.product(parameters_combinations, scenarios_evaluation))
        parameters_combinations = [tuple(elem for sublist in p for elem in (sublist if isinstance(sublist, tuple) else [sublist])) for p in product]

        for idx, comb in enumerate(parameters_combinations):
            selected_elements = comb[:8]
            unique_id_parts = []
            for element in selected_elements:
                if isinstance(element, list):
                    unique_id_parts.append(str(len(element)))
                elif isinstance(element, dict):
                    unique_id_parts.append(str(len(element.keys())))
                else:
                    unique_id_parts.append(str(element))

            unique_id = f"ID_{'_'.join(unique_id_parts)}"
            comb_with_id = (unique_id,) + comb
            self.combinations.append(comb_with_id)

        return self.combinations

    def get_combinations_expected(self) -> itertools.product:
        # Read solutions to be evaluated from the B&C folder:
        parameters_combinations = []
        folder_path = self.folder_path.replace(self.folder_path.split("/")[-2], "bc")

        # Iterate over all files in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith('.json') and 'ID_20_1_' in filename and '_3_True_False_True_True' in filename:
                file_path = os.path.join(folder_path, filename)
                with open(file_path, 'r') as file:
                    data = json.load(file)

                    # Extract and store the value associated with the "Y" key
                    y_string = data.get('Y', '')
                    Y_solutions = {} # ast.literal_eval(y_string)

                    # Extract and store the values of N and m
                    n = 20 # data.get('N', None)
                    m = 1 # data.get('m', None)
                    capacity_satellites = data.get('capacity_satellites', None)
                    is_continuous_x = data.get('is_continuous_x', False)
                    type_of_flexibility = data.get('type_of_flexibility', False)
                    type_of_cost_serving = data.get('type_of_cost_serving', False)
                    alpha = data.get('alpha', None)
                    split = 3 # data.get('split', None)
                    warm_start_MP = True # data.get('warm_start_MP', False)
                    warm_start_SP = False # data.get('warm_start_SP', False)
                    valid_inequalities = True # data.get('valid_inequalities', False)
                    reformulated = True # data.get('reformulated', False)

                    parameters_combinations.append((
                        n, m, capacity_satellites, is_continuous_x, type_of_flexibility, type_of_cost_serving, alpha, Y_solutions
                    ))

        product = parameters_combinations

        parameters_combinations = [tuple(elem for sublist in p for elem in (sublist if isinstance(sublist, tuple) else [sublist])) for p in product]

        for idx, comb in enumerate(parameters_combinations):
            selected_elements = comb[:7]
            unique_id_parts = []
            for element in selected_elements:
                if isinstance(element, list):
                    unique_id_parts.append(str(len(element)))
                elif isinstance(element, dict):
                    unique_id_parts.append(str(len(element.keys())))
                else:
                    unique_id_parts.append(str(element))

            unique_id = f"ID_{'_'.join(unique_id_parts)}"
            comb_with_id = (unique_id,) + comb
            self.combinations.append(comb_with_id)

        return self.combinations

    def get_scenarios_evaluation(self):
        path_json = PATH_SAMPLING_SCENARIO + "evaluation.json"
        if os.path.exists(path_json):
            with open(path_json, "r") as file:
                data = json.load(file)
                id_scenarios_sample = data["id_scenarios_sample"]

        return id_scenarios_sample