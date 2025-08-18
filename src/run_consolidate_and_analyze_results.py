"""Consolidating and analyzing results."""
import json
import os
import csv
import pandas as pd
import ast
import itertools

def consolidate_bc(path_files, path_save):
    # List of fields you're interested in
    # all_fields = [
    #     "Y", "actual_run_time", "optimality_gap", "objective_value", "best_bound_value",
    #     "time_best_known_solution", "total_cuts", "upper_bound_updates", "subproblems_solved",
    #     "ID", "m", "N", "capacity_satellites", "is_continuous_x",
    #     "type_of_flexibility", "type_of_cost_serving", "alpha", "periods",
    #     "max_run_time", "warm_start", "split", "warm_start_subproblems",
    #     "valid_inequalities", "reformulated"
    # ]
    interested_fields = [
        "ID", "Y", "objective_value", "best_bound_value", "actual_run_time", "optimality_gap",
        "time_best_known_solution", "total_cuts", "upper_bound_updates", "subproblems_solved",
        "m", "N", "capacity_satellites", "is_continuous_x",
        "type_of_flexibility", "alpha", "periods",
        "max_run_time", "valid_inequalities"
    ]

    # List to hold all consolidated data in order
    consolidated_data = []

    # Iterate over each file in the directory
    for filename in os.listdir(path_files):
        if filename.endswith(".json"):
            # Open and read the JSON file
            with open(os.path.join(path_files, filename), 'r') as f:
                data = json.load(f)

                # Build the row with filename first, then the rest of the fields
                row = {"Filename": filename}
                for field in interested_fields:
                    if field in data:
                        row[field] = data[field]
                    else:
                        row[field] = None
                        print(f"[BC] Alert: Field '{field}' not found in file {filename}.")

                # Append the row to the consolidated list
                consolidated_data.append(row)

    # Convert the consolidated data to a Pandas DataFrame
    df = pd.DataFrame(consolidated_data)

    # Define the output CSV file
    output_csv = os.path.join(path_save, "consolidated_results_bc.csv")

    # Write the DataFrame to a CSV file
    df.to_csv(output_csv, index=False)

    print(f"Consolidation complete. Results saved to {output_csv}")

    # Return the DataFrame for further use
    return df


def consolidate_eval_bc(path):
    # List of fields you're interested in
    # all_fields = [
    #     "Y", "Z", "X", "W",
    #     "Satellite_operating_periods", "Satellite_total_capacity_limit", "Satellite_total_capacity_used",
    #     "Satellite_average_capacity_limit", "Satellite_average_capacity_used",
    #     "DC_average_fleet_used", "run_time", "total_cost"
    # ]
    interested_fields = [
        "Y", "Z", "X", "W",
        "Satellite_operating_periods", "Satellite_total_capacity_limit", "Satellite_total_capacity_used",
        "Satellite_average_capacity_limit", "Satellite_average_capacity_used",
        "DC_average_fleet_used", "run_time", "total_cost"
    ]

    # List to hold all consolidated data in order
    consolidated_data = []

    # Iterate over each file in the directory
    validate = 0
    for filename in os.listdir(path):
        if filename.endswith(".json") and 'solution_branch_and_cut_ID_' in filename:
            validate += 1
            # Open and read the JSON file
            with open(os.path.join(path, filename), 'r') as f:
                data = json.load(f)

                # Build the row with filename first, then the rest of the fields
                row = {"Filename": filename}
                for field in interested_fields:
                    if field in data:
                        row[field] = data[field]
                    else:
                        row[field] = None
                        print(f"[EVALUATION BC] Alert: Field '{field}' not found in file {filename}.")

                # Append the row to the consolidated list
                consolidated_data.append(row)

    # Convert the consolidated data to a Pandas DataFrame
    df = pd.DataFrame(consolidated_data)

    # Define the output CSV file
    if validate != 32000:
        print(f"[EVALUATION BC] Problem!!! validate is {validate}")
    output_csv = os.path.join(path, "consolidated_results_evaluation_bc.csv")

    # Write the DataFrame to a CSV file
    df.to_csv(output_csv, index=False)

    print(f"Consolidation complete. Results saved to {output_csv}")

    # Return the DataFrame for further use
    return df


def consolidate_eval_expected(path):
    # List of fields you're interested in
    # all_fields = [
    #     "Y", "Z", "X", "W",
    #     "Satellite_operating_periods", "Satellite_total_capacity_limit", "Satellite_total_capacity_used",
    #     "Satellite_average_capacity_limit", "Satellite_average_capacity_used",
    #     "DC_average_fleet_used", "run_time", "total_cost"
    # ]
    interested_fields = [
        "Y", "Z", "X", "W",
        "Satellite_operating_periods", "Satellite_total_capacity_limit", "Satellite_total_capacity_used",
        "Satellite_average_capacity_limit", "Satellite_average_capacity_used",
        "DC_average_fleet_used", "run_time", "total_cost"
    ]

    # List to hold all consolidated data in order
    consolidated_data = []

    # Iterate over each file in the directory
    validate = 0
    for filename in os.listdir(path):
        if filename.endswith(".json") and 'solution_expected_ID_' in filename:
            validate += 1
            # Open and read the JSON file
            with open(os.path.join(path, filename), 'r') as f:
                data = json.load(f)

                # Build the row with filename first, then the rest of the fields
                row = {"Filename": filename}
                for field in interested_fields:
                    if field in data:
                        row[field] = data[field]
                    else:
                        row[field] = None
                        print(f"[EVALUATION EXPECTED] Alert: Field '{field}' not found in file {filename}.")

                # Append the row to the consolidated list
                consolidated_data.append(row)

    # Convert the consolidated data to a Pandas DataFrame
    df = pd.DataFrame(consolidated_data)

    # Define the output CSV file
    if validate != 1600:
        print(f"[EVALUATION EXPECTED] Problem!!! validate is {validate}")
    output_csv = os.path.join(path, "consolidated_results_evaluation_expected.csv")

    # Write the DataFrame to a CSV file
    df.to_csv(output_csv, index=False)

    print(f"Consolidation complete. Results saved to {output_csv}")

    # Return the DataFrame for further use
    return df


def consolidate_expected(path):
    # List of fields you're interested in
    # all_fields = [
    #     "Y", "objective", "cost_installation_satellites", "cost_operating_satellites",
    #     "cost_served_from_satellite", "cost_served_from_dc", "scenarios", "Solver information",
    #     "ID", "m", "N", "capacity_satellites", "is_continuous_x",
    #     "type_of_flexibility", "type_of_cost_serving", "alpha", "periods", "max_run_time"
    # ]

    interested_fields = [
        "ID", "objective", "cost_installation_satellites", "cost_operating_satellites",
        "cost_served_from_satellite", "cost_served_from_dc", "scenarios", "Solver information", "m", "N",
        "capacity_satellites", "is_continuous_x", "type_of_flexibility", "alpha", "periods", "max_run_time"
    ]

    # Dictionary mapping each nested field to its subfields
    nested_fields = {
        "Solver information": ["actual_run_time", "optimality_gap", "objective_value", "best_bound_value"],
    }

    consolidated_data = []

    for filename in os.listdir(path):
        if filename.endswith(".json"):
            with open(os.path.join(path, filename), 'r') as f:
                data = json.load(f)

                row = {"Filename": filename}
                for field in interested_fields:
                    if field in data:
                        row[field] = data[field]
                    else:
                        row[field] = None
                        print(f"[EXPECTED] Alert: Field '{field}' not found in file {filename}.")

                # Extract nested fields and add them as separate columns
                for nested_field, subfields in nested_fields.items():
                    nested_data = data.get(nested_field, {})
                    for subfield in subfields:
                        row[f"{subfield}"] = nested_data.get(subfield, None)
                    if nested_field in row:
                        del row[nested_field]

                # Append the row to the consolidated list
                consolidated_data.append(row)

    # Convert the consolidated data to a Pandas DataFrame
    df = pd.DataFrame(consolidated_data)

    # Define the output CSV file
    output_csv = os.path.join(path, "consolidated_results_expected.csv")

    # Write the DataFrame to a CSV file
    df.to_csv(output_csv, index=False)

    print(f"Consolidation complete. Results saved to {output_csv}")

    # Return the DataFrame for further use
    return df


def consolidate_saa(path):
    # all_fields = [
    #     "Y", "objective", "cost_installation_satellites", "cost_operating_satellites",
    #     "cost_served_from_satellite", "cost_served_from_dc", "scenarios", "Solver information",
    #     "ID", "m", "N", "capacity_satellites", "is_continuous_x",
    #     "type_of_flexibility", "type_of_cost_serving", "alpha", "periods", "max_run_time"
    # ]

    interested_fields = [
            "ID", "objective", "cost_installation_satellites", "cost_operating_satellites",
            "cost_served_from_satellite", "cost_served_from_dc", "scenarios", "Solver information", "m", "N",
            "capacity_satellites", "is_continuous_x", "type_of_flexibility", "alpha", "periods", "max_run_time"
    ]

    # Dictionary mapping each nested field to its subfields
    nested_fields = {
        "Solver information": ["actual_run_time", "optimality_gap", "objective_value", "best_bound_value"],
    }

    consolidated_data = []

    for filename in os.listdir(path):
        if filename.endswith(".json"):
            with open(os.path.join(path, filename), 'r') as f:
                data = json.load(f)

                row = {"Filename": filename}
                for field in interested_fields:
                    if field in data:
                        row[field] = data[field]
                    else:
                        row[field] = None
                        print(f"[SAA] Alert: Field '{field}' not found in file {filename}.")

                # Extract nested fields and add them as separate columns
                for nested_field, subfields in nested_fields.items():
                    nested_data = data.get(nested_field, {})
                    for subfield in subfields:
                        row[f"{subfield}"] = nested_data.get(subfield, None)
                    if nested_field in row:
                        del row[nested_field]

                # Append the row to the consolidated list
                consolidated_data.append(row)

    # Convert the consolidated data to a Pandas DataFrame
    df = pd.DataFrame(consolidated_data)

    # Define the output CSV file
    output_csv = os.path.join(path, "consolidated_results_saa.csv")

    # Write the DataFrame to a CSV file
    df.to_csv(output_csv, index=False)

    print(f"Consolidation complete. Results saved to {output_csv}")

    # Return the DataFrame for further use
    return df


def calculate_basecase_results_bc(path, path_export, path_save, all_results_df):
    satellite_condition = True
    x_condition = True
    flexibility_condition = True
    alpha_condition = True
    output_file_name = "basecase"
    if 'capacity_satellites' in all_results_df.columns and not all_results_df['capacity_satellites'].isnull().all():
        all_results_df['capacity_satellites'] = all_results_df['capacity_satellites'].astype(str)
        satellite_condition = all_results_df['capacity_satellites'] == str({'0': 0, '4': 4, '8': 8, '12': 12})
        output_file_name += "_sat_4_8_12"
    if 'is_continuous_x' in all_results_df.columns and not all_results_df['is_continuous_x'].isnull().all():
        x_condition = all_results_df['is_continuous_x'] == False
        output_file_name += "_x_bool"
    if 'type_of_flexibility' in all_results_df.columns and not all_results_df['type_of_flexibility'].isnull().all():
        flexibility_condition = all_results_df['type_of_flexibility'] == 2
        output_file_name += "_flex_2"
    if 'alpha' in all_results_df.columns and not all_results_df['alpha'].isnull().all():
        alpha_condition = all_results_df['alpha'] == 1
        output_file_name += "_alpha_1"

    wip_basecase_df = all_results_df[satellite_condition & x_condition & flexibility_condition & alpha_condition]

    # Group the DataFrame by the unique values in column 'N'
    grouped = wip_basecase_df.groupby('N')

    # Calculate the average of 'optimality_gap' for each group
    avg_opt_gap = grouped['optimality_gap'].mean()

    # Calculate the average of 'actual_run_time' for each group
    avg_run_time = grouped['actual_run_time'].mean()

    # Calculate the number of rows with 'optimality_gap' < 0.01 for each group
    num_optimal_sol = grouped.apply(lambda x: (x['optimality_gap'] < 0.01).sum())

    # Combine these into a new DataFrame
    basecase_df = pd.DataFrame({
        'avg_opt_gap': avg_opt_gap,
        'avg_run_time': avg_run_time,
        'num_optimal_sol': num_optimal_sol
    }).reset_index()

    os.makedirs(path_export, exist_ok=True)

    # Define the output CSV file
    output_csv = os.path.join(path_save, f"{output_file_name}.csv")

    # Write the DataFrame to a CSV file
    basecase_df.to_csv(output_csv, index=False)

    print(f"Basecase table complete. Results saved to {output_csv}")

    print(basecase_df)

    return basecase_df


def calculate_satellites_utilization(path, all_results_df):
    # Define processing functions for each column
    def process_y(data_dict):
        result = {}
        for key, value in data_dict.items():
            if value > 0.5:  # Only interested in values greater than 0.5
                new_key = f"Y_cap_{key[0]}"
                result[new_key] = key[1]
        return result

    def process_z(data_dict):
        count_dict = {}
        operational_cap = {}
        operational_cap_avg = {}

        # Iterate over each key-value pair in the dictionary
        for key, value in data_dict.items():
            first_key = key[0]
            second_key_value = float(key[1])

            if value > 0.1:
                if first_key not in count_dict:
                    count_dict[first_key] = 0
                    operational_cap[first_key] = 0

                if second_key_value > 0:  # Count if the value is greater than zero
                    count_dict[first_key] += 1
                    operational_cap[first_key] += second_key_value  # Sum over the second key
            else:
                if first_key not in count_dict:
                    count_dict[first_key] = 0
                    operational_cap[first_key] = 0

        # Calculate the average operational capacity for each first key
        for key in operational_cap.keys():
            if count_dict[key] > 0:
                operational_cap_avg[key] = float(operational_cap[key] / count_dict[key])
            else:
                operational_cap_avg[key] = 0

        # Prepare the result dict with the required naming convention
        result = {f"Z_count_{key}": count for key, count in count_dict.items()}
        result.update({f"Z_avg_operational_cap_{key}": avg for key, avg in operational_cap_avg.items()})

        return result

    def process_z_num_operating_periods(data_dict):
        num_period_dict = {}

        # Iterate over each key-value pair in the dictionary
        for key, value in data_dict.items():
            if key not in num_period_dict:
                num_period_dict[key] = value
            else:
                print('[process_z_num_operating_periods] this should not happen!')

        # Prepare the result dict with the required naming convention
        result = {f"Z_num_{key}": count for key, count in num_period_dict.items()}

        return result

    def process_z_cap_limit(data_dict):
        limit_tot_cap_dict = {}

        # Iterate over each key-value pair in the dictionary
        for key, value in data_dict.items():
            if key not in limit_tot_cap_dict:
                limit_tot_cap_dict[key] = value
            else:
                print('[process_z_cap_limit] this should not happen!')

        # Prepare the result dict with the required naming convention
        result = {f"Z_total_cap_lim_{key}": count for key, count in limit_tot_cap_dict.items()}

        return result

    def process_x_cap_used(data_dict):
        used_cap_dict = {}

        # Iterate over each key-value pair in the dictionary
        for key, value in data_dict.items():
            if key not in used_cap_dict:
                used_cap_dict[key] = value
            else:
                print('[process_x_cap_used] this should not happen!')

        # Prepare the result dict with the required naming convention
        result = {f"X_cap_used_{key}": count for key, count in used_cap_dict.items()}

        return result

    def process_z_avg_cap_limit(data_dict):
        limit_avg_cap_dict = {}

        # Iterate over each key-value pair in the dictionary
        for key, value in data_dict.items():
            if key not in limit_avg_cap_dict:
                limit_avg_cap_dict[key] = value
            else:
                print('[process_z_avg_cap_limit] this should not happen!')

        # Prepare the result dict with the required naming convention
        result = {f"Z_avg_cap_lim_{key}": count for key, count in limit_avg_cap_dict.items()}

        return result

    def process_x_avg_cap_used(data_dict):
        used_avg_cap_dict = {}

        # Iterate over each key-value pair in the dictionary
        for key, value in data_dict.items():
            if key not in used_avg_cap_dict:
                used_avg_cap_dict[key] = value
            else:
                print('[process_x_avg_cap_used] this should not happen!')

        # Prepare the result dict with the required naming convention
        result = {f"X_avg_cap_used_{key}": count for key, count in used_avg_cap_dict.items()}

        return result

    # Define a mapping from column names to processing functions
    column_processors = {
        'Y': process_y,
        # 'Z': process_z,
        "Satellite_operating_periods": process_z_num_operating_periods,
        "Satellite_total_capacity_limit": process_z_cap_limit,
        "Satellite_total_capacity_used": process_x_cap_used,
        "Satellite_average_capacity_limit": process_z_avg_cap_limit,
        "Satellite_average_capacity_used": process_x_avg_cap_used,
    }

    # Function to process each row and create the new DataFrame
    def process_row(row):
        new_row = {'Filename': row['Filename']}

        for col, processor in column_processors.items():
            if pd.notna(row[col]):  # Check if the column is not NaN
                try:
                    data_dict = ast.literal_eval(row[col])  # Convert string to dictionary
                    new_row.update(processor(data_dict))  # Apply the appropriate processing function
                except (ValueError, SyntaxError) as e:
                    print(f"Error processing {col} in file {row['Filename']}: {e}")

        return new_row

    # Apply the function to each row and create a new DataFrame
    satellite_avg_utilization_df = pd.DataFrame([process_row(row) for _, row in all_results_df.iterrows()])

    # Fill NaN values with empty strings or 0, depending on your preference
    satellite_avg_utilization_df = satellite_avg_utilization_df.fillna('')

    # Define the output CSV file
    output_file_name = "satellites_utilization"
    output_csv = os.path.join(path, f"{output_file_name}.csv")

    # Write the DataFrame to a CSV file
    basecase_df.to_csv(output_csv, index=False)

    # output
    return satellite_avg_utilization_df


def evaluate_bc(path_bc, path_eval, path_fixed_costs, results_bc):
    # Table columns
    column_names = ['n', 'm', 'cap', 'x continuous', 'flexibility', 'alpha',
                    'y', 'bc values', 'fixed cost', 'evaluations', 'avg fixed costs',
                    'upper limit 095', 'bc values list', 'avg bc values', 'lower limit 095', 'statistical opt gap']

    N = [5, 10, 15, 20]
    capacity_satellites = [4, 7]
    is_continuous_x = [True, False]
    type_of_flexibility = [2]  # [1, 2, 3]
    type_of_cost_serving = [2]
    alpha = [1.0]
    beta = [1.0]
    split = [6]

    # Load the fixed costs data from the Excel file
    fixed_costs_df = pd.read_excel(path_fixed_costs)

    # Ensure that the 'cost_fixed' column is converted from string to dictionary
    fixed_costs_df['cost_fixed'] = fixed_costs_df['cost_fixed'].apply(ast.literal_eval)

    # Generate all combinations of the parameters excluding m
    parameters_combinations = itertools.product(
        N, capacity_satellites, is_continuous_x, type_of_flexibility, type_of_cost_serving, alpha, beta, split
    )

    df_dict = {}
    for comb in parameters_combinations:
        # Extract the values for this combination
        n, cap, x_continuous, flexibility, cost_serving, alpha_val, beta_val, split_val = comb

        # Create a DataFrame for this specific combination
        rows = []
        # average fixed costs fY for m = 1 to 20
        total_fixed_cost_sum = 0
        bc_values_list = []
        for m in range(1, 21):
            unique_id_parts = list(
                map(str, [n, m, cap, x_continuous, flexibility, cost_serving, alpha_val, beta_val, split_val]))
            solution_id = f"ID_{'_'.join(unique_id_parts)}"
            solution_id_without_split = f"ID_{'_'.join(unique_id_parts[:-1])}"

            # Find the row in results_bc with the matching ID
            match_row = results_bc.loc[results_bc['ID'] == solution_id]

            if not match_row.empty and len(match_row) == 1:
                # Extract and safely evaluate the dictionary from the Y column
                y_dict = match_row.iloc[0]['Y']
                if isinstance(y_dict, str):
                    y_dict = ast.literal_eval(y_dict)  # Convert string to dictionary if needed
                bc_obj_value = match_row.iloc[0]['objective_value']
            else:
                print(f"[evaluate_bc] This should not happen! ID {solution_id} is missing or duplicated")
                y_dict = {}
                bc_obj_value = 0

            bc_values_list.append(bc_obj_value)

            # Calculate the fixed cost based on y_dict
            total_fixed_cost = 0
            for key, value in y_dict.items():
                if value > 0.1 and int(key[1]) > 0.1:
                    # Find the corresponding cost_fixed for the satellite
                    fixed_cost_row = fixed_costs_df[fixed_costs_df['id_satellite'] == key[0]]
                    if not fixed_cost_row.empty:
                        cost_fixed_dict = fixed_cost_row.iloc[0]['cost_fixed']
                        # Add the fixed cost for this satellite/capacity
                        if str(key[1]) in cost_fixed_dict.keys():
                            total_fixed_cost += cost_fixed_dict[str(key[1])] * value
                        else:
                            print(f"[evaluate_bc] Capacity {str(key[1])} not found in cost_fixed for satellite {key[0]}")
                    else:
                        print(f"[evaluate_bc] No fixed cost found for satellite {key[0]}")

            total_fixed_cost_sum += total_fixed_cost  # Accumulate fixed costs

            # Collect evaluations from the path_eval files
            evaluations = []
            if m == 1:
                num_files = 0
                for filename in os.listdir(path_eval):
                    if filename.startswith(
                            f"solution_branch_and_cut_{solution_id_without_split}_evaluated_on_scenario_") and filename.endswith(
                            ".json"):
                        num_files += 1
                        file_path = os.path.join(path_eval, filename)
                        with open(file_path, 'r') as file:
                            data = json.load(file)
                            if 'total_cost' in data:
                                evaluations.append(data['total_cost'])

                if num_files < 100:
                    print(f"[evaluate_bc] There should be 100 files for ID {solution_id}")

            row = {
                'n': n,
                'm': m,
                'cap': cap,
                'x continuous': x_continuous,
                'flexibility': flexibility,
                'alpha': alpha_val,
                'y': y_dict,
                'bc values': bc_obj_value,
                'fixed cost': total_fixed_cost,
                'evaluations': evaluations, # saved only in m = 1
                'avg fixed costs': None, # saved only in m = 1
                'upper limit 095': 0, # todo
                'bc values list': [], # saved only in m = 1
                'avg bc values': 0, # saved only in m = 1
                'lower limit 095': 0, # todo
                'statistical opt gap': 0, # todo
            }
            rows.append(row)

        # Calculate the average fixed cost across all m
        avg_fixed_cost = total_fixed_cost_sum / 20

        # Update each row with the calculated average fixed cost
        for row in rows:
            if row['m'] == 1:
                row['avg fixed costs'] = avg_fixed_cost
                row['bc values list'] = bc_values_list
                row['avg bc values'] = sum(bc_values_list) / len(bc_values_list) if bc_values_list else 0

        # Convert the list of rows into a DataFrame once after the loop
        df = pd.DataFrame(rows, columns=column_names)

        # Store the DataFrame in the dictionary using the combination (excluding m)
        df_dict[f"n{n}_cap{cap}_x{x_continuous}_flex{flexibility}_alpha{alpha_val}_beta{beta_val}_split{split_val}"] = df

    return df_dict


def export_evaluation_bc_dataframes(evaluation_df, output_directory):
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    for key, df in evaluation_df.items():
        # Define the filename using the key
        filename = f"{key}.csv"

        # Define the full path for the file
        file_path = os.path.join(output_directory, filename)

        # Export the DataFrame to a CSV file
        df.to_csv(file_path, index=False)

        print(f"Exported DataFrame to {file_path}")


if __name__ == "__main__":
    """
    Running: 
        - Consolidate results
        - Analyze results
    """

    ROOT_PATH = "/Users/juanpina/Dropbox (MIT)/01 Postdoctoral Research/02 Work in Progress/code_multi_period"
    FOLDER_PATH_BC = ROOT_PATH + "/results/bc/"
    FOLDER_PATH_EVAL = ROOT_PATH + "/results/evaluation_bc/"
    FOLDER_PATH_EXPECTED = ROOT_PATH + "/results/expected/"
    FOLDER_PATH_SAA = ROOT_PATH + "/results/extended_saa_model/"
    FOLDER_PATH_EXPORT = ROOT_PATH + "/results/export_evaluation_bc/"
    FOLDER_PATH_BASECASE = ROOT_PATH + "/results/export_basecase_bc/"
    FOLDER_SATELLITES_FIXED_COSTS = ROOT_PATH + "/data/input_satellites.xlsx"

    FOLDER_TO_SAVE_RESULTS = ROOT_PATH + "/results/consolidated_results/"

    # (1) Consolidate results:
    print("Consolidate BC - Start")
    all_results_bc_df = consolidate_bc(FOLDER_PATH_BC, FOLDER_TO_SAVE_RESULTS)
    print("Consolidate BC - Done")

    # all_results_evaluation_bc_df = consolidate_eval_bc(FOLDER_PATH_EVAL)
    # all_results_evaluation_expected_df = consolidate_eval_expected(FOLDER_PATH_EVAL)
    # all_results_expected_df = consolidate_expected(FOLDER_PATH_EXPECTED)
    # all_results_saa_df = consolidate_saa(FOLDER_PATH_SAA)

    # (2) Analyze results:
    print("Consolidate base case - Start")
    basecase_df = calculate_basecase_results_bc(FOLDER_PATH_BC, FOLDER_PATH_BASECASE, FOLDER_TO_SAVE_RESULTS, all_results_bc_df)
    print("Consolidate base case - Done")

    # sat_avg_utilization_df = calculate_satellites_utilization(FOLDER_PATH_EVAL, all_results_evaluation_bc_df)

    print("Consolidate Evaluation BC - Start")
    evaluation_bc_df = evaluate_bc(FOLDER_PATH_BC, FOLDER_PATH_EVAL, FOLDER_SATELLITES_FIXED_COSTS,  all_results_bc_df)
    print("Consolidate Evaluation BC - Done")

    export_evaluation_bc_dataframes(evaluation_bc_df, FOLDER_TO_SAVE_RESULTS)