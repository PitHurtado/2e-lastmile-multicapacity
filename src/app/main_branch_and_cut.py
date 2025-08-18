"""Module of main branch and cut."""
import json
from src.instance.instance import Instance
from src.model.branch_and_cut import Branch_and_Cut
from src.utils import LOGGER as logger


class Main:
    def __init__(self, folder_path: str,
                 combination: list,
                 run_time: int,
                 profiler: bool,
                 ):
        self.folder_path = folder_path
        self.combination = combination
        (
            ID,
            N,
            M,
            capacity_satellites,
            is_continuous_x,
            type_of_flexibility,
            type_of_cost_serving,
            alpha,
            split,
            warm_start_MP,
            warm_start_SP,
            reformulated,
            valid_inequalities,
        ) = self.combination

        logger.info(f"[MAIN BRANCH AND CUT] Reading instance: {self.combination}")
        self.instance = Instance(
            id_instance=ID,
            capacity_satellites=capacity_satellites,
            is_continuous_x=is_continuous_x,
            alpha=alpha,
            type_of_flexibility=type_of_flexibility,
            type_of_cost_serving=type_of_cost_serving,
            periods=12,
            N=N,
            is_evaluation=False,
            scenario_evaluation_id=1, # This number is not used.
        )

        self.warm_start_MP = warm_start_MP
        self.split = split
        self.warm_start_SP = warm_start_SP
        self.valid_inequalities = valid_inequalities
        self.reformulated = reformulated
        self.max_run_time = run_time
        self.split = split
        self.profiler = profiler

        logger.info(f"[MAIN BRANCH AND CUT] Instance read: {combination}")
        self.solver = Branch_and_Cut(self.instance, self.warm_start_MP, self.valid_inequalities, self.max_run_time, self.profiler, self.split, self.warm_start_SP, self.reformulated)

    def solve(self):
        # (1) Solve branch and cut:
        logger.info(f"[MAIN BRANCH AND CUT] Solving started")
        self.solver.solve()
        logger.info(f"[MAIN BRANCH AND CUT] Done")

        # (2) Save results
        logger.info(f"[MAIN BRANCH AND CUT] Saving results")
        results = {"Y": str(self.solver.get_y_solution())}
        results.update(self.solver.get_final_metrics())
        results.update(self.solver.Cuts.get_metrics())
        results.update(self.get_information())

        file_name = self.folder_path + f"test_solution_branch_and_cut_{self.instance.id_instance}.json"
        results_to_save = {str(key): value for key, value in results.items()}

        with open(file_name, "w") as json_file:
            json.dump(results_to_save, json_file, indent = 4)

        logger.info(f"[MAIN BRANCH AND CUT] Results saved in {file_name}")

    def get_information(self):
        """Return a dictionary with info combination."""
        (
            ID,
            N,
            M,
            capacity_satellites,
            is_continuous_x,
            type_of_flexibility,
            type_of_cost_serving,
            alpha,
            split,
            warm_start_MP,
            warm_start_SP,
            reformulated,
            valid_inequalities,
        ) = self.combination

        info_combination = {
                "ID": ID,
                "m": M,
                "N": N,
                "capacity_satellites": capacity_satellites,
                "is_continuous_x": is_continuous_x,
                "type_of_flexibility": type_of_flexibility,
                "type_of_cost_serving": type_of_cost_serving,
                "alpha": alpha,
                "periods": 12,
                "max_run_time": self.max_run_time,
                "warm_start_MP": self.warm_start_MP,
                "split": self.split,
                "warm_start_SP": self.warm_start_SP,
                "valid_inequalities": self.valid_inequalities,
                "reformulated": self.reformulated,
        }

        return info_combination
