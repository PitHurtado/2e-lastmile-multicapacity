"""Module to define Sub Problem and Cut Generator"""
import sys
import time
from typing import Any, Dict

import numpy as np
from gurobipy import GRB, quicksum

from src.classes import Satellite
from src.instance.instance import Instance
from src.model.master_problem import MasterProblem
from src.model.sub_problem import SubProblem
from src.utils import LOGGER as logger


class Cuts:
    """Class to define the Cut Generator"""

    def __init__(self, instance: Instance, LB: Dict[Any, float], valid_inequalities: bool, split: int, warm_start_subproblems: bool, reformulated: bool):

        Cuts.valid_inequalities = valid_inequalities
        Cuts.SPs: Dict[Any, SubProblem] = self.__create_subproblems(instance, split, warm_start_subproblems)
        Cuts.LB = LB

        # Parameters
        Cuts.periods: int = instance.periods
        Cuts.satellites: Dict[str, Satellite] = instance.satellites
        Cuts.instance = instance
        Cuts.reformulated = reformulated

        # Other parameters
        Cuts.optimality_cuts = 0
        Cuts.best_solution = {}
        Cuts.upper_bound_updated = 0
        Cuts.upper_bound = sys.maxsize
        Cuts.subproblem_solved = 0
        Cuts.start_time = 0
        Cuts.time_best_solution_found = 0
        Cuts.run_times = []

    @staticmethod
    def add_cuts(model, where) -> None:
        """Add cuts"""
        if where == GRB.Callback.MIPSOL:
            Cuts.add_cut_integer_solution(model)
            logger.info(f"[CUT] Optimality cuts: {Cuts.optimality_cuts}")

        # elif where == GRB.Callback.MIPNODE:
        #     if model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL and Cuts.valid_inequalities:
        #         Cuts.add_cut_fractional_solution(model)

    @staticmethod
    def add_cut_integer_solution(model: MasterProblem) -> None:
        """Add optimality cuts."""
        # Retrieve current solution
        Y = model.cbGetSolution(model._Y)
        θ = model.cbGetSolution(model._θ)

        # Solve subproblems
        total_subproblem_cost = 0
        C = {}
        if Cuts.reformulated:
            X = {}
            W = {}

        for n, scenario in Cuts.instance.scenarios.items():
            logger.info(f"[CUT] Solving subproblem: Scenario {n}")
            subproblem_runtime, subproblem_costs, X_values, W_values, Z_values = Cuts.SPs[n].solve_model(fixed_y=Y)

            Cuts.subproblem_solved += 1
            total_subproblem_cost = 0
            for t in range(Cuts.periods):
                # Saving subproblem values
                C[(n,t)] = subproblem_costs[t]
                total_subproblem_cost += subproblem_costs[t]

                if Cuts.reformulated:
                    # Saving X and W values
                    for k in scenario.pixels.keys():
                        for s, satellite in Cuts.satellites.items():
                            X[(s, k, n, t)] = X_values[(s, k, t)]

                        W[(k, n, t)] = W_values[(k,t)]

        Cuts.run_times.append(subproblem_runtime)

        logger.info(f"[CUT] Subproblems solved")

        total_cost = (
            np.sum(
                [
                    satellite.cost_fixed[q] * Y[(s, q)]
                    for s, satellite in Cuts.instance.satellites.items()
                    for q, capacity in satellite.capacity.items() if capacity > 0
                ]
            )
            + (1 / len(Cuts.instance.scenarios)) * total_subproblem_cost
        )

        # Update upper bound and best solution found so far
        if total_cost < Cuts.upper_bound:
            Cuts.upper_bound = total_cost
            Cuts.best_solution = Y
            Cuts.time_best_solution_found = round(time.time() - Cuts.start_time, 3)
            Cuts.upper_bound_updated += 1
            model.cbSetSolution(model._Y, Y)
            model.cbSetSolution(model._θ, C)

            if Cuts.reformulated:
                model.cbSetSolution(model._X, X)
                model.cbSetSolution(model._W, W)

            model.cbUseSolution()

        # Add optimality cuts
        ε = 0.01 # 25
        one_violated = False
        for n in Cuts.instance.scenarios.keys():
            for t in range(Cuts.periods):
                if θ[(n,t)] + ε < C[(n,t)]:
                    one_violated = True
                    lhs = model._θ[(n,t)]

                    if Cuts.valid_inequalities:
                        # Valid inequalities:
                        act_function = Cuts.get_activation_function_valid_inequalities(model, Y)
                        rhs = C[(n,t)] + (C[(n,t)] - Cuts.LB[(n,t)]) * act_function

                    else:
                        # Classical optimality cuts of Laporte:
                        act_function = Cuts.get_activation_function_laporte(model, Y)
                        rhs = Cuts.LB[(n,t)] + (C[(n,t)] - Cuts.LB[(n,t)]) * act_function

                    model.cbLazy(lhs >= rhs)
                    Cuts.optimality_cuts += 1

        if one_violated:
            if any(Y[(s, q)] > 0.5 for s, satellite in Cuts.satellites.items() for q, capacity in satellite.capacity.items() if capacity > 0):
                act_function = Cuts.get_activation_function_laporte(model, Y)
                model.cbLazy(act_function <= 0)
                Cuts.optimality_cuts += 1

        # print(f"Cuts: {Cuts.optimality_cuts}")

    @staticmethod
    def add_cut_fractional_solution(model):
        # Retrieve current solution
        Y = model.cbGetNodeRel(model._Y)
        θ = model.cbGetNodeRel(model._θ)

        # Construct integer feasible solution
        Y_feasible = {}
        epsilon = 0.01
        for s, satellite in Cuts.satellites.items():
            for q in satellite.capacity.keys():
                Y_feasible[(s,q)] = Y[(s,q)]
                if epsilon < Y[(s, q)] < 1 - epsilon:
                    for q_key, q_value in satellite.capacity.items():
                        Y_feasible[(s,q_key)] = int(q_value == max(q for q in satellite.capacity.values()))
                    break

        # Solve subproblems
        total_subproblem_cost = 0
        C = {}
        if Cuts.reformulated:
            X = {}
            W = {}

        for n, scenario in Cuts.instance.scenarios.items():
            logger.info(f"[CUT] Solving subproblem: Scenario {n}")
            subproblem_runtime, subproblem_costs, X_values, W_values, Z_values = Cuts.SPs[n].solve_model(fixed_y=Y_feasible)

            Cuts.subproblem_solved += 1
            total_subproblem_cost = 0
            for t in range(Cuts.periods):
                # Saving subproblem values
                C[(n,t)] = subproblem_costs[t]
                total_subproblem_cost += subproblem_costs[t]

                if Cuts.reformulated:
                    # Saving X and W values
                    for k in scenario.pixels.keys():
                        for s, satellite in Cuts.satellites.items():
                            X[(s, k, n, t)] = X_values[(s, k, t)]

                        W[(k, n, t)] = W_values[(k,t)]

        Cuts.run_times.append(subproblem_runtime)

        logger.info(f"[CUT] Subproblems solved")

        total_cost = (
                np.sum(
                    [
                        satellite.cost_fixed[q] * Y_feasible[(s, q)]
                        for s, satellite in Cuts.instance.satellites.items()
                        for q, capacity in satellite.capacity.items() if capacity > 0
                    ]
                )
                + (1 / len(Cuts.instance.scenarios)) * total_subproblem_cost
        )

        # Update upper bound and best solution found so far
        if total_cost < Cuts.upper_bound:
            Cuts.upper_bound = total_cost
            Cuts.best_solution = Y_feasible
            Cuts.time_best_solution_found = round(time.time() - Cuts.start_time, 3)
            Cuts.upper_bound_updated += 1
            model.cbSetSolution(model._Y, Y_feasible)
            model.cbSetSolution(model._θ, C)

            if Cuts.reformulated:
                model.cbSetSolution(model._X, X)
                model.cbSetSolution(model._W, W)

            model.cbUseSolution()

        # Add optimality cuts
        ε = 1e-4
        one_violated = False
        for n in Cuts.instance.scenarios.keys():
            for t in range(Cuts.periods):
                if θ[(n,t)] + ε < C[(n,t)]:
                    one_violated = True
                    lhs = model._θ[(n,t)]
                    act_function = Cuts.get_activation_function_valid_inequalities(model, Y_feasible)
                    rhs = C[(n,t)] + (C[(n,t)] - Cuts.LB[(n,t)]) * act_function

                    model.cbLazy(lhs >= rhs)
                    Cuts.optimality_cuts += 1

        if one_violated:
            if any(Y_feasible[(s, q)] > 0.5 for s, satellite in Cuts.satellites.items() for q, capacity in satellite.capacity.items() if capacity > 0):
                act_function = Cuts.get_activation_function_laporte(model, Y_feasible)
                model.cbLazy(act_function <= 0)
                Cuts.optimality_cuts += 1

    @staticmethod
    def get_activation_function_laporte(model, Y):
        """Get the activation function Laporte cuts"""
        epsilon = 1e-4
        activation = (
            quicksum(
                model._Y[(s, q)]
                for s, satellite in Cuts.satellites.items()
                for q in satellite.capacity.keys()
                if Y[(s, q)] + epsilon >= 1
            )
            - np.sum(
                [
                    1
                    for s, satellite in Cuts.satellites.items()
                    for q in satellite.capacity.keys()
                    if Y[(s, q)] + epsilon >= 1
                ]
            )
            + 1
        )
        return activation

    @staticmethod
    def get_activation_function_valid_inequalities(model, Y):
        """Get the activation function Inequalities (18)"""
        epsilon = 1e-4
        B = [
            (s, q_key)
            for s, satellite in Cuts.satellites.items()
            for q_key, q_value in satellite.capacity.items()
            for q_bar_key, q_bar_value in satellite.capacity.items()
            if Y[(s, q_bar_key)] >= 1 - epsilon and q_value <= q_bar_value
        ]

        B_complement = set(
            (s, q_key)
            for s, satellite in Cuts.satellites.items()
            for q_key, q_value in satellite.capacity.items()
        ) - set(B)

        activation = (
            quicksum(
                model._Y[pair]
                for pair in B
            )
            - quicksum(
                model._Y[pair]
                for pair in B_complement
            )
            - np.sum(
                [
                    1
                    for s, satellite in Cuts.satellites.items()
                    for q_bar in satellite.capacity.keys()
                    if Y[(s, q_bar)] >= 1 - epsilon
                ]
            )
        )
        return activation

    def __create_subproblems(self, instance: Instance, split, warm_start_subproblems) -> Dict[Any, SubProblem]:
        """Create the subproblems"""
        subproblems = {}
        for n in instance.scenarios.keys():
            scenario = instance.scenarios[n]
            subproblems[n] = SubProblem(instance, instance.periods, scenario, split, warm_start_subproblems)

        return subproblems

    @staticmethod
    def set_start_time(start_time):
        """Set start time"""
        Cuts.start_time = start_time


    @staticmethod
    def get_metrics():
        metrics = {'time_best_known_solution': Cuts.time_best_solution_found, # In seconds
                   'total_cuts': Cuts.optimality_cuts,
                   'upper_bound_updates': Cuts.upper_bound_updated,
                   'subproblems_solved': Cuts.subproblem_solved}

        return metrics
