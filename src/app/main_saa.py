"""Main module for the SAA application."""
from typing import Any, Dict, List

from src.instance.experiment import Experiment
from src.model.sample_average_approximation import SampleAverageApproximation
from src.utils import LOGGER as logger

if __name__ == "__main__":
    # (1) Generate instance:
    folder_path = "../results/saa/"
    logger.info("[MAIN SAA] Generating instances")
    instances_generated: List[Dict[str, Any]] = Experiment(
        N_evaluation=2, M=2, folder_path=folder_path
    ).generate_instances()  # TODO change this
    logger.info("[MAIN SAA] Instances generated")

    # (2) Solve and save results:
    run_time = 3600
    for i, experiment in enumerate(instances_generated):
        id_experiment = (1 + i) * 100
        logger.info(f"[MAIN SAA] Experiment {id_experiment} started")
        solver_saa = SampleAverageApproximation(
            experiment=experiment,
            id_experiment=id_experiment,
        )
        solver_saa.run()
        logger.info(f"[MAIN SAA] Experiment {id_experiment} finished")
