"""Main script for the example."""

import logging
import concurrent.futures
import os
from argparse import ArgumentParser

import numpy as np
from bayes_opt import acquisition, BayesianOptimization
from sklearn.gaussian_process.kernels import Matern

import config
from database_components.base import Base
from evaluator import Evaluator
from database_components.experiment import Experiment
from database_components.generation import Generation
from database_components.population import Population
from sqlalchemy.orm import Session
import body_getter
from database_components.genotype import Genotype
from database_components.individual import Individual

from revolve2.experimentation.database import OpenMethod, open_database_sqlite
from revolve2.experimentation.rng import seed_from_time, make_rng


def run_experiment(genotype, experiment, file):
    logging.info("----------------")
    logging.info("Start experiment")

    # Open the database, only if it does not already exists.
    dbengine = open_database_sqlite(
        file.replace(".sqlite", "") + "_" + str(experiment) + ".sqlite", open_method=OpenMethod.NOT_EXISTS_AND_CREATE
    )
    # Create the structure of the database.
    Base.metadata.create_all(dbengine)

    # Create an rng seed.
    rng_seed = seed_from_time() % 2**32  # Cma seed must be smaller than 2**32.
    rng = make_rng(rng_seed)

    # Create and save the experiment instance.
    experiment = Experiment(rng_seed=rng_seed)
    logging.info("Saving experiment configuration.")
    with Session(dbengine) as session:
        session.add(experiment)
        session.commit()

    # Intialize the evaluator that will be used to evaluate robots.
    evaluator = Evaluator(
        headless=True,
        num_simulators=config.NUM_SIMULATORS,
        environment='noisy'
    )

    pbounds = {}

    for uuid in genotype.brain.keys():
        pbounds['amplitude_' + str(uuid)] = [0, 1]
        pbounds['phase_' + str(uuid)] = [0, 1]
        pbounds['offset_' + str(uuid)] = [0, 1]

    optimizer = BayesianOptimization(
        f=None,
        pbounds=pbounds,
        allow_duplicate_points=True,
        random_state=int(rng.integers(low=0, high=2 ** 32)),
        acquisition_function=acquisition.UpperConfidenceBound(kappa=3,
                                                              random_state=rng.integers(low=0, high=2 ** 32))
    )
    optimizer.set_gp_params(alpha=1e-10, kernel=Matern(nu=config.NU, length_scale=config.LENGTH_SCALE, length_scale_bounds="fixed"))

    # Run cma for the defined number of generations.
    logging.info("Start optimization process.")

    for i in range(config.NUM_GENERATIONS + config.NUM_RANDOM_SAMPLES):
        logging.info(f"Generation {i + 1} / {config.NUM_GENERATIONS + config.NUM_RANDOM_SAMPLES}.")

        next_point = optimizer.suggest()

        new_learn_genotype = Genotype(brain={}, body=genotype.body)
        for brain_uuid in genotype.brain.keys():
            new_learn_genotype.brain[brain_uuid] = np.array(
                [
                    next_point['amplitude_' + str(brain_uuid)],
                    next_point['phase_' + str(brain_uuid)],
                    next_point['offset_' + str(brain_uuid)],
                ]
            )
        robot = new_learn_genotype.develop()

        fitness = evaluator.evaluate(robot)

        optimizer.register(params=next_point, target=fitness)
        print(f"Fitness: {fitness}")

        population = Population(
            individuals=[
                Individual(genotype=new_learn_genotype, fitness=fitness)
            ]
        )

        # Make it all into a generation and save it to the database.
        generation = Generation(
            experiment=experiment,
            generation_index=i,
            population=population,
        )

        with Session(dbengine, expire_on_commit=False) as session:
            session.add(generation)
            session.commit()
    return True


def read_args():
    # Read args
    parser = ArgumentParser()
    parser.add_argument("--learn_environment", required=True)
    args = parser.parse_args()

    return "learn-", args.learn_environment


def run_experiments():
    file_name = 'learn-30'
    folder = "results/0301"
    files = [file for file in os.listdir(folder) if file.startswith(file_name)]
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=config.NUM_PARALLEL_PROCESSES
    ) as executor:
        futures = []
        for file in files:
            genotype = body_getter.get_best_genotype(folder + "/" + file)
            for i in range(1, config.RUNS + 1):
                futures.append(executor.submit(run_experiment, genotype, i, file))
    for future in futures:
        future.result()


def main() -> None:
    """Run the program."""
    # Set up logging.
    run_experiments()


if __name__ == "__main__":
    main()
