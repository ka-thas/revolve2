"""Main script for the example."""
import concurrent.futures
import logging
import time
from random import random

import multineat
import numpy as np
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

import config
import selection
from database_components.base import Base
from database_components.experiment import Experiment
from database_components.generation import Generation
from database_components.genotype import Genotype
from database_components.individual import Individual
from database_components.learn_genotype import LearnGenotype
from database_components.learn_individual import LearnIndividual
from database_components.population import Population
from evaluator import Evaluator
from revolve2.experimentation.database import OpenMethod, open_database_sqlite
from revolve2.experimentation.logging import setup_logging
from revolve2.experimentation.rng import make_rng, seed_from_time
from revolve2.modular_robot.body.v1 import ActiveHingeV1


def run_experiment(dbengine: Engine) -> None:
    """
    Run an experiment.

    :param dbengine: An openened database with matching initialize database structure.
    """
    logging.info("----------------")
    logging.info("Start experiment")

    # Set up the random number generator.
    rng_seed = seed_from_time() % 2**32
    rng = make_rng(rng_seed)

    # Create and save the experiment instance.
    experiment = Experiment(rng_seed=rng_seed)
    logging.info("Saving experiment configuration.")
    with Session(dbengine) as session:
        session.add(experiment)
        session.commit()

    # Initialize the evaluator that will be used to evaluate robots.
    evaluator = Evaluator(headless=True, num_simulators=config.NUM_SIMULATORS)
    innov_db_body = multineat.InnovationDatabase()

    # Create an initial population.
    logging.info("Generating initial population.")

    initial_genotypes = [
        Genotype.initialize(
            innov_db_body, rng
        )
        for _ in range(config.POPULATION_SIZE)
    ]

    # Evaluate the initial population.
    logging.info("Evaluating initial population.")

    initial_objective_values, initial_genotypes = learn_population(genotypes=initial_genotypes, evaluator=evaluator, dbengine=dbengine, rng=rng)

    # Create a population of individuals, combining genotype with fitness.
    individuals = []
    for objective_value, genotype in zip(initial_objective_values, initial_genotypes):
        individual = Individual(genotype=genotype, objective_value=objective_value, original_generation=0)
        individuals.append(individual)
    population = Population(
        individuals=individuals
    )
    selection.calculate_reproduction_fitness(population)
    generation = Generation(
        experiment=experiment, generation_index=0, population=population
    )
    logging.info("Saving generation.")
    with Session(dbengine, expire_on_commit=False) as session:
        session.add(generation)
        session.commit()

    # Start the actual optimization process.
    logging.info("Start optimization process.")
    while generation.generation_index < config.NUM_GENERATIONS:
        logging.info(
            f"Real generation {generation.generation_index + 1} / {config.NUM_GENERATIONS}."
        )

        offspring_genotypes = selection.generate_offspring(innov_db_body, rng, population)

        # Evaluate the offspring.
        offspring_objective_values, offspring_genotypes = learn_population(genotypes=offspring_genotypes, evaluator=evaluator, dbengine=dbengine, rng=rng)

        # Make an intermediate offspring population.
        offspring_individuals = [
            Individual(genotype=genotype, objective_value=objective_value, original_generation=generation.generation_index + 1) for
            genotype, objective_value in zip(offspring_genotypes, offspring_objective_values)]
        offspring_population = Population(
                individuals=offspring_individuals
            )
        # Create the next population by selecting survivors.
        selection.calculate_survival_fitness(population, offspring_population)
        population = selection.select_survivors(rng, population, offspring_population)
        # TODO: Change selection method
        selection.calculate_reproduction_fitness(population)

        # Make it all into a generation and save it to the database.
        generation = Generation(
            experiment=experiment,
            generation_index=generation.generation_index + 1,
            population=population,
        )
        logging.info("Saving real generation.")
        with Session(dbengine, expire_on_commit=False) as session:
            session.add(generation)
            session.commit()


def learn_population(genotypes, evaluator, dbengine, rng):
    result_objective_values = []
    new_genotypes = []
    for genotype in genotypes:
        objective_value, best_individual, learn_individuals = learn_genotype_cppn(genotype, evaluator, rng)
        result_objective_values.append(objective_value)
        new_genotypes.append(best_individual.morphology_genotype)

        for learn_individual in learn_individuals:
            with Session(dbengine, expire_on_commit=False) as session:
                session.add(learn_individual)
                session.commit()
    return result_objective_values, new_genotypes


def learn_genotype(genotype: Genotype, evaluator, rng: np.random.Generator):
    # We get the brain uuids from the developed body, because if it is too big we don't want to learn unused uuids
    developed_body = genotype.develop_body()
    genotype.update_brain_parameters(developed_body, rng)

    best_objective_value = None
    best_individual = None
    learn_individuals = []

    next_brains = [genotype.brain]
    for i in range(config.LEARN_POPULATION_SIZE - 1):
        next_brains.append(genotype.mutate_brain(rng).brain)

    logging.info(f"Start training initial learn population.")
    population = []
    for next_brain in next_brains:
        objective_value, new_learn_genotype = run_brain(developed_body, next_brain, evaluator)

        learn_individual = LearnIndividual(morphology_genotype=genotype, genotype=new_learn_genotype,
                                           objective_value=objective_value, generation_index=0)

        if best_objective_value is None or objective_value >= best_objective_value:
            best_objective_value = objective_value
            best_individual = learn_individual

        learn_individuals.append(learn_individual)
        population.append((next_brain, objective_value))

    logging.info(f"Finish training initial learn population.")

    for i in range(config.LEARN_NUM_GENERATIONS):
        logging.info(f"Learn generation {i + 1} / {config.LEARN_NUM_GENERATIONS}.")
        brains = [individual[0] for individual in population]

        triplet = rng.choice(brains, 3, False)
        next_brains = create_new_triplet(triplet, genotype.brain.keys())

        for next_brain in next_brains:
            objective_value, new_learn_genotype = run_brain(developed_body, next_brain, evaluator)

            learn_individual = LearnIndividual(morphology_genotype=genotype, genotype=new_learn_genotype,
                                               objective_value=objective_value, generation_index=i+1)

            if best_objective_value is None or objective_value >= best_objective_value:
                best_objective_value = objective_value
                best_individual = learn_individual

            learn_individuals.append(learn_individual)
            population.append((next_brain, objective_value))
            population = sorted(population, key=lambda t: t[1], reverse=True)[:config.LEARN_POPULATION_SIZE]

    return best_objective_value, best_individual, learn_individuals

def learn_genotype_cppn(genotype: Genotype, evaluator, rng: np.random.Generator):
    developed_body = genotype.develop_body()
    learn_individuals = []
    population = []
    next_brains = []

    if len(genotype.best_population) == 0:
        genotype.innov_db_brain = multineat.InnovationDatabase()
        for _ in range(config.LEARN_POPULATION_SIZE):
            brain = LearnGenotype.random_brain(genotype.innov_db_brain, rng)
            new_learn_genotype = LearnGenotype(brain=brain.brain)
            learn_individual = LearnIndividual(morphology_genotype=genotype, genotype=new_learn_genotype,
                                               objective_value=0, generation_index=0)
            genotype.best_population.append(learn_individual)

    for individual in genotype.best_population:
        next_brains.append(individual.genotype.brain)
    best_individual = None
    best_objective_value = None

    for i in range(config.LEARN_NUM_GENERATIONS):
        for next_brain in next_brains:
            objective_value, new_learn_genotype = run_brain(developed_body, next_brain, evaluator)
            learn_individual = LearnIndividual(morphology_genotype=genotype, genotype=new_learn_genotype,
                                               objective_value=objective_value, generation_index=i + 1)

            if best_objective_value is None or objective_value >= best_objective_value:
                best_objective_value = objective_value
                best_individual = learn_individual

            learn_individuals.append(learn_individual)
            population.append(learn_individual)
        population = sorted(population, key=lambda t: t.objective_value, reverse=True)[:config.LEARN_POPULATION_SIZE]

        next_brains = []
        for j in range(config.LEARN_POPULATION_SIZE):
            parent1, parent2 = rng.choice(population[:int(config.LEARN_POPULATION_SIZE / 2)], 2)
            child = LearnGenotype.crossover_brain(parent1.genotype, parent2.genotype, rng).mutate_brain(genotype.innov_db_brain, rng)
            next_brains.append(child.brain)


    best_individual.morphology_genotype.best_population = population
    return best_objective_value, best_individual, learn_individuals

def run_brain(developed_body, brain, evaluator):
    new_learn_genotype = LearnGenotype(brain=brain)
    robot = new_learn_genotype.develop(developed_body)

    # Evaluate them.
    start_time = time.time()
    objective_value = evaluator.evaluate(robot)
    end_time = time.time()
    new_learn_genotype.execution_time = end_time - start_time

    return objective_value, new_learn_genotype

def create_new_triplet(old_triplet, brain_keys):
    new_triplet = [{}, {}, {}]
    for brain_key in brain_keys:
        new_triplet[0][brain_key] = (old_triplet[0][brain_key] +
                                     config.SCALING_FACTOR * (old_triplet[1][brain_key] - old_triplet[2][brain_key]))
        new_triplet[1][brain_key] = (old_triplet[1][brain_key] +
                                     config.SCALING_FACTOR * (old_triplet[2][brain_key] - new_triplet[0][brain_key]))
        new_triplet[2][brain_key] = (old_triplet[2][brain_key] +
                                     config.SCALING_FACTOR * (new_triplet[0][brain_key] - new_triplet[1][brain_key]))

    # TODO: Crossover

    return new_triplet

def main() -> None:
    """Run the program."""

    # Set up logging.
    setup_logging(file_name="log.txt")

    # Open the database, only if it does not already exists.
    dbengine = open_database_sqlite(
        config.DATABASE_FILE, open_method=OpenMethod.OVERWITE_IF_EXISTS
    )
    # Create the structure of the database.
    Base.metadata.create_all(dbengine)

    # Run the experiment several times.
    for _ in range(config.NUM_REPETITIONS):
        run_experiment(dbengine)


if __name__ == "__main__":
    main()
