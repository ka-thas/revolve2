"""Main script for the example."""
import concurrent.futures
import logging
import time
from argparse import ArgumentParser

import numpy as np
from bayes_opt import acquisition, BayesianOptimization
from matplotlib import pyplot as plt
from sklearn.gaussian_process.kernels import Matern
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import BayesianRidge, Ridge, LinearRegression
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

import config
import selection
from custom_bayesian_optimization import CustomBayesianOptimization
from custom_upper_confidence_bound import CustomUpperConfidenceBound
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

    # Create an initial population.
    logging.info("Generating initial population.")

    initial_genotypes = [
        Genotype.initialize(
            rng=rng,
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

        offspring_genotypes = selection.generate_offspring(rng, population)

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
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=config.NUM_PARALLEL_PROCESSES
    ) as executor:
        futures = [
            executor.submit(learn_genotype, genotype, evaluator, rng)
            for genotype in genotypes
        ]
    result_objective_values = []
    genotypes = []
    for future in futures:

        objective_value, learn_individuals = future.result()
        result_objective_values.append(objective_value)
        genotypes.append(learn_individuals[0].morphology_genotype)

        for learn_individual in learn_individuals:
            with Session(dbengine, expire_on_commit=False) as session:
                session.add(learn_individual)
                session.commit()
    return result_objective_values, genotypes


def learn_genotype(genotype, evaluator, rng):
    # We get the brain uuids from the developed body, because if it is too big we don't want to learn unused uuids
    developed_body = genotype.develop_body()
    genotype.update_brain_parameters(developed_body, rng)
    brain_uuids = list(genotype.brain.keys())

    if len(brain_uuids) == 0:
        empty_learn_genotype = LearnGenotype(brain=genotype.brain)
        return 0, [LearnIndividual(morphology_genotype=genotype, genotype=empty_learn_genotype, objective_value=0, generation_index=0)]

    best_objective_value = None
    learn_individuals = []
    inherited_experience = [experience for experience in genotype.inherited_experience if experience[2] == 1]
    sorted_inherited_experience = genotype.update_values_with_genotype(
        sorted(inherited_experience, key=lambda x: x[1], reverse=True))
    alphas = np.array([])

    if len(sorted_inherited_experience) > 0 and config.BONUS:
        coefficients, intercept, mean, std = get_prior_information(genotype, rng, sorted_inherited_experience)
        optimizer = CustomBayesianOptimization(
            f=None,
            pbounds=genotype.get_p_bounds(),
            allow_duplicate_points=True,
            random_state=int(rng.integers(low=0, high=2 ** 32)),
            acquisition_function=acquisition.UpperConfidenceBound(kappa=3,
                                                                  random_state=rng.integers(low=0, high=2 ** 32)),
            coefficients=coefficients,
            intercept=intercept,
            old_mean=mean,
            old_std=std
        )
        empty_sample = {}
        for key in sorted_inherited_experience[0][0]:
            empty_sample[key] = 0
        alphas = np.append(alphas, config.ALPHA)
        optimizer.register(params=empty_sample, target=0)
        optimizer.set_gp_params(alpha=alphas)
    else:
        optimizer = BayesianOptimization(
            f=None,
            pbounds=genotype.get_p_bounds(),
            allow_duplicate_points=True,
            random_state=int(rng.integers(low=0, high=2 ** 32)),
            acquisition_function=acquisition.UpperConfidenceBound(kappa=3,
                                                                  random_state=rng.integers(low=0, high=2 ** 32))
        )
        optimizer.set_gp_params(alpha=[])

    optimizer.set_gp_params(kernel=Matern(nu=config.NU, length_scale=config.LENGTH_SCALE, length_scale_bounds="fixed"))
    if config.INHERIT_SAMPLES and config.NUM_REDO_INHERITED_SAMPLES == 0:
        i = 0
        for inherited_experience_sample, objective_value, inheritance_number in sorted_inherited_experience:
            if i < config.NUM_REDO_INHERITED_SAMPLES:
                continue
            alphas = np.append(alphas, config.INHERITED_ALPHA)
            optimizer.register(params=inherited_experience_sample, target=objective_value)
            optimizer.set_gp_params(alpha=alphas)

    for i in range(config.LEARN_NUM_GENERATIONS + config.NUM_REDO_INHERITED_SAMPLES):
        logging.info(f"Learn generation {i + 1} / {config.LEARN_NUM_GENERATIONS + config.NUM_REDO_INHERITED_SAMPLES}.")
        if i < config.NUM_REDO_INHERITED_SAMPLES and len(sorted_inherited_experience) > 0:
            next_point = sorted_inherited_experience[i][0]
        elif config.EVOLUTIONARY_SEARCH and i == 0:
            next_point = genotype.get_evolutionary_search_next_point()
            next_point = dict(sorted(next_point.items()))
        else:
            next_point = optimizer.suggest()
            next_point = dict(sorted(next_point.items()))

        new_learn_genotype = LearnGenotype(brain={})
        new_learn_genotype.next_point_to_brain(next_point, list(genotype.brain.keys()))
        robot = new_learn_genotype.develop(developed_body)

        # Evaluate them.
        start_time = time.time()
        objective_value = evaluator.evaluate(robot)
        end_time = time.time()
        new_learn_genotype.execution_time = end_time - start_time

        if best_objective_value is None or objective_value >= best_objective_value:
            best_objective_value = objective_value

        alphas = np.append(alphas, config.ALPHA)
        optimizer.register(params=next_point, target=objective_value)
        optimizer.set_gp_params(alpha=alphas)
        genotype.experience.append((next_point, objective_value))

        learn_individual = LearnIndividual(morphology_genotype=genotype, genotype=new_learn_genotype, objective_value=objective_value, generation_index=i)
        learn_individuals.append(learn_individual)

    return best_objective_value, learn_individuals

def get_prior_information(genotype, rng, sorted_inherited_experience):
    bonus_optimizer = BayesianOptimization(
        f=None,
        pbounds=genotype.get_p_bounds(),
        allow_duplicate_points=True,
        random_state=int(rng.integers(low=0, high=2 ** 32)),
        acquisition_function=acquisition.UpperConfidenceBound(kappa=3, random_state=rng.integers(low=0, high=2 ** 32))
    )
    bonus_optimizer.set_gp_params(alpha=0.1, kernel=Matern(nu=config.NU, length_scale=config.LENGTH_SCALE,
                                                           length_scale_bounds="fixed"))
    objective_values = []
    for inherited_experience_sample, objective_value, inheritance_number in sorted_inherited_experience:
        bonus_optimizer.register(params=inherited_experience_sample, target=objective_value)
        objective_values.append(objective_value)
    bonus_optimizer.suggest()
    poly = PolynomialFeatures(2)
    model = BayesianRidge()
    x_poly = poly.fit_transform(bonus_optimizer._gp.X_train_)
    model.fit(x_poly, bonus_optimizer._gp.y_train_)
    coefficients = model.coef_
    intercept = model.intercept_

    return coefficients, intercept, float(np.mean(objective_values)), float(np.std(objective_values))


def read_args():
    # Read args
    parser = ArgumentParser()
    parser.add_argument("--learn", required=True)
    parser.add_argument("--controllers", required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--repetition", required=True)
    parser.add_argument("--survivorselect", required=True)
    parser.add_argument("--parentselect", required=True)
    parser.add_argument("--inheritsamples", required=True)
    parser.add_argument("--grow", required=False)
    args = parser.parse_args()
    config.NUM_REDO_INHERITED_SAMPLES = int(args.inheritsamples)
    config.INHERIT_SAMPLES = True
    config.EVOLUTIONARY_SEARCH = False
    config.BONUS = False
    if config.NUM_REDO_INHERITED_SAMPLES == -1:
        config.INHERIT_SAMPLES = False
        config.NUM_REDO_INHERITED_SAMPLES = 0
        config.EVOLUTIONARY_SEARCH = True
    if config.NUM_REDO_INHERITED_SAMPLES == -2:
        config.INHERIT_SAMPLES = False
        config.NUM_REDO_INHERITED_SAMPLES = 0
        config.BONUS = True
    config.LEARN_NUM_GENERATIONS = int(args.learn) - config.NUM_REDO_INHERITED_SAMPLES
    config.CONTROLLERS = int(args.controllers)
    config.ENVIRONMENT = args.environment
    config.SURVIVOR_SELECT_STRATEGY = args.survivorselect
    config.PARENT_SELECT_STRATEGY = args.parentselect
    extra = ""
    if args.grow:
        config.INIT_MIN_MODULES = 5
        config.INIT_MAX_MODULES = 10
        extra += "_grow-1"
    controllers_string = 'adaptable' if config.CONTROLLERS == -1 else config.CONTROLLERS
    config.DATABASE_FILE = ("learn-" + str(args.learn) + "_controllers-" + str(controllers_string) + "_survivorselect-"
                            + args.survivorselect + "_parentselect-" + args.parentselect + "_inheritsamples-" + args.inheritsamples + "_environment-"
                            + args.environment + extra + "_" + str(args.repetition) + ".sqlite")


def main() -> None:
    """Run the program."""
    if config.READ_ARGS:
        read_args()
    config.NUM_GENERATIONS = (int((config.FUNCTION_EVALUATIONS / (int(config.LEARN_NUM_GENERATIONS + config.NUM_REDO_INHERITED_SAMPLES) * config.OFFSPRING_SIZE)))
                              - int(config.POPULATION_SIZE / config.OFFSPRING_SIZE))

    # Set up logging.
    setup_logging(file_name="log.txt")

    # Open the database, only if it does not already exists.
    dbengine = open_database_sqlite(
        config.DATABASE_FILE, open_method=OpenMethod.NOT_EXISTS_AND_CREATE
    )
    # Create the structure of the database.
    Base.metadata.create_all(dbengine)

    # Run the experiment several times.
    for _ in range(config.NUM_REPETITIONS):
        run_experiment(dbengine)


if __name__ == "__main__":
    main()
