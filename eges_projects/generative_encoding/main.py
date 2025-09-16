"""Main script for the example."""
import concurrent.futures
import logging

import numpy as np
import numpy.typing as npt
from bayes_opt import BayesianOptimization, UtilityFunction
from sklearn.gaussian_process.kernels import Matern
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

import config
from database_components.base import Base
from database_components.experiment import Experiment
from database_components.generation import Generation
from database_components.genotype import Genotype
from database_components.individual import Individual
from database_components.learn_generation import LearnGeneration
from database_components.learn_genotype import LearnGenotype
from database_components.learn_individual import LearnIndividual
from database_components.learn_population import LearnPopulation
from database_components.population import Population
from evaluator import Evaluator
from revolve2.experimentation.database import OpenMethod, open_database_sqlite
from revolve2.experimentation.logging import setup_logging
from revolve2.experimentation.optimization.ea import population_management, selection
from revolve2.experimentation.rng import make_rng, seed_from_time
from revolve2.modular_robot.body.base import ActiveHinge


def select_parent(
        rng: np.random.Generator,
        population: Population,
        offspring_size: int,
) -> npt.NDArray[np.float_]:
    """
    Select pairs of parents using a tournament.

    :param rng: Random number generator.
    :param population: The population to select from.
    :param offspring_size: The number of parent pairs to select.
    :returns: Pairs of indices of selected parents. offspring_size x 2 ints.
    """
    return np.array(
        [
            selection.multiple_unique(
                1,
                [individual.genotype for individual in population.individuals],
                [individual.fitness for individual in population.individuals],
                lambda _, fitnesses: selection.tournament(rng, fitnesses, k=4),
            )
            for _ in range(int(offspring_size))
        ],
    )


def select_survivors_tournament(
    rng: np.random.Generator,
    original_population: Population,
    offspring_population: Population,
) -> Population:
    """
    Select survivors using a tournament.

    :param rng: Random number generator.
    :param original_population: The population the parents come from.
    :param offspring_population: The offspring.
    :returns: A newly created population.
    """
    original_survivors, offspring_survivors = population_management.steady_state(
        [i.genotype for i in original_population.individuals],
        [i.fitness for i in original_population.individuals],
        [i.genotype for i in offspring_population.individuals],
        [i.fitness for i in offspring_population.individuals],
        lambda n, genotypes, fitnesses: selection.multiple_unique(
            n,
            genotypes,
            fitnesses,
            lambda _, fitnesses: selection.tournament(rng, fitnesses, k=4),
        ),
    )

    return Population(
        individuals=[
            Individual(
                genotype=original_population.individuals[i].genotype,
                fitness=original_population.individuals[i].fitness,
                original_generation=original_population.individuals[i].original_generation
            )
            for i in original_survivors
        ]
        + [
            Individual(
                genotype=offspring_population.individuals[i].genotype,
                fitness=offspring_population.individuals[i].fitness,
                original_generation=original_population.individuals[i].original_generation
            )
            for i in offspring_survivors
        ]
    )


def find_best_robot(
        current_best: Individual | None, population: list[Individual]
) -> Individual:
    """
    Return the best robot between the population and the current best individual.

    :param current_best: The current best individual.
    :param population: The population.
    :returns: The best individual.
    """
    return max(
        population + [] if current_best is None else [current_best],
        key=lambda x: x.fitness,
    )


def run_experiment(dbengine: Engine) -> None:
    """
    Run an experiment.

    :param dbengine: An openened database with matching initialize database structure.
    """
    logging.info("----------------")
    logging.info("Start experiment")

    # Set up the random number generator.
    rng_seed = seed_from_time() % 2 ** 32
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

    initial_fitnesses, initial_genotypes = learn_population(genotypes=initial_genotypes, evaluator=evaluator, dbengine=dbengine, rng=rng)

    # Create a population of individuals, combining genotype with fitness.
    individuals = []
    for genotype, fitness in zip(initial_genotypes, initial_fitnesses):
        individual = Individual(genotype=genotype, fitness=fitness, original_generation=0)
        individuals.append(individual)
    population = Population(
        individuals=individuals
    )
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
        parents = select_parent(rng, population, config.OFFSPRING_SIZE)
        offspring_genotypes = []
        for [parent_i] in parents:
            child_genotype = population.individuals[parent_i].genotype.mutate(rng)
            child_genotype.parent_1_genotype_id = population.individuals[parent_i].genotype.id
            offspring_genotypes.append(child_genotype)

        # Evaluate the offspring.
        offspring_fitnesses, offspring_genotypes = learn_population(genotypes=offspring_genotypes, evaluator=evaluator, dbengine=dbengine, rng=rng)

        # Make an intermediate offspring population.
        offspring_individuals = [
            Individual(genotype=genotype, fitness=fitness, original_generation=generation.generation_index + 1) for
            genotype, fitness in zip(offspring_genotypes, offspring_fitnesses)]
        # Create the next population by selecting survivors.
        population = select_survivors_tournament(
            rng,
            population,
            Population(
                individuals=offspring_individuals
            ),
        )
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
    result_fitnesses = []
    genotypes = []
    for future in futures:

        fitness, learn_generations = future.result()
        result_fitnesses.append(fitness)
        genotypes.append(learn_generations[0].genotype)

        for learn_generation in learn_generations:
            with Session(dbengine, expire_on_commit=False) as session:
                session.add(learn_generation)
                session.commit()
    return result_fitnesses, genotypes


def learn_genotype(genotype, evaluator, rng):
    developed_body = genotype.body.develop_body()
    brain_uuids = set()
    for active_hinge in developed_body.find_modules_of_type(ActiveHinge):
        brain_uuids.add(active_hinge.map_uuid)
    brain_uuids = list(brain_uuids)

    if len(brain_uuids) == 0:
        empty_learn_genotype = LearnGenotype(brain=genotype.brain, body=genotype.body)
        population = LearnPopulation(
            individuals=[
                LearnIndividual(genotype=empty_learn_genotype, fitness=0)
            ]
        )
        return 0, [LearnGeneration(
            genotype=genotype,
            generation_index=0,
            learn_population=population,
        )]

    pbounds = {}
    for key in brain_uuids:
        pbounds['amplitude_' + str(key)] = [0, 1]
        pbounds['phase_' + str(key)] = [0, 1]

    optimizer = BayesianOptimization(
        f=None,
        pbounds=pbounds,
        allow_duplicate_points=True,
        random_state=int(rng.integers(low=0, high=2**32))
    )
    optimizer.set_gp_params(alpha=config.ALPHA, kernel=Matern(nu=config.NU, length_scale=config.LENGTH_SCALE, length_scale_bounds=(config.LENGTH_SCALE - 0.01, config.LENGTH_SCALE + 0.01)))
    utility = UtilityFunction(kind="ucb", kappa=config.KAPPA)

    best_fitness = None
    learn_generations = []
    best_point = {}
    for i in range(config.LEARN_NUM_GENERATIONS + config.NUM_RANDOM_SAMPLES):
        logging.info(f"Learn generation {i + 1} / {config.LEARN_NUM_GENERATIONS + config.NUM_RANDOM_SAMPLES}.")
        if i < config.NUM_RANDOM_SAMPLES:
            next_point = {}
            for key in brain_uuids:
                next_point['amplitude_' + str(key)] = genotype.brain[key][0]
                next_point['phase_' + str(key)] = genotype.brain[key][1]
        else:
            next_point = optimizer.suggest(utility)
            next_point = dict(sorted(next_point.items()))
            next_best = utility.utility([list(next_point.values())], optimizer._gp, 0)
            for _ in range(10000):
                possible_point = {}
                for key in best_point.keys():
                    possible_point[key] = best_point[key] + np.random.normal(0, config.NEIGHBOUR_SCALE)
                possible_point = dict(sorted(possible_point.items()))

                utility_value = utility.utility([list(possible_point.values())], optimizer._gp, 0)
                if utility_value > next_best:
                    next_best = utility_value
                    next_point = possible_point

        new_learn_genotype = LearnGenotype(brain={}, body=genotype.body)
        for brain_uuid in brain_uuids:
            new_learn_genotype.brain[brain_uuid] = np.array(
                [
                    next_point['amplitude_' + str(brain_uuid)],
                    next_point['phase_' + str(brain_uuid)],
                ]
            )
        robot = new_learn_genotype.develop()

        # Evaluate them.
        fitness = evaluator.evaluate(robot)

        if best_fitness is None or fitness >= best_fitness:
            best_fitness = fitness
            best_point = next_point

        optimizer.register(params=next_point, target=fitness)

        # From the samples and fitnesses, create a population that we can save.
        population = LearnPopulation(
            individuals=[
                LearnIndividual(genotype=new_learn_genotype, fitness=fitness)
            ]
        )

        # Make it all into a generation and save it to the database.
        learn_generation = LearnGeneration(
            genotype=genotype,
            generation_index=i,
            learn_population=population,
        )
        learn_generations.append(learn_generation)

    return best_fitness, learn_generations


def main() -> None:
    """Run the program."""
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
