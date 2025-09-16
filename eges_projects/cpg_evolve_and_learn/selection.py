import numpy as np
import numpy.typing as npt


import config
from database_components.genotype import Genotype
from database_components.individual import Individual
from database_components.population import Population

from revolve2.experimentation.optimization.ea import population_management, selection

def select_parents(
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
                2,
                [individual.genotype for individual in population.individuals],
                [individual.reproduction_fitness for individual in population.individuals],
                lambda _, fitnesses: selection.tournament(rng, fitnesses, k=config.PARENT_TOURNAMENT_SIZE),
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
        [i.survivor_fitness for i in original_population.individuals],
        [i.genotype for i in offspring_population.individuals],
        [i.survivor_fitness for i in offspring_population.individuals],
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
                objective_value=original_population.individuals[i].objective_value,
                original_generation=original_population.individuals[i].original_generation
            )
            for i in original_survivors
        ]
        + [
            Individual(
                genotype=offspring_population.individuals[i].genotype,
                fitness=offspring_population.individuals[i].fitness,
                objective_value=offspring_population.individuals[i].objective_value,
                original_generation=offspring_population.individuals[i].original_generation
            )
            for i in offspring_survivors
        ]
    )


def select_survivors_best(
        original_population: Population,
        offspring_population: Population,
) -> Population:
    individuals = []
    for individual in original_population.individuals:
        individuals.append(
            (individual.genotype, individual.objective_value, individual.survivor_fitness, individual.original_generation))
    for individual in offspring_population.individuals:
        individuals.append(
            (individual.genotype, individual.objective_value, individual.survivor_fitness, individual.original_generation))
    individuals = sorted(individuals, key=lambda x: (-x[2], -x[1]))
    survivors = individuals[:config.POPULATION_SIZE]

    return Population(
        individuals=[
            Individual(
                genotype=genotype,
                fitness=fitness,
                objective_value=objective_value,
                original_generation=original_generation
            )
            for (genotype, objective_value, fitness, original_generation) in survivors
        ]
    )


def generate_offspring(innov_db_body, rng, population):
    offspring_genotypes = []
    parents = select_parents(rng, population, config.OFFSPRING_SIZE)
    # Create offspring. One offspring per pair of parents, with a copy of one of its parents' distribution
    for parent1_i, parent2_i in parents:
        child_genotype_1 = Genotype.crossover(
            population.individuals[parent1_i].genotype,
            population.individuals[parent2_i].genotype,
            rng,
        )
        new_genotype = child_genotype_1.mutate(innov_db_body, rng)
        new_genotype.parent_1_genotype_id = population.individuals[parent1_i].genotype.id
        new_genotype.parent_2_genotype_id = population.individuals[parent2_i].genotype.id

        offspring_genotypes.append(new_genotype)
    return offspring_genotypes


def select_survivors(rng, original_population, offspring_population):
    strategy_to_function = {
        'tournament': lambda: select_survivors_tournament(rng, original_population, offspring_population),
        'best': lambda: select_survivors_best(original_population, offspring_population),
        'newest': lambda: select_survivors_best(original_population, offspring_population),
    }

    select_function = strategy_to_function.get(config.SURVIVOR_SELECT_STRATEGY)

    if select_function is None:
        raise ValueError(f"Unrecognized SURVIVOR_SELECT_STRATEGY: {config.SURVIVOR_SELECT_STRATEGY}")

    return select_function()


def calculate_fitness(individuals: list, strategy_type: str, fitness_attr: str):
    strategy_to_fitness = {
        'tournament': lambda ind: ind.objective_value,
        'best': lambda ind: ind.objective_value,
        'all': lambda ind: ind.objective_value,
        'newest': lambda ind: ind.original_generation,
    }

    fitness_calculator = strategy_to_fitness.get(strategy_type)

    if fitness_calculator is None:
        raise ValueError(f"Unrecognized strategy: {strategy_type}")

    for individual in individuals:
        fitness = fitness_calculator(individual)
        setattr(individual, fitness_attr, fitness)

def calculate_reproduction_fitness(population: Population):
    calculate_fitness(population.individuals, config.PARENT_SELECT_STRATEGY, 'reproduction_fitness')

def calculate_survival_fitness(original_population: Population, offspring_population: Population):
    combined_individuals = original_population.individuals + offspring_population.individuals
    calculate_fitness(combined_individuals, config.SURVIVOR_SELECT_STRATEGY, 'survivor_fitness')



