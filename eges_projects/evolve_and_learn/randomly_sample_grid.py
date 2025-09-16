import concurrent.futures
import json
import os
import random
from argparse import ArgumentParser

from bayes_opt import BayesianOptimization, acquisition
from sklearn.gaussian_process.kernels import Matern
from sqlalchemy import Column, Integer, Float, Boolean, select
from sqlalchemy.orm import Session, declarative_base

from database_components.generation import Generation
from database_components.genotype import Genotype
from database_components.individual import Individual
from database_components.learn_genotype import LearnGenotype
from database_components.population import Population
from evaluator import Evaluator

from genotypes.body_genotype_direct import CoreGenotype, BodyDeveloper
from revolve2.experimentation.database import open_database_sqlite, OpenMethod
from revolve2.experimentation.rng import seed_from_time, make_rng

Base = declarative_base()

class RandomSample(Base):
    __tablename__ = 'random_sample'
    id = Column(Integer, primary_key=True, autoincrement=True)
    genotype_id = Column(Integer)
    repetition = Column(Integer)
    objective_value = Column(Float)
    length_scale = Column(Float)
    alpha = Column(Float)
    kappa = Column(Float)
    do_random = Column(Boolean)

def main(inherit_samples, environment, repetition):
    evaluator = Evaluator(headless=True, num_simulators=1)
    database_name = f"learn-30_controllers-adaptable_survivorselect-newest_parentselect-tournament_inheritsamples-{inherit_samples}_environment-{environment}_{repetition}."
    files = [file for file in os.listdir('results/new_big') if file.startswith(database_name)]

    for file_name in files:
        # Load the best individual from the database.
        dbengine = open_database_sqlite(
            'results/new_big/' + file_name, open_method=OpenMethod.OPEN_IF_EXISTS
        )
        dbengine_write = open_database_sqlite(
            'results/random_grid/' + file_name, open_method=OpenMethod.NOT_EXISTS_AND_CREATE
        )
        Base.metadata.create_all(dbengine_write)

        with Session(dbengine) as ses:
            genotypes = ses.execute(
                select(Genotype, Genotype.id, Generation.generation_index, Individual.objective_value)
                .join_from(Generation, Population, Generation.population_id == Population.id)
                .join_from(Population, Individual, Population.id == Individual.population_id)
                .join_from(Individual, Genotype, Individual.genotype_id == Genotype.id)
                .where(Generation.generation_index <= 501)
            ).fetchall()

        # Deduplicate manually, preserving order
        unique_genotypes = {}
        for genotype, genotype_id, generation_index, objective_value in genotypes:
            if genotype_id not in unique_genotypes:
                unique_genotypes[genotype_id] = (genotype, generation_index, objective_value)

        # Convert to a list of tuples
        genotypes = [(genotype_id, genotype, generation_index, objective_value) for
                     genotype_id, (genotype, generation_index, objective_value) in unique_genotypes.items()]

        genotypes = sorted(genotypes, key=lambda x: x[3], reverse=True)[:10]

        with concurrent.futures.ProcessPoolExecutor(
                max_workers=100
        ) as executor:
            futures = []
            for length_scale in [0.1, 0.2, 0.4, 0.8]:
                for alpha in [1e-10, 1e-5, 1e-2, 1]:
                    for kappa in [1, 2, 3]:
                        for do_random in [True, False]:
                            for genotype_id, genotype, _, _ in genotypes:
                                futures.append(executor.submit(sample, evaluator, genotype, genotype_id, length_scale, alpha, kappa, do_random))

        for future in futures:
            samples, genotype_id, length_scale, alpha, kappa, do_random = future.result()
            for (i, objective_value) in samples:
                new_data = RandomSample(genotype_id=genotype_id, repetition=i, objective_value=objective_value, length_scale=length_scale, alpha=alpha, kappa=kappa, do_random=do_random)
                with Session(dbengine_write) as session:
                    session.add(new_data)
                    session.commit()

def sample(evaluator, genotype, genotype_id, length_scale, alpha, kappa, do_random):
    result = []

    rng_seed = seed_from_time() % 2 ** 32
    rng = make_rng(rng_seed)

    developed_body = genotype.develop_body()
    brain_uuids = list(genotype.brain.keys())

    optimizer = BayesianOptimization(
        f=None,
        pbounds=genotype.get_p_bounds(),
        allow_duplicate_points=True,
        acquisition_function=acquisition.UpperConfidenceBound(kappa=kappa)
    )
    optimizer.set_gp_params(alpha=alpha)
    optimizer.set_gp_params(
        kernel=Matern(nu=5 / 2, length_scale=length_scale, length_scale_bounds="fixed"))

    for i in range(30):
        if i < 5 and do_random:
            next_point = genotype.get_random_next_point(rng)
        else:
            next_point = optimizer.suggest()
        next_point = dict(sorted(next_point.items()))

        new_learn_genotype = LearnGenotype(brain={})
        new_learn_genotype.next_point_to_brain(next_point, brain_uuids)
        modular_robot = new_learn_genotype.develop(developed_body)

        objective_value = evaluator.evaluate(modular_robot)
        result.append((i, objective_value))
        optimizer.register(params=next_point, target=objective_value)

    return result, genotype_id, length_scale, alpha, kappa, do_random

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--environment", required=True)
    parser.add_argument("--inheritsamples", required=True)
    parser.add_argument("--repetition", required=True)
    args = parser.parse_args()
    main(args.inheritsamples, args.environment, args.repetition)