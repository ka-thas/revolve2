import concurrent.futures
import json
import os
import random
from argparse import ArgumentParser

from bayes_opt import BayesianOptimization, acquisition
from sklearn.gaussian_process.kernels import Matern
from sqlalchemy import Column, Integer, Float, select
from sqlalchemy.orm import Session, declarative_base

from database_components.generation import Generation
from database_components.genotype import Genotype
from database_components.individual import Individual
from database_components.learn_genotype import LearnGenotype
from database_components.population import Population
from evaluator import Evaluator
import config

from genotypes.body_genotype_direct import CoreGenotype, BodyDeveloper
from revolve2.experimentation.database import open_database_sqlite, OpenMethod

Base = declarative_base()

class RandomSample(Base):
    __tablename__ = 'random_sample'
    id = Column(Integer, primary_key=True, autoincrement=True)
    genotype_id = Column(Integer)
    repetition = Column(Integer)
    objective_value = Column(Float)
    learn = Column(Integer)

def main(inherit_samples, environment, repetition):
    iterations = 500
    robots = 20
    processes = 20

    config.ENVIRONMENT = environment
    evaluator = Evaluator(headless=True, num_simulators=1)
    database_name = f"learn-30_controllers-adaptable_survivorselect-newest_parentselect-tournament_inheritsamples-{inherit_samples}_environment-{environment}_{repetition}."
    files = [file for file in os.listdir('results/new_big') if file.startswith(database_name)]

    for file_name in files:
        # Load the best individual from the database.
        dbengine = open_database_sqlite(
            'results/new_big/' + file_name, open_method=OpenMethod.OPEN_IF_EXISTS
        )
        dbengine_write = open_database_sqlite(
            'results/random_long/' + file_name, open_method=OpenMethod.NOT_EXISTS_AND_CREATE
        )
        Base.metadata.create_all(dbengine_write)

        with Session(dbengine) as ses:
            genotypes = ses.execute(
                select(Genotype, Genotype.id)
                .join_from(Generation, Population, Generation.population_id == Population.id)
                .join_from(Population, Individual, Population.id == Individual.population_id)
                .join_from(Individual, Genotype, Individual.genotype_id == Genotype.id)
                .where(Generation.generation_index <= 501)
            ).fetchall()

        # Deduplicate manually, preserving order
        unique_genotypes = {}
        for genotype, genotype_id in genotypes:
            if genotype_id not in unique_genotypes:
                unique_genotypes[genotype_id] = genotype

        # Convert to a list of tuples
        genotypes = [(genotype_id, genotype) for
                     genotype_id, genotype in unique_genotypes.items()]

        genotypes = random.sample(genotypes, robots)

        # Random samples
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=processes
        ) as executor:
            futures = []
            for genotype_id, genotype in genotypes:
                futures.append(
                    executor.submit(sample, evaluator, genotype._serialized_body, genotype_id, iterations))

        for future in futures:
            samples, genotype_id = future.result()
            for (i, objective_value) in samples:
                new_data = RandomSample(genotype_id=genotype_id, repetition=i, objective_value=objective_value, learn=0)
                with Session(dbengine_write) as session:
                    session.add(new_data)
                    session.commit()

        # Learn
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=100
        ) as executor:
            futures = []
            for genotype_id, genotype in genotypes:
                futures.append(executor.submit(learn, evaluator, genotype, genotype_id, iterations))

        for future in futures:
            samples, genotype_id = future.result()
            for (i, objective_value) in samples:
                new_data = RandomSample(genotype_id=genotype_id, repetition=i, objective_value=objective_value, learn=1)
                with Session(dbengine_write) as session:
                    session.add(new_data)
                    session.commit()

def sample(evaluator, serialized_body, genotype_id, iterations):
    result = []

    body = CoreGenotype(0.0).deserialize(json.loads(serialized_body))
    body.reverse_phase_function(body.reverse_phase)
    body_developer = BodyDeveloper(body)
    body_developer.develop()
    brain_uuids = body.check_for_brains()

    for i in range(iterations):
        next_point = {}
        for key in brain_uuids:
            next_point['amplitude_' + str(key)] = random.random()
            next_point['phase_sin_' + str(key)] = random.random()
            next_point['phase_cos_' + str(key)] = random.random()
            next_point['offset_' + str(key)] = random.random()

        new_learn_genotype = LearnGenotype(brain={})
        new_learn_genotype.next_point_to_brain(next_point, brain_uuids)

        modular_robot = new_learn_genotype.develop(body_developer.body)
        objective_value = evaluator.evaluate(modular_robot)
        result.append((i, objective_value))
    return result, genotype_id

def learn(evaluator, genotype, genotype_id, iterations):
    result = []

    optimizer = BayesianOptimization(
        f=None,
        pbounds=genotype.get_p_bounds(),
        allow_duplicate_points=True,
        acquisition_function=acquisition.UpperConfidenceBound(kappa=config.KAPPA)
    )
    optimizer.set_gp_params(alpha=config.ALPHA)
    optimizer.set_gp_params(
        kernel=Matern(nu=5 / 2, length_scale=config.LENGTH_SCALE, length_scale_bounds="fixed"))
    developed_body = genotype.develop_body()
    brain_uuids = list(genotype.brain.keys())

    for i in range(iterations):
        next_point = optimizer.suggest()
        next_point = dict(sorted(next_point.items()))
        new_learn_genotype = LearnGenotype(brain={})
        new_learn_genotype.next_point_to_brain(next_point, brain_uuids)
        modular_robot = new_learn_genotype.develop(developed_body)
        objective_value = evaluator.evaluate(modular_robot)
        optimizer.register(params=next_point, target=objective_value)

        result.append((i, objective_value))
    return result, genotype_id

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--environment", required=True)
    parser.add_argument("--inheritsamples", required=True)
    parser.add_argument("--repetition", required=True)
    args = parser.parse_args()
    main(args.inheritsamples, args.environment, args.repetition)