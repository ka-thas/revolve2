import concurrent.futures

import random
from argparse import ArgumentParser

import pandas as pd
from bayes_opt import BayesianOptimization, acquisition
from sklearn.gaussian_process.kernels import Matern

import config
from database_components.generation import Generation
from database_components.genotype import Genotype
from database_components.individual import Individual
from database_components.learn_genotype import LearnGenotype
from database_components.population import Population
from evaluator import Evaluator
from revolve2.experimentation.database import open_database_sqlite, OpenMethod
from revolve2.experimentation.rng import seed_from_time, make_rng
from sqlalchemy import select
from sqlalchemy.orm import Session
import sys

def get_best_genotypes():
    genotypes = []
    for inherit_samples in ['-1', '5', '0']:
        for environment in ['flat', 'noisy', 'hills', 'steps']:
            for repetition in range(1, 21):
                if environment == 'noisy' and inherit_samples == '5' and repetition == 4:
                    continue
                database_name = f"learn-30_controllers-adaptable_survivorselect-newest_parentselect-tournament_inheritsamples-{inherit_samples}_environment-{environment}_{repetition}.sqlite"
                dbengine = open_database_sqlite(
                    'results/new_big/' + database_name, open_method=OpenMethod.OPEN_IF_EXISTS
                )

                with Session(dbengine) as ses:
                    genotype = ses.execute(
                        select(Genotype)
                        .join_from(Generation, Population, Generation.population_id == Population.id)
                        .join_from(Population, Individual, Population.id == Individual.population_id)
                        .join_from(Individual, Genotype, Individual.genotype_id == Genotype.id)
                        .where(Generation.generation_index <= 501)
                        .order_by(Individual.objective_value.desc())
                        .limit(1)
                    ).fetchone()
                genotypes.append(genotype[0])
    return genotypes

def main():
    parser = ArgumentParser()
    parser.add_argument("--kappa", required=True)
    parser.add_argument("--alpha", required=True)
    parser.add_argument("--length-scale", required=True)
    parser.add_argument("--do-random", required=True)
    args = parser.parse_args()

    print(f"Received Arguments: {args}")

    kappa = float(args.kappa)
    alpha = float(args.alpha)
    length_scale = float(args.length_scale)
    do_random = args.do_random == '1'

    # number_of_robots = 100
    number_of_iterations = 30
    environments = ['flat', 'noisy', 'hills', 'steps']

    rng_seed = 1111972312
    rng = make_rng(rng_seed)

    # random_genotypes = [
    #     Genotype.initialize(
    #         rng=rng,
    #     )
    #     for _ in range(number_of_robots)
    # ]

    best_genotypes = get_best_genotypes()

    evaluators = []
    for environment in environments:
        config.ENVIRONMENT = environment
        evaluators.append(Evaluator(headless=True, num_simulators=1))

    with concurrent.futures.ProcessPoolExecutor(
            max_workers=24
    ) as executor:
        futures = []
        robot_id = 1
        for genotype in best_genotypes:
            futures.append(executor.submit(test_robot, genotype, evaluators, number_of_iterations, robot_id, rng, kappa, alpha, length_scale, do_random))
            robot_id += 1

    dfs = []
    for future in futures:
        df, robot_id = future.result()
        df['robot_id'] = robot_id
        dfs.append(df)
    result_df = pd.concat(dfs)
    result_df['rng'] = rng_seed
    result_df.to_csv(f"best_robots_learn_{kappa}_{alpha}_{length_scale}_{do_random}.csv", index=False)

def test_robot(genotype, evaluators, number_of_iterations, robot_id, rng, kappa, alpha, length_scale, do_random):
    print(f"Robot {robot_id} started", flush=True)
    sys.stdout.flush()
    optimizers = []
    for _ in evaluators:
        optimizer = BayesianOptimization(
            f=None,
            pbounds=genotype.get_p_bounds(),
            allow_duplicate_points=True,
            random_state=int(rng.integers(low=0, high=2 ** 32)),
            acquisition_function=acquisition.UpperConfidenceBound(kappa=kappa,
                                                                  random_state=rng.integers(low=0, high=2 ** 32))
        )
        optimizer.set_gp_params(alpha=alpha)
        optimizer.set_gp_params(
            kernel=Matern(nu=5/2, length_scale=length_scale, length_scale_bounds="fixed"))
        optimizers.append(optimizer)

    result = []
    developed_body = genotype.develop_body()
    brain_uuids = list(genotype.brain.keys())
    for i in range(number_of_iterations):
        print(f"Robot {robot_id}, Iteration number: {i}", flush=True)
        sys.stdout.flush()
        objective_values = []
        for optimizer, evaluator in zip(optimizers, evaluators):
            if i < 5 and do_random:
                next_point = genotype.get_random_next_point(rng)
            else:
                next_point = optimizer.suggest()
            next_point = dict(sorted(next_point.items()))

            new_learn_genotype = LearnGenotype(brain={})
            new_learn_genotype.next_point_to_brain(next_point, brain_uuids)
            modular_robot = new_learn_genotype.develop(developed_body)

            objective_value = evaluator.evaluate(modular_robot)
            objective_values.append(objective_value)
            optimizer.register(params=next_point, target=objective_value)
        result.append(objective_values)
    print(f"Robot {robot_id} finished", flush=True)
    sys.stdout.flush()
    return pd.DataFrame(result, columns=['flat', 'noisy', 'hills', 'steps']), robot_id

if __name__ == '__main__':
    main()