import os

import pandas
import matplotlib.pyplot as plt
import pandas as pd

from database_components.genotype import Genotype
from database_components.individual import Individual
from database_components.population import Population
from database_components.generation import Generation
from sqlalchemy import select

from revolve2.experimentation.database import OpenMethod, open_database_sqlite


def get_all_genotypes(learn, controllers, environment, survivor_select, folder, inherit_samples):
    database_name = f"learn-{learn}_controllers-{controllers}_survivorselect-{survivor_select}_parentselect-tournament_inheritsamples-{inherit_samples}_environment-{environment}"
    print(database_name)
    files = [file for file in os.listdir(folder) if file.startswith(database_name)]
    if len(files) == 0:
        return None
    dfs = []
    experiment_id = 1
    for file_name in files:
        dbengine = open_database_sqlite(folder + "/" + file_name, open_method=OpenMethod.OPEN_IF_EXISTS)

        df_mini = pandas.read_sql(
            select(
                Generation.generation_index,
                Genotype.id,
                Genotype.parent_1_genotype_id
            )
            .join_from(Generation, Population, Generation.population_id == Population.id)
            .join_from(Population, Individual, Population.id == Individual.population_id)
            .join_from(Individual, Genotype, Individual.genotype_id == Genotype.id),
            dbengine,
        )
        df_mini['experiment_id'] = experiment_id
        dfs.append(df_mini)
        experiment_id += 1
    return pandas.concat(dfs)


def get_origin_id(genotype_id, df_experiment):
    parent_id = list(df_experiment.loc[df_experiment['id'] == genotype_id]['parent_1_genotype_id'])[0]
    if parent_id == -1:
        return genotype_id
    return get_origin_id(parent_id, df_experiment)


def main():
    result = {
        'learn': [],
        'inherit_samples': [],
        'generation': [],
        'experiment_id': [],
        'average_number_of_origin_ids': [],
    }
    for learn in ['30']:
        for inherit_samples in ['-1', '5', '0', '-2']:
            df = get_all_genotypes(learn, 'adaptable', 'noisy', 'newest', './results/0301', inherit_samples)

            experiments = df['experiment_id'].nunique()
            for experiment_id in range(1, experiments + 1):
                print(experiment_id)
                df_experiment = df.loc[df['experiment_id'] == experiment_id]
                for generation_id in range(100):
                    df_generation = df_experiment.loc[df_experiment['generation_index'] == generation_id]
                    genotype_ids = list(df_generation['id'])
                    origin_ids = []
                    for genotype_id in genotype_ids:
                        origin_ids.append(get_origin_id(genotype_id, df_experiment))
                    result['learn'].append(learn)
                    result['generation'].append(generation_id)
                    result['experiment_id'].append(experiment_id)
                    result['average_number_of_origin_ids'].append(len(set(origin_ids)))
                    result['inherit_samples'].append(inherit_samples)
    pd.DataFrame(result).to_csv("results/first-parent-0301.csv", index=False)

def plot():
    df = pd.read_csv('./results/first-parent-0301.csv')
    fig, ax = plt.subplots(nrows=4, sharex='col', sharey='row')
    for i, inherit_samples in enumerate([-1, 5, 0, -2]):
        df_mini = df.loc[df['inherit_samples'] == inherit_samples]
        avg_origin = df_mini.groupby('generation')['average_number_of_origin_ids'].mean()

        print(avg_origin)

        avg_origin.plot(ax=ax[i])
    plt.show()


if __name__ == '__main__':
    main()
