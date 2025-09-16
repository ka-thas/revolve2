import os

from database_components.experiment import Experiment
from database_components.generation import Generation
from database_components.individual import Individual
from database_components.genotype import Genotype
from database_components.population import Population

import pandas
import matplotlib.pyplot as plt

from revolve2.experimentation.database import open_database_sqlite, OpenMethod
from sqlalchemy import select


def categorize_mutation(mutation_parameter):
    if mutation_parameter < 0.45:
        return 'Add'
    elif 0.33 <= mutation_parameter < 0.9:
        return 'Remove'
    else:
        return 'Reverse'


def get_df(learn, evosearch, controllers, environment, survivor_select, folder, popsize):
    database_name = f"learn-{learn}_evosearch-{evosearch}_controllers-{controllers}_survivorselect-{survivor_select}_parentselect-tournament_environment-{environment}"
    print(database_name)
    files = [file for file in os.listdir(folder) if file.startswith(database_name)]
    if len(files) == 0:
        return None
    dfs = []
    i = 1
    for file_name in files:
        dbengine = open_database_sqlite(folder + "/" + file_name, open_method=OpenMethod.OPEN_IF_EXISTS)

        df_mini = pandas.read_sql(
            select(
                Experiment.id.label("experiment_id"),
                Generation.generation_index,
                Genotype.id.label("genotype_id"),
                Genotype.parent_1_genotype_id,
                Genotype.parent_2_genotype_id,
                Genotype.mutation_parameter,
                Individual.objective_value.label("fitness")
            )
            .join_from(Experiment, Generation, Experiment.id == Generation.experiment_id)
            .join_from(Generation, Population, Generation.population_id == Population.id)
            .join_from(Population, Individual, Population.id == Individual.population_id)
            .join_from(Individual, Genotype, Individual.genotype_id == Genotype.id),
            dbengine,
        )
        df_mini = df_mini[df_mini['experiment_id'] == 1]
        df_mini['experiment_id'] = i

        df_mini['function_evaluations'] = df_mini['generation_index'] * int(learn) * popsize + int(learn) * popsize
        dfs.append(df_mini)
        i += 1
    return pandas.concat(dfs)


def lets_go_mutation_plot_exclamation_mark(df, ax_thingy):
    mutation_counts = df.groupby(['function_evaluations', 'mutation']).size().reset_index(name='count')
    total_counts = mutation_counts.groupby('function_evaluations')['count'].sum().reset_index()
    mutation_counts = pandas.merge(mutation_counts, total_counts, on='function_evaluations', suffixes=('', '_total'))
    mutation_counts['proportion'] = mutation_counts['count'] / mutation_counts['count_total']
    mutation_proportions_pivot = mutation_counts.pivot_table(index='function_evaluations', columns='mutation',
                                                             values='proportion', fill_value=0)
    mutation_proportions_pivot.reset_index(inplace=True)

    ax_thingy.plot(mutation_proportions_pivot['function_evaluations'], mutation_proportions_pivot['Add'])
    ax_thingy.plot(mutation_proportions_pivot['function_evaluations'], mutation_proportions_pivot['Remove'])
    ax_thingy.plot(mutation_proportions_pivot['function_evaluations'], mutation_proportions_pivot['Reverse'])
    ax_thingy.axhline(y=0.45, color='black', linestyle='--')
    ax_thingy.axhline(y=0.1, color='black', linestyle='--')
    ax_thingy.axis(ymin=0, ymax=1)


def plot_mutation(df, ax_thingy):
    df = df.copy()
    df = df[df['mutation_parameter'] > 0]
    df['mutation'] = df['mutation_parameter'].apply(categorize_mutation)
    df.drop(columns=['mutation_parameter'], inplace=True)

    lets_go_mutation_plot_exclamation_mark(df, ax_thingy)


def main() -> None:
    fig, ax = plt.subplots(nrows=2, ncols=4, sharex='col', sharey='row')
    for i, (folder, popsize) in enumerate([("./results/1609", 50), ("./results/1709", 50), ("./results/1709_2", 200), ("./results/1909", 100)]):
        for j, learn in enumerate(['1', '30']):
            df = get_df(learn, '1', 'adaptable', 'noisy', 'newest', folder, popsize)
            if df is None:
                continue
            plot_mutation(df, ax[j][i])
    ax[0][0].legend(['Add', 'Remove', 'Reverse'], loc='upper left')
    plt.show()


if __name__ == "__main__":
    main()
