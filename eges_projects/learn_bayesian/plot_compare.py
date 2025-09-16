"""Plot fitness over generations for all experiments, averaged."""

import matplotlib.pyplot as plt
import pandas
import os

from database_components.experiment import Experiment
from database_components.generation import Generation
from database_components.individual import Individual
from database_components.population import Population
from sqlalchemy import select

from revolve2.experimentation.database import OpenMethod, open_database_sqlite


def get_df(learn, survivor_select):
    until_generation_index = 2501 if learn == '1' else 167
    database_name = f"learn-{learn}_evosearch-1_controllers-adaptable_survivorselect-{survivor_select}_parentselect-tournament_environment-noisy"
    files = [file for file in os.listdir("../evolve_and_learn/results/2309") if file.startswith(database_name)]
    if len(files) == 0:
        return None
    dfs = []
    i = 1
    for file_name in files:
        dbengine = open_database_sqlite("../evolve_and_learn/results/2309/" + file_name, open_method=OpenMethod.OPEN_IF_EXISTS)

        df_mini = pandas.read_sql(
            select(
                Experiment.id.label("experiment_id"),
                Generation.generation_index,
                Individual.objective_value.label("fitness")
            )
            .join_from(Experiment, Generation, Experiment.id == Generation.experiment_id)
            .join_from(Generation, Population, Generation.population_id == Population.id)
            .join_from(Population, Individual, Population.id == Individual.population_id)
            .where(Generation.generation_index < until_generation_index),
            dbengine,
        )
        df_mini = df_mini[df_mini['experiment_id'] == 1]
        df_mini['experiment_id'] = i

        df_mini['function_evaluations'] = df_mini['generation_index'] * int(learn) * 200 + int(learn) * 200
        dfs.append(df_mini)
        i += 1
    return pandas.concat(dfs)

def get_extra_df(survivor_select):
    max_records = []
    for i in range(12):
        dfs = []
        experiment = i + 1
        database_name = f"learn-1_evosearch-1_controllers-adaptable_survivorselect-{survivor_select}_parentselect-tournament_environment-noisy_{experiment}"
        files = [file for file in os.listdir("results/after_learn_2309") if file.startswith(database_name)]
        if len(files) == 0:
            continue
        for file_name in files:
            dbengine = open_database_sqlite("results/after_learn_2309/" + file_name, open_method=OpenMethod.OPEN_IF_EXISTS)

            df_mini = pandas.read_sql(
                select(
                    Experiment.id.label("experiment_id"),
                    Generation.generation_index,
                    Individual.fitness
                )
                .join_from(Experiment, Generation, Experiment.id == Generation.experiment_id)
                .join_from(Generation, Population, Generation.population_id == Population.id)
                .join_from(Population, Individual, Population.id == Individual.population_id),
                dbengine,
            )
            dfs.append(df_mini)
        result = pandas.concat(dfs)
        result['experiment_id'] = experiment
        max_records.append(result[result['fitness'] == result['fitness'].max()])
    result_df = pandas.concat(max_records)
    result_df['generation_index'] = 166
    result_df['function_evaluations'] = result_df['generation_index'] * 200 + 200
    return result_df


def plot_database(ax_thingy, x_axis, learn, survivor_select, remove_worst_x):
    max_or_mean = 'max'
    df = get_df(learn, survivor_select)

    if learn == '1':
        extra_df = get_extra_df(survivor_select)
        df = pandas.concat([df, extra_df])

    if df is None:
        return

    agg_per_experiment_per_generation = (
        df.groupby(["experiment_id", x_axis])
        .agg({"fitness": ["max", "mean"]})
        .reset_index()
    )
    agg_per_experiment_per_generation.columns = [
        "experiment_id",
        x_axis,
        "max_fitness",
        "mean_fitness",
    ]

    # gen_165 = agg_per_experiment_per_generation[agg_per_experiment_per_generation['generation_index'] == 165].set_index('experiment_id')['max_fitness']
    # gen_166 = agg_per_experiment_per_generation[agg_per_experiment_per_generation['generation_index'] == 166].set_index('experiment_id')['max_fitness']
    # max_fitness_165_166 = gen_165.combine(gen_166, max)
    # agg_per_experiment_per_generation.loc[agg_per_experiment_per_generation['generation_index'] == 166, 'max_fitness'] = agg_per_experiment_per_generation['experiment_id'].map(max_fitness_165_166)

    gen_166_df = agg_per_experiment_per_generation[agg_per_experiment_per_generation['generation_index'] == 166]
    print(gen_166_df)
    lowest_fitness_ids = gen_166_df.nsmallest(remove_worst_x, 'max_fitness')['experiment_id'].unique()
    agg_per_experiment_per_generation = agg_per_experiment_per_generation[~agg_per_experiment_per_generation['experiment_id'].isin(lowest_fitness_ids)]

    agg_per_experiment_per_generation['max_fitness'] = \
    agg_per_experiment_per_generation.groupby(['experiment_id', x_axis])['max_fitness'].transform('max').groupby(
    agg_per_experiment_per_generation['experiment_id']).cummax()

    agg_per_generation = (
        agg_per_experiment_per_generation.groupby(x_axis)
        .agg({"max_fitness": ["mean", "std"], "mean_fitness": ["mean", "std"]})
        .reset_index()
    )
    agg_per_generation.columns = [
        x_axis,
        "max_fitness_mean",
        "max_fitness_std",
        "mean_fitness_mean",
        "mean_fitness_std",
    ]

    learn_to_color = {
        '1': 'red',
        '30': 'blue',
        '50': 'red',
    }

    learn_to_label = {
        '1': 'Evolution only',
        '30': 'Evolution with learning',
        '50': 'Learn budget: 50',
    }

    ax_thingy.plot(
        agg_per_generation[x_axis],
        agg_per_generation[max_or_mean + "_fitness_mean"],
        linewidth=2,
        color=learn_to_color[learn],
        label=learn_to_label[learn],
    )
    ax_thingy.fill_between(
        agg_per_generation[x_axis],
        agg_per_generation[max_or_mean + "_fitness_mean"]
        - agg_per_generation[max_or_mean + "_fitness_std"],
        agg_per_generation[max_or_mean + "_fitness_mean"]
        + agg_per_generation[max_or_mean + "_fitness_std"],
        alpha=0.1,
        color=learn_to_color[learn],
    )

    ax_thingy.set_xlim(0, 170)


def main() -> None:
    fig, ax = plt.subplots(nrows=2, ncols=2, sharex='col', sharey='row')
    for learn in ['1']:
        for i, survivor_select in enumerate(['best', 'newest']):
            for j, remove_thingy in enumerate([0]):
                plot_database(ax[i][j], 'generation_index', learn, survivor_select, remove_thingy)

    fig.text(0.08, 0.3, 'Survivor selection: Generational', va='center', rotation='vertical', fontsize=16)
    fig.text(0.08, 0.7, 'Survivor selection: Elitist', va='center', rotation='vertical', fontsize=16)
    fig.text(0.45, 0.07, 'Generations', va='center', rotation='horizontal', fontsize=16)

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    #plt.show()


if __name__ == "__main__":
    main()
