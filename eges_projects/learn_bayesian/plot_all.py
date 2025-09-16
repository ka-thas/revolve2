"""Plot fitness over generations for all experiments, averaged."""
import matplotlib.pyplot as plt
import pandas
import os
import re
from database_components.experiment import Experiment
from database_components.generation import Generation
from database_components.individual import Individual
from database_components.population import Population
from sqlalchemy import select

from revolve2.experimentation.database import OpenMethod, open_database_sqlite


def get_df(learn, evosearch, controllers, survivor_select, environment, new_environment):
    database_name = f"learn-{learn}_evosearch-{evosearch}_controllers-{controllers}_select-{survivor_select}_environment-{environment}.*{new_environment}.*"
    regex = re.compile(database_name)
    print(database_name)
    files = [file for file in os.listdir("results/after_learn_all") if regex.match(file)]
    if len(files) == 0:
        return None
    dfs = []
    i = 1
    for file_name in files:
        dbengine = open_database_sqlite("results/after_learn/" + file_name, open_method=OpenMethod.OPEN_IF_EXISTS)

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
        df_mini = df_mini[df_mini['experiment_id'] == 1]
        df_mini['experiment_id'] = i

        if df_mini.empty:
            print(file_name)

        dfs.append(df_mini)
        i += 1
    return pandas.concat(dfs)


def plot_database(ax_thingy, learn, old_environment, new_environment):
    max_or_mean = 'max'
    df = get_df(learn, 1, 'adaptable', 'tournament', old_environment, new_environment)

    if df is None:
        return

    agg_per_experiment_per_generation = (
        df.groupby(["experiment_id", "generation_index"])
        .agg({"fitness": ["max", "mean"]})
        .reset_index()
    )
    agg_per_experiment_per_generation.columns = [
        "experiment_id",
        "generation_index",
        "max_fitness",
        "mean_fitness",
    ]
    agg_per_generation = (
        agg_per_experiment_per_generation.groupby("generation_index")
        .agg({"max_fitness": ["mean", "std"], "mean_fitness": ["mean", "std"]})
        .reset_index()
    )
    agg_per_generation.columns = [
        "generation_index",
        "max_fitness_mean",
        "max_fitness_std",
        "mean_fitness_mean",
        "mean_fitness_std",
    ]

    learn_to_color = {
        1: 'purple',
        30: 'blue',
    }

    learn_to_label = {
        1: 'Learn budget: 1',
        30: 'Learn budget: 30',
    }

    ax_thingy.plot(
        agg_per_generation["generation_index"],
        agg_per_generation[max_or_mean + "_fitness_mean"],
        linewidth=2,
        color=learn_to_color[learn],
        label=learn_to_label[learn]
    )
    ax_thingy.fill_between(
        agg_per_generation["generation_index"],
        agg_per_generation[max_or_mean + "_fitness_mean"]
        - agg_per_generation[max_or_mean + "_fitness_std"],
        agg_per_generation[max_or_mean + "_fitness_mean"]
        + agg_per_generation[max_or_mean + "_fitness_std"],
        alpha=0.1,
        color=learn_to_color[learn],
    )


def main():
    fig, ax = plt.subplots(nrows=4, ncols=3, sharex='col', sharey='row')
    for learn in [1, 30]:
        for j, old_environment in enumerate(['flat', 'steps', 'noisy']):
            for i, new_environment in enumerate(['flat', 'steps', 'noisy', 'hills']):
                plot_database(ax[i][j], learn, old_environment, new_environment)
    ax[0][0].legend(loc='upper left', fontsize=10)

    fig.text(0.04, 0.5, 'Fitness', va='center', rotation='vertical', fontsize=16)

    fig.text(0.2, 0.9, 'Old environment: Flat', va='center', rotation='horizontal', fontsize=14)
    fig.text(0.48, 0.9, 'Old environment: Steps', va='center', rotation='horizontal', fontsize=14)
    fig.text(0.75, 0.9, 'Old environment: Noisy', va='center', rotation='horizontal', fontsize=14)

    fig.text(0.08, 0.2, 'New environment: Hills', va='center', rotation='vertical', fontsize=16)
    fig.text(0.08, 0.4, 'New environment: Noisy', va='center', rotation='vertical', fontsize=14)
    fig.text(0.08, 0.6, 'New environment: Steps', va='center', rotation='vertical', fontsize=14)
    fig.text(0.08, 0.8, 'New environment: Flat', va='center', rotation='vertical', fontsize=14)

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()


if __name__ == "__main__":
    main()