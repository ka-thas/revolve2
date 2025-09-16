"""Plot fitness over generations for all experiments, averaged."""

import matplotlib.pyplot as plt
import pandas
import os

import pandas as pd
from matplotlib.ticker import ScalarFormatter

from database_components.experiment import Experiment
from database_components.generation import Generation
from database_components.individual import Individual
from database_components.population import Population
from sqlalchemy import select

from revolve2.experimentation.database import OpenMethod, open_database_sqlite

def get_df(learn, controllers, environment, survivor_select, folder, popsize, inherit_samples):
    database_name = f"learn-{learn}_controllers-{controllers}_survivorselect-{survivor_select}_parentselect-tournament_inheritsamples-{inherit_samples}_environment-{environment}"
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
                Individual.objective_value.label("fitness")
            )
            .join_from(Experiment, Generation, Experiment.id == Generation.experiment_id)
            .join_from(Generation, Population, Generation.population_id == Population.id)
            .join_from(Population, Individual, Population.id == Individual.population_id),
            dbengine,
        )
        df_mini = df_mini[df_mini['experiment_id'] == 1]
        df_mini['experiment_id'] = i

        df_mini['morphologies'] = (df_mini['generation_index']) * popsize + 200
        df_mini['function_evaluations'] = df_mini['generation_index'] * int(learn) * popsize + int(learn) * 200
        dfs.append(df_mini)
        i += 1
    return pandas.concat(dfs)


def plot_database(ax_thingy, x_axis, learn, environment, controllers, survivor_select, folder, popsize, inherit_samples):
    max_or_mean = 'max'
    df = get_df(learn, controllers, environment, survivor_select, folder, popsize, inherit_samples)

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

    agg_per_experiment_per_generation['max_fitness'] = agg_per_experiment_per_generation.groupby(['experiment_id', x_axis])['max_fitness'].transform('max').groupby(
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

    to_color = {
        '-1': 'red',
        '0': 'blue',
        '5': 'green',
        '-2': 'black',
        '3': 'yellow',
        '4': 'purple',
        '50': 'grey',
    }

    to_label = {
        '-1': 'No inheritance',
        '0': 'Inherit samples',
        '5': 'Redo samples',
        '-2': 'Inherit prior'
    }

    ax_thingy.plot(
        agg_per_generation[x_axis],
        agg_per_generation[max_or_mean + "_fitness_mean"],
        linewidth=2,
        color=to_color[inherit_samples],
        label=to_label[inherit_samples],
    )
    ax_thingy.fill_between(
        agg_per_generation[x_axis],
        agg_per_generation[max_or_mean + "_fitness_mean"]
        - agg_per_generation[max_or_mean + "_fitness_std"],
        agg_per_generation[max_or_mean + "_fitness_mean"]
        + agg_per_generation[max_or_mean + "_fitness_std"],
        color=to_color[inherit_samples],
        alpha=0.1,
    )

    ax_thingy.set_ylim(0, 12)


def main() -> None:
    fig, ax = plt.subplots()
    folder, popsize = ("./results/0301", 20)
    for i, survivor_select in enumerate(['newest']):
        for inherit_samples in ['-1', '5', '0', '-2']:
            plot_database(ax, 'generation_index', '30', 'noisy', 'adaptable', survivor_select, folder, popsize,
                          inherit_samples)
    folder, popsize = ("./results/0301/extra", 20)
    for i, survivor_select in enumerate(['newest']):
        for inherit_samples in ['-1', '5', '0', '-2']:
            plot_database(ax, 'generation_index', '30', 'noisy', 'adaptable', survivor_select, folder, popsize,
                          inherit_samples)

    ax.legend(loc='lower right', fontsize=10)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


if __name__ == "__main__":
    main()
