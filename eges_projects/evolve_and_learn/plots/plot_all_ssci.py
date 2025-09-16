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
                Individual.objective_value.label("fitness")
            )
            .join_from(Experiment, Generation, Experiment.id == Generation.experiment_id)
            .join_from(Generation, Population, Generation.population_id == Population.id)
            .join_from(Population, Individual, Population.id == Individual.population_id)
            .where(Generation.generation_index <= 2500),
            dbengine,
        )
        df_mini = df_mini[df_mini['experiment_id'] == 1]
        df_mini['experiment_id'] = i

        df_mini['morphologies'] = (df_mini['generation_index'] + 1) * popsize
        df_mini['function_evaluations'] = df_mini['generation_index'] * int(learn) * popsize + int(learn) * popsize
        dfs.append(df_mini)
        i += 1
    return pandas.concat(dfs)


def plot_database(ax_thingy, x_axis, learn, environment, controllers, evosearch, survivor_select, folder, popsize):
    max_or_mean = 'max'
    df = get_df(learn, evosearch, controllers, environment, survivor_select, folder, popsize)

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

    learn_to_color = {
        '1': 'red',
        '30': 'blue',
        '50': 'red',
        '100': 'green',
        '500': 'black',
    }

    learn_to_label = {
        '1': 'Evolution only',
        '30': 'Evolution with learning',
        '50': 'Learn budget: 50',
        '100': 'Learn budget: 100',
        '500': 'Learn budget: 500',
    }

    survivor_select_to_alpha = {
        'best': 1,
        'newest': 0.5
    }

    ax_thingy.plot(
        agg_per_generation[x_axis],
        agg_per_generation[max_or_mean + "_fitness_mean"],
        linewidth=2,
        color=learn_to_color[learn],
        label=learn_to_label[learn],
        # alpha=survivor_select_to_alpha[survivor_select],
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

    ax_thingy.set_ylim(0, 15)
    if x_axis == "function_evaluations":
        ax_thingy.set_xlim(0, 5e5)
    else:
        ax_thingy.set_xlim(0, 166)


def plot_ted(learn, survivor_select, x_axis, ax_thingy):
    if x_axis == 'generation_index':
        x_axis = 'generation'
    learn = int(learn)
    thingy = 'average_tree_edit_distance'
    df = pandas.read_csv("results/ssci/tree-edit-distance-2309-30best.csv", sep=",")
    df_2 = pandas.read_csv("results/ssci/tree-edit-distance-2309-30newest.csv", sep=",")
    df_3 = pandas.read_csv("results/ssci/tree-edit-distance-2309-1newest.csv", sep=",")
    df_4 = pandas.read_csv("results/ssci/tree-edit-distance-2309-1best.csv", sep=",")
    df = pd.concat([df, df_2, df_3, df_4], ignore_index=True)

    learn_to_color = {
        1: 'red',
        30: 'blue',
    }
    learn_to_label = {
        1: 'Evolution only',
        30: 'Evolution with learning',
    }
    current_df = df.loc[df['learn'] == learn]
    current_df = current_df.loc[current_df['survivor_select'] == survivor_select]
    current_df['function_evaluations'] = current_df['generation'] * int(learn) * 200 + int(learn) * 200

    agg_per_experiment_per_generation = (
        current_df.groupby(["experiment_id", x_axis])
        .agg({thingy: ["max", "mean"]})
        .reset_index()
    )
    agg_per_experiment_per_generation.columns = [
        "experiment_id",
        x_axis,
        "max_" + thingy,
        "mean_" + thingy,
    ]

    if learn == 1:
        print(list(agg_per_experiment_per_generation.loc[agg_per_experiment_per_generation[x_axis] == 166]["max_" + thingy]))
    else:
        print(list(agg_per_experiment_per_generation.loc[agg_per_experiment_per_generation[x_axis] == 166][
                       "max_" + thingy]))

    agg_per_generation = (
        agg_per_experiment_per_generation.groupby(x_axis)
        .agg({"max_" + thingy: ["mean", "std"], "mean_" + thingy: ["mean", "std"]})
        .reset_index()
    )
    agg_per_generation.columns = [
        x_axis,
        "max_" + thingy + "_mean",
        "max_" + thingy + "_std",
        "mean_" + thingy + "_mean",
        "mean_" + thingy + "_std",
    ]

    ax_thingy.plot(
        agg_per_generation[x_axis],
        agg_per_generation["max_" + thingy + "_mean"],
        linewidth=2,
        color=learn_to_color[learn],
        label=learn_to_label[learn]
    )
    ax_thingy.fill_between(
        agg_per_generation[x_axis],
        agg_per_generation["max_" + thingy + "_mean"]
        - agg_per_generation["max_" + thingy + "_std"],
        agg_per_generation["max_" + thingy + "_mean"]
        + agg_per_generation["max_" + thingy + "_std"],
        alpha=0.1,
        color=learn_to_color[learn]
    )
    ax_thingy.set_ylim(0, 15)
    if x_axis == "function_evaluations":
        ax_thingy.set_xlim(0, 5e5)
    else:
        ax_thingy.set_xlim(0, 166)

    # formatter = ScalarFormatter(useOffset=False)
    # formatter.set_powerlimits((0, 0))
    # ax_thingy.xaxis.set_major_formatter(formatter)

def main() -> None:
    fig, ax = plt.subplots(nrows=2, ncols=2, sharex='col', sharey='row')
    folder, popsize = ("./results/ssci", 200)
    for i, x_axis in enumerate(['generation_index']):
        for (learn, evosearch) in [('1', '1'), ('30', '1')]:
            for j, survivor_select in enumerate(['best', 'newest']):
                plot_database(ax[0][j], x_axis, learn, 'noisy', 'adaptable', evosearch, survivor_select, folder, popsize)
                plot_ted(learn, survivor_select, x_axis, ax[1][j])

    ax[0][0].annotate('8.81', xy=(166, 8.81), xytext=(120, 7),
                      arrowprops=dict(arrowstyle='->', lw=1, color='red'), color='red', fontsize=10)
    ax[0][1].annotate('8.75', xy=(166, 8.75), xytext=(120, 7),
                      arrowprops=dict(arrowstyle='->', lw=1, color='red'), color='red', fontsize=10)
    ax[0][0].annotate('12.77', xy=(166, 12.77), xytext=(120, 14),
                      arrowprops=dict(arrowstyle='->', lw=1, color='blue'), color='blue', fontsize=10)
    ax[0][1].annotate('11.81', xy=(166, 11.81), xytext=(120, 13),
                      arrowprops=dict(arrowstyle='->', lw=1, color='blue'), color='blue', fontsize=10)
    ax[1][0].annotate('6.01', xy=(166, 6.01), xytext=(120, 8),
                      arrowprops=dict(arrowstyle='->', lw=1, color='red'), color='red', fontsize=10)
    ax[1][1].annotate('7.43', xy=(166, 7.43), xytext=(120, 9),
                      arrowprops=dict(arrowstyle='->', lw=1, color='red'), color='red', fontsize=10)
    ax[1][0].annotate('3.87', xy=(166, 3.87), xytext=(120, 2),
                      arrowprops=dict(arrowstyle='->', lw=1, color='blue'), color='blue', fontsize=10)
    ax[1][1].annotate('6.62', xy=(166, 6.62), xytext=(120, 5),
                      arrowprops=dict(arrowstyle='->', lw=1, color='blue'), color='blue', fontsize=10)

    # ax[0][0].annotate('12.32', xy=(500000, 12.32), xytext=(400000, 10),
    #                   arrowprops=dict(arrowstyle='->', lw=1, color='red'), color='red', fontsize=10)
    # ax[0][1].annotate('10.86', xy=(500000, 10.86), xytext=(400000, 8),
    #                   arrowprops=dict(arrowstyle='->', lw=1, color='red'), color='red', fontsize=10)
    # ax[0][0].annotate('12.33', xy=(500000, 12.33), xytext=(400000, 14),
    #                   arrowprops=dict(arrowstyle='->', lw=1, color='blue'), color='blue', fontsize=10)
    # ax[0][1].annotate('11.38', xy=(500000, 11.38), xytext=(400000, 14),
    #                   arrowprops=dict(arrowstyle='->', lw=1, color='blue'), color='blue', fontsize=10)
    # ax[1][0].annotate('2.91', xy=(500000, 2.91), xytext=(400000, 1),
    #                   arrowprops=dict(arrowstyle='->', lw=1, color='red'), color='red', fontsize=10)
    # ax[1][1].annotate('6.9', xy=(500000, 6.9), xytext=(400000, 8.5),
    #                   arrowprops=dict(arrowstyle='->', lw=1, color='red'), color='red', fontsize=10)
    # ax[1][0].annotate('4.52', xy=(500000, 4.52), xytext=(400000, 5.5),
    #                   arrowprops=dict(arrowstyle='->', lw=1, color='blue'), color='blue', fontsize=10)
    # ax[1][1].annotate('6.33', xy=(500000, 6.33), xytext=(400000, 4.5),
    #                   arrowprops=dict(arrowstyle='->', lw=1, color='blue'), color='blue', fontsize=10)

    ax[0][0].legend(loc='lower left', fontsize=10)

    x_ticks = ax[1][1].get_xticks()
    y_ticks = ax[0][0].get_yticks()
    # ax[1][1].set_xticks(x_ticks[x_ticks != 0])
    ax[0][0].set_yticks(y_ticks[y_ticks != 0])

    fig.text(0.02, 0.3, 'Tree edit distance', va='center', rotation='vertical', fontsize=12)
    fig.text(0.02, 0.7, 'Objective value', va='center', rotation='vertical', fontsize=12)

    fig.text(0.15, 0.95, 'Survivor selection: Elitist', va='center', rotation='horizontal', fontsize=12)
    fig.text(0.5, 0.95, 'Survivor selection: Generational', va='center', rotation='horizontal', fontsize=12)
    fig.text(0.45, 0.01, 'Generations', va='center', rotation='horizontal', fontsize=12)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


if __name__ == "__main__":
    main()
