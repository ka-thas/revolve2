import pandas
import json
import os
from revolve2.experimentation.database import open_database_sqlite, OpenMethod
from sqlalchemy import select

from genotypes.body_genotype_direct import CoreGenotype
from database_components.experiment import Experiment
from database_components.generation import Generation
from database_components.genotype import Genotype
from database_components.individual import Individual
from database_components.population import Population

import matplotlib.pyplot as plt


def calculate_number_of_controllers(serialized_body):
    return len(CoreGenotype(0.0).deserialize(json.loads(serialized_body)).check_for_brains())


def calculate_number_of_modules(serialized_body):
    return CoreGenotype(0.0).deserialize(json.loads(serialized_body)).get_amount_modules()


def calculate_number_of_hinges(serialized_body):
    return CoreGenotype(0.0).deserialize(json.loads(serialized_body)).get_amount_hinges()


def get_df(learn, evosearch, controllers, environment, survivor_select):
    database_name = f"learn-{learn}_evosearch-{evosearch}_controllers-{controllers}_select-{survivor_select}_environment-{environment}"
    print(database_name)
    files = [file for file in os.listdir("results/3008") if file.startswith(database_name)]
    if len(files) == 0:
        return None
    dfs = []
    i = 1
    for file_name in files:
        dbengine = open_database_sqlite("results/3008/" + file_name, open_method=OpenMethod.OPEN_IF_EXISTS)

        df_mini = pandas.read_sql(
            select(
                Experiment.id.label("experiment_id"),
                Generation.generation_index,
                Individual.fitness,
                Genotype._serialized_body
            )
            .join_from(Experiment, Generation, Experiment.id == Generation.experiment_id)
            .join_from(Generation, Population, Generation.population_id == Population.id)
            .join_from(Population, Individual, Population.id == Individual.population_id)
            .join_from(Individual, Genotype, Individual.genotype_id == Genotype.id),
            dbengine,
        )
        df_mini = df_mini[df_mini['experiment_id'] == 1]
        df_mini['experiment_id'] = i

        df_mini['morphologies'] = df_mini['generation_index'] * 50 + 50
        df_mini['function_evaluations'] = df_mini['generation_index'] * int(learn) * 50 + int(learn) * 50
        df_mini['controllers'] = df_mini['serialized_body'].apply(lambda x: calculate_number_of_controllers(x))
        df_mini['modules'] = df_mini['serialized_body'].apply(lambda x: calculate_number_of_modules(x))
        df_mini['hinges'] = df_mini['serialized_body'].apply(lambda x: calculate_number_of_hinges(x))
        df_mini = df_mini.drop(columns=['serialized_body'])
        dfs.append(df_mini)
        i += 1
    return pandas.concat(dfs)


def make_plot(df):
    fig, ax = plt.subplots(nrows=3, ncols=3, sharex='col', sharey='row')
    x_axis = 'function_evaluations'
    learn_to_color = {
        '1': 'purple',
        '30': 'blue',
        '50': 'red'
    }
    learn_to_label = {
        '1': 'Learn budget: 1',
        '30': 'Learn budget: 30',
        '50': 'Learn budget: 50',
    }

    for i, environment in enumerate(['flat', 'steps', 'noisy']):
        for learn in ['1', '30']:
            for j, thingy in enumerate(['modules', 'hinges', 'controllers']):
                current_df = df.loc[df['environment'] == environment]
                current_df = current_df.loc[current_df['learn'] == learn]

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

                ax[j][i].plot(
                    agg_per_generation[x_axis],
                    agg_per_generation["mean_" + thingy + "_mean"],
                    linewidth=2,
                    color=learn_to_color[learn],
                    label=learn_to_label[learn],
                )
                ax[j][i].fill_between(
                    agg_per_generation[x_axis],
                    agg_per_generation["mean_" + thingy + "_mean"]
                    - agg_per_generation["mean_" + thingy + "_std"],
                    agg_per_generation["mean_" + thingy + "_mean"]
                    + agg_per_generation["mean_" + thingy + "_std"],
                    alpha=0.1,
                    color=learn_to_color[learn]
                )
                #ax[j][i].set_xlim(0, 7000)

    # Titles for each row
    row_titles = ['Modules', 'Hinges', 'Controllers']

    # Add a title to each row
    for ax_row, title in zip(ax, row_titles):
        # Set the title for the first subplot of each row
        ax_row[0].annotate(title, xy=(0.5, 1), xycoords='axes fraction', ha='center', va='bottom', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))

    ax[0][2].legend(loc='upper left', fontsize=10)
    fig.text(0.5, 0.04, 'Function Evaluations', ha='center')

    fig.text(0.2, 0.9, 'Flat', va='center', rotation='horizontal', fontsize=14)
    fig.text(0.48, 0.9, 'Steps', va='center', rotation='horizontal', fontsize=14)
    fig.text(0.75, 0.9, 'Noisy', va='center', rotation='horizontal', fontsize=14)

    plt.show()


def main() -> None:
    dfs = []
    for (learn, evosearch) in [('1', '1'), ('30', '1')]:
        for controllers in ['adaptable']:
            for environment in ['flat', 'steps', 'noisy']:
                current_result = get_df(learn, evosearch, controllers, environment, 'tournament')
                current_result['learn'] = learn
                current_result['expected_controllers'] = controllers
                current_result['environment'] = environment
                dfs.append(current_result)
    df = pandas.concat(dfs)
    df['learn'] = df['learn'].astype('category')
    df['expected_controllers'] = df['expected_controllers'].astype('category')
    df['environment'] = df['environment'].astype('category')
    # df.to_csv('results/robot-info-1208.csv', sep="\t")

    # df = pandas.read_csv('results/robot-info-1208.csv', sep="\t", index_col=0)
    # df['learn'] = df['learn'].astype('category')
    # df['expected_controllers'] = df['expected_controllers'].astype('category')
    # df['environment'] = df['environment'].astype('category')

    make_plot(df)


if __name__ == "__main__":
    main()
