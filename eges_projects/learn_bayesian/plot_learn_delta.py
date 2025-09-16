import numpy as np
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


def main():
    fig, ax = plt.subplots(nrows=4, ncols=3, sharex='col', sharey='row')

    for j, old_environment in enumerate(['flat', 'steps', 'noisy']):
        for i, new_environment in enumerate(['flat', 'steps', 'noisy', 'hills']):
            for learn, color in zip([1, 30], ['red', 'blue']):
                df = get_df(learn, 1, 'adaptable', 'tournament', old_environment, new_environment)

                # Filter out the rows corresponding to the two parts globally before grouping
                random_part = df[df['generation_index'] < 30]
                last_part = df[df['generation_index'] >= 30]

                # Find the max fitness for each experiment_id in both parts
                random_part = random_part.groupby('experiment_id')['fitness'].max()
                last_part = last_part.groupby('experiment_id')['fitness'].max()

                # Calculate the learning deltas by subtracting random_part_max from last_part_max
                vp = ax[i][j].violinplot((last_part - random_part).tolist(), showmeans=True)

                # Change the color of each violin's body to red
                for body in vp['bodies']:
                    body.set_facecolor(color)
                    body.set_edgecolor(color)

                # Customize the appearance of cmins, cmaxes, cbars, and cmeans
                for part_name, line_width in zip(['cmins', 'cmaxes', 'cbars', 'cmeans'], [3, 3, 3, 1]):
                    vp[part_name].set_edgecolor(color)
                    vp[part_name].set_linewidth(line_width)

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