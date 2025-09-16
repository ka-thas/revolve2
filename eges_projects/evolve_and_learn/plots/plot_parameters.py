import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import uuid

from database_components.learn_individual import LearnIndividual
from database_components.learn_genotype import LearnGenotype

import pandas

from revolve2.experimentation.database import open_database_sqlite, OpenMethod
from sqlalchemy import select


def deserialize_brain(serialized_brain):
    result = []
    for value in serialized_brain.split(';'):
        new_uuid, values = value.split(':')
        string_list = values.split(',')
        float_list = [float(value) for value in string_list]
        result.append(np.array(float_list))
    return result


def number_of_brains(serialized_brain):
    return len(serialized_brain.split(';'))


def get_df(learn, evosearch, controllers, environment, survivor_select):
    database_name = f"learn-{learn}_evosearch-{evosearch}_controllers-{controllers}_select-{survivor_select}_environment-{environment}"
    print(database_name)
    files = [file for file in os.listdir("results/1208") if file.startswith(database_name)]
    if len(files) == 0:
        return None
    dfs = []
    i = 1
    for file_name in files:
        dbengine = open_database_sqlite("results/1208/" + file_name, open_method=OpenMethod.OPEN_IF_EXISTS)
        df_mini = pandas.read_sql(
            select(
                LearnIndividual.fitness,
                LearnGenotype._serialized_brain
            )
            .join_from(LearnIndividual, LearnGenotype, LearnIndividual.genotype_id == LearnGenotype.id),
            dbengine
        )
        dfs.append(df_mini)
        i += 1
    return pandas.concat(dfs)


def to_correct_df(df):
    df['brain'] = df['serialized_brain'].apply(lambda x: deserialize_brain(x))
    df['number_of_brains'] = df['serialized_brain'].apply(lambda x: number_of_brains(x))
    flatten_brain = [item for sublist in df['brain'] for item in sublist]
    expanded_df = pd.DataFrame(flatten_brain, columns=['amplitude', 'phase', 'touch_weight', 'touch_phase_offset']).reset_index(drop=True)

    fitnesses = df['fitness'].repeat(df['number_of_brains']).reset_index(drop=False)
    expanded_df['fitness'] = fitnesses['fitness']
    expanded_df['id'] = fitnesses['index']
    df_sorted = expanded_df.sort_values(by='amplitude', ascending=False)
    filtered_df = df_sorted.drop_duplicates(subset='id', keep='first')
    filtered_df = filtered_df[filtered_df['fitness'] > 0]
    return filtered_df


def main() -> None:
    fig, ax = plt.subplots(nrows=4, ncols=3)
    dfs = []
    for i, environment in enumerate(['flat']):
        for j, (learn, evosearch) in enumerate([('1', '1'), ('30', '1')]):
            df = get_df(learn, evosearch, 'adaptable', environment, 'tournament')
            df = to_correct_df(df)
            # df.to_csv('results/' + str(learn) + "-" + environment + "-parameters.csv", sep="\t")
            # df = pandas.read_csv('results/' + str(learn) + "-" + environment + "-parameters.csv", sep="\t", index_col=0)
            sampled_df = df.sample(frac=1)
            dfs.append(sampled_df)
            sampled_df.sort_values('fitness', ascending=True, inplace=True)
            ax[i][j].scatter(sampled_df['amplitude'], sampled_df['phase'], c=sampled_df['fitness'], cmap='Greys')
            ax[i][j].title.set_text("Learn: " + learn + "-" + evosearch)
    # end_df = pd.concat(dfs)
    # ax[0].scatter(end_df['fitness'], end_df['amplitude'], alpha=0.1)
    # ax[1].scatter(end_df['fitness'], end_df['phase'], alpha=0.1)
    # ax[2].scatter(end_df['fitness'], end_df['touch_weight'], alpha=0.1)
    # ax[3].scatter(end_df['fitness'], end_df['neighbour_touch_weight'], alpha=0.1)
    # ax[4].scatter(end_df['fitness'], end_df['touch_phase_offset'], alpha=0.1)
    plt.show()


if __name__ == "__main__":
    main()
