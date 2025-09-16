import pandas
import os
import matplotlib.pyplot as plt
from revolve2.experimentation.database import open_database_sqlite, OpenMethod
from sqlalchemy import select

from database_components.experiment import Experiment
from database_components.generation import Generation
from database_components.genotype import Genotype
from database_components.individual import Individual
from database_components.population import Population
from database_components.learn_individual import LearnIndividual


def get_df(learn, controllers, environment, survivor_select, inherit_samples):
    database_name = f"learn-{learn}_controllers-{controllers}_survivorselect-{survivor_select}_parentselect-tournament_inheritsamples-{inherit_samples}_environment-{environment}"
    print(database_name)
    files = [file for file in os.listdir("results/0301") if file.startswith(database_name)]
    if len(files) == 0:
        return None
    dfs = []
    i = 1
    for file_name in files:
        if i > 5:
            break
        dbengine = open_database_sqlite("results/0301/" + file_name, open_method=OpenMethod.OPEN_IF_EXISTS)

        df_mini = pandas.read_sql(
            select(
                Experiment.id.label("experiment_id"),
                Generation.generation_index,
                Genotype.id.label("genotype_id"),
                LearnIndividual.generation_index.label('learn_generation_index'),
                LearnIndividual.objective_value
            )
            .join_from(Experiment, Generation, Experiment.id == Generation.experiment_id)
            .join_from(Generation, Population, Generation.population_id == Population.id)
            .join_from(Population, Individual, Population.id == Individual.population_id)
            .join_from(Individual, Genotype, Individual.genotype_id == Genotype.id)
            .join_from(Genotype, LearnIndividual, Genotype.id == LearnIndividual.morphology_genotype_id),
            dbengine,
        )
        df_mini['experiment_id'] = i
        dfs.append(df_mini)
        i += 1
    return pandas.concat(dfs)


def main() -> None:
    fig, ax = plt.subplots(nrows=4, sharex='col', sharey='row')
    for i, inherit_samples in enumerate(['-1', '5', '0', '-2']):
        df = get_df('30', 'adaptable', 'noisy', 'newest', inherit_samples)
        grouped = df.groupby(['experiment_id', 'genotype_id'])

        snaggywaggy = {
            'difference': [],
            'generation_index': [],
        }
        for name, group in grouped:
            first_five = group.loc[group['learn_generation_index'] < 5]
            last_five = group.loc[group['learn_generation_index'] >= 30 - 5]
            snaggywaggy['difference'].append(last_five.max()['objective_value'] - first_five.max()['objective_value'])
            snaggywaggy['generation_index'].append(first_five.mean()['generation_index'])

        thisisit = pandas.DataFrame(snaggywaggy)
        agg = (
            thisisit.groupby(["generation_index"])
            .agg({"difference": ["mean"]})
            .reset_index()
        )
        agg.columns = [
            "generation_index",
            "mean_difference",
        ]
        ax[i].plot(agg['generation_index'], agg['mean_difference'])
    ax[0].set_title('No inheritance')
    ax[1].set_title('Redo samples')
    ax[2].set_title('Inherit samples')
    ax[3].set_title('Inherit prior')
    plt.show()


if __name__ == "__main__":
    main()
