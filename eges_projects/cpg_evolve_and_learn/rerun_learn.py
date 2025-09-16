"""Rerun the best robot between all experiments."""

import config
from evaluator import Evaluator
from sqlalchemy import select
from sqlalchemy.orm import Session

from database_components.learn_genotype import LearnGenotype
from database_components.learn_individual import LearnIndividual
from database_components.genotype import Genotype
from database_components.individual import Individual
from database_components.population import Population
from database_components.generation import Generation
from revolve2.experimentation.database import OpenMethod, open_database_sqlite
from revolve2.experimentation.logging import setup_logging


def main() -> None:
    """Perform the rerun."""
    setup_logging()

    # Load the best individual from the database.
    dbengine = open_database_sqlite(
        config.DATABASE_FILE, open_method=OpenMethod.OPEN_IF_EXISTS
    )

    with Session(dbengine) as ses:
        row = ses.execute(
            select(Genotype, LearnGenotype, LearnIndividual.objective_value)
            .join_from(LearnGenotype, LearnIndividual, LearnGenotype.id == LearnIndividual.genotype_id)
            .join_from(LearnIndividual, Genotype, LearnIndividual.morphology_genotype_id == Genotype.id)
            .join_from(Genotype, Individual, Individual.genotype_id == Genotype.id)
            .join_from(Individual, Population, Population.id == Individual.population_id)
            .join_from(Population, Generation, Generation.population_id == Population.id)
            .where(Generation.generation_index > -1)
            .order_by(LearnIndividual.objective_value.desc())
            .limit(1)
        ).one()
        assert row is not None

        genotype = row[0]
        learn_genotype = row[1]
        fitness = row[2]
    modular_robot = learn_genotype.develop(genotype.develop_body())

    # Create the evaluator.
    evaluator = Evaluator(headless=False, num_simulators=1)

    # Show the robot.
    evaluator.evaluate(modular_robot)


if __name__ == "__main__":

    main()
