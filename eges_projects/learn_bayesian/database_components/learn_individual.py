"""Individual class."""

from dataclasses import dataclass

import sqlalchemy

from database_components.genotype import Genotype
from .base import Base
from .learn_genotype import LearnGenotype

from sqlalchemy import orm

from revolve2.experimentation.optimization.ea import Individual as GenericIndividual


@dataclass
class LearnIndividual(
    Base, GenericIndividual[LearnGenotype], population_table="learn_population", kw_only=True
):
    """An individual in a population."""

    __tablename__ = "learn_individual"
    objective_value: orm.Mapped[float] = orm.mapped_column(nullable=False)
    morphology_genotype_id: orm.Mapped[int] = orm.mapped_column(
        sqlalchemy.ForeignKey("genotype.id"), nullable=False, init=False
    )
    morphology_genotype: orm.Mapped[Genotype] = orm.relationship()
