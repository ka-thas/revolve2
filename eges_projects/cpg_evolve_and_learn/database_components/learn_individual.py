"""Individual class."""

from dataclasses import dataclass

import sqlalchemy
from sqlalchemy import orm

from database_components.base import Base
from database_components.genotype import Genotype
from database_components.learn_genotype import LearnGenotype

from revolve2.experimentation.database import HasId


@dataclass
class LearnIndividual(Base, HasId):
    """An individual in a population."""

    __tablename__ = "learn_individual"
    objective_value: orm.Mapped[float] = orm.mapped_column(nullable=False)
    generation_index: orm.Mapped[int] = orm.mapped_column(nullable=False)
    morphology_genotype_id: orm.Mapped[int] = orm.mapped_column(
        sqlalchemy.ForeignKey("genotype.id"), nullable=False, init=False
    )
    morphology_genotype: orm.Mapped[Genotype] = orm.relationship()
    genotype_id: orm.Mapped[int] = orm.mapped_column(
        sqlalchemy.ForeignKey("learn_genotype.id"), nullable=False, init=False
    )
    genotype: orm.Mapped[LearnGenotype] = orm.relationship()
