"""Generation class."""

import sqlalchemy
import sqlalchemy.orm as orm
from database_components.base import Base
from database_components.genotype import Genotype
from database_components.learn_population import LearnPopulation

from revolve2.experimentation.database import HasId


class LearnGeneration(Base, HasId):
    """A single finished iteration of BO."""

    __tablename__ = "learn_generation"

    genotype_id: orm.Mapped[int] = orm.mapped_column(
        sqlalchemy.ForeignKey("genotype.id"), nullable=False, init=False
    )
    genotype: orm.Mapped[Genotype] = orm.relationship()
    generation_index: orm.Mapped[int] = orm.mapped_column(nullable=False)
    learn_population_id: orm.Mapped[int] = orm.mapped_column(
        sqlalchemy.ForeignKey("learn_population.id"), nullable=False, init=False
    )
    learn_population: orm.Mapped[LearnPopulation] = orm.relationship()
