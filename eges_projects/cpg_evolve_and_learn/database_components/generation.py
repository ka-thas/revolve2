"""Generation class."""

import sqlalchemy
import sqlalchemy.orm as orm
from database_components.base import Base
from database_components.experiment import Experiment
from database_components.population import Population

from revolve2.experimentation.database import HasId


class Generation(Base, HasId):
    """A single finished iteration of evolution."""

    __tablename__ = "generation"

    experiment_id: orm.Mapped[int] = orm.mapped_column(
        sqlalchemy.ForeignKey("experiment.id"), nullable=False, init=False
    )
    experiment: orm.Mapped[Experiment] = orm.relationship()
    generation_index: orm.Mapped[int] = orm.mapped_column(nullable=False)
    population_id: orm.Mapped[int] = orm.mapped_column(
        sqlalchemy.ForeignKey("population.id"), nullable=False, init=False
    )
    population: orm.Mapped[Population] = orm.relationship()
