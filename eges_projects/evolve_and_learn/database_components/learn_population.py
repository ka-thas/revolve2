"""Population class."""

from database_components.base import Base
from database_components.learn_individual import LearnIndividual

from revolve2.experimentation.optimization.ea import Population as GenericPopulation


class LearnPopulation(Base, GenericPopulation[LearnIndividual], kw_only=True):
    """A population of individuals."""

    __tablename__ = "learn_population"
