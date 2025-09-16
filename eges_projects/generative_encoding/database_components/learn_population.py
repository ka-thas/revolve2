"""Population class."""

from .base import Base
from .learn_individual import LearnIndividual

from revolve2.experimentation.optimization.ea import Population as GenericPopulation


class LearnPopulation(Base, GenericPopulation[LearnIndividual], kw_only=True):
    """A population of individuals."""

    __tablename__ = "learn_population"
