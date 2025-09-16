"""Individual class."""

from dataclasses import dataclass

from .base import Base
from .learn_genotype import LearnGenotype

from revolve2.experimentation.optimization.ea import Individual as GenericIndividual


@dataclass
class LearnIndividual(
    Base, GenericIndividual[LearnGenotype], population_table="learn_population", kw_only=True
):
    """An individual in a population."""

    __tablename__ = "learn_individual"
