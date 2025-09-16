"""Genotype class."""

from __future__ import annotations

from .base import Base
from genotypes.brain_genotype_simple import BrainGenotype
from genotypes.body_genotype_direct import BodyGenotypeDirect

from revolve2.experimentation.database import HasId
from revolve2.modular_robot import ModularRobot


class LearnGenotype(Base, HasId, BrainGenotype, BodyGenotypeDirect):
    """A genotype that is an array of parameters."""

    __tablename__ = "learn_genotype"

    def develop(self) -> ModularRobot:
        """
        Develop the genotype into a modular robot.

        :returns: The created robot.
        """
        body = self.develop_body()
        brain = self.develop_brain(body)
        return ModularRobot(body=body, brain=brain)
