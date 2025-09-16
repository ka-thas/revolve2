"""Genotype class."""

from __future__ import annotations

from .base import Base
from brain_genotype import BrainGenotype
from body_genotype_generative import BodyGenotypeGenerative

from revolve2.experimentation.database import HasId
from revolve2.modular_robot import ModularRobot


class LearnGenotype(Base, HasId, BrainGenotype, BodyGenotypeGenerative):
    """A genotype that is an array of parameters."""

    __tablename__ = "learn_genotype"

    def develop(self) -> ModularRobot:
        """
        Develop the genotype into a modular robot.

        :returns: The created robot.
        """
        body = self.body.develop_body()
        brain = self.develop_brain(body)
        return ModularRobot(body=body, brain=brain)
