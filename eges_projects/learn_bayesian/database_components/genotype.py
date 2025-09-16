"""Genotype class."""

from __future__ import annotations

import numpy as np
import sqlalchemy.orm as orm
from .base import Base

from genotypes.body_genotype_direct import BodyGenotypeDirect
from genotypes.brain_genotype_simple import BrainGenotype
from revolve2.experimentation.database import HasId
from revolve2.modular_robot import ModularRobot


class Genotype(Base, HasId, BodyGenotypeDirect, BrainGenotype):
    """SQLAlchemy model for a genotype for a modular robot body and brain."""

    __tablename__ = "genotype"
    parent_1_genotype_id: orm.Mapped[int] = orm.mapped_column(default=-1)
    parent_2_genotype_id: orm.Mapped[int] = orm.mapped_column(default=-1)
    mutation_parameter: orm.Mapped[float] = orm.mapped_column(nullable=True, default=None)

    @classmethod
    def initialize(
        cls,
        rng: np.random.Generator,
    ) -> Genotype:
        """
        Create a random genotype.

        :param innov_db_body: Multineat innovation database for the body. See Multineat library.
        :param innov_db_brain: Multineat innovation database for the brain. See Multineat library.
        :param rng: Random number generator.
        :returns: The created genotype.
        """
        brain = cls.initialize_brain(rng=rng)
        body = cls.initialize_body(rng=rng, brain=brain)

        return Genotype(body=body.body, brain=brain.brain)

    def develop(self) -> ModularRobot:
        """
        Develop the genotype into a modular robot.

        :returns: The created robot.
        """
        body = self.develop_body()
        brain = self.develop_brain(body)
        return ModularRobot(body=body, brain=brain)
