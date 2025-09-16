"""Genotype class."""

from __future__ import annotations

import numpy as np
import sqlalchemy.orm as orm
from .base import Base
from body_genotype_generative import BodyGenotypeGenerative
from brain_genotype import BrainGenotype

from revolve2.experimentation.database import HasId


class Genotype(Base, HasId, BodyGenotypeGenerative, BrainGenotype):
    """SQLAlchemy model for a genotype for a modular robot body and brain."""

    __tablename__ = "genotype"
    parent_1_genotype_id: orm.Mapped[int] = orm.mapped_column(default=-1)
    parent_2_genotype_id: orm.Mapped[int] = orm.mapped_column(default=-1)

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
        brain = cls.initialize_brain()
        body = cls.initialize_body(rng=rng, brain=brain)

        return Genotype(body=body, brain=brain.brain)

    def mutate(
        self,
        rng: np.random.Generator,
    ) -> Genotype:
        """
        Mutate this genotype.

        This genotype will not be changed; a mutated copy will be returned.

        :param innov_db_body: Multineat innovation database for the body. See Multineat library.
        :param innov_db_brain: Multineat innovation database for the brain. See Multineat library.
        :param rng: Random number generator.
        :returns: A mutated copy of the provided genotype.
        """
        brain = self.mutate_brain(rng)
        body = self.body.mutate_body(rng, brain)

        return Genotype(body=body, brain=brain.brain)
