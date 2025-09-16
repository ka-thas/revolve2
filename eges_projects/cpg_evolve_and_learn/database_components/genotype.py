"""Genotype class."""

from __future__ import annotations

from copy import copy

import multineat
import numpy as np
import sqlalchemy.orm as orm
from dataclasses import field

from revolve2.standards.genotypes.cppnwin.modular_robot.v1 import BodyGenotypeOrmV1
from revolve2.standards.genotypes.cppnwin.modular_robot import BrainGenotypeCpgOrm
from .base import Base

from genotypes.brain_genotype_direct import BrainGenotype
from genotypes.body_genotype_direct_asym import BodyGenotypeDirect
from revolve2.experimentation.database import HasId


class Genotype(Base, HasId, BodyGenotypeOrmV1, BrainGenotype):
    """SQLAlchemy model for a genotype for a modular robot body and brain."""

    __tablename__ = "genotype"
    parent_1_genotype_id: orm.Mapped[int] = orm.mapped_column(default=-1)
    parent_2_genotype_id: orm.Mapped[int] = orm.mapped_column(default=-1)
    innov_db_brain: multineat.InnovationDatabase = field(default_factory=lambda: None)
    best_population: list = field(default_factory=lambda: [])

    @classmethod
    def initialize(
        cls,
        innov_db_body: multineat.InnovationDatabase,
        rng: np.random.Generator,
    ) -> Genotype:
        """
        Create a random genotype.

        :param innov_db_body: Multineat innovation database for the body. See Multineat library.
        :param innov_db_brain: Multineat innovation database for the brain. See Multineat library.
        :param rng: Random number generator.
        :returns: The created genotype.
        """
        body = cls.random_body(innov_db_body, rng)

        genotype = Genotype(body=body.body, brain={})
        #genotype.innov_db_brain = multineat.InnovationDatabase()
        return genotype

    def mutate(
        self,
        innov_db_body: multineat.InnovationDatabase,
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
        body = self.mutate_body(innov_db_body, rng)

        genotype = Genotype(body=body.body, brain=copy(self.brain))
        genotype.innov_db_brain = self.innov_db_brain
        return genotype

    @classmethod
    def crossover(
            cls,
            parent1: Genotype,
            parent2: Genotype,
            rng: np.random.Generator,
    ) -> Genotype:
        """
        Perform crossover between two genotypes.

        :param parent1: The first genotype.
        :param parent2: The second genotype.
        :param rng: Random number generator.
        :returns: A newly created genotype.
        """
        body = cls.crossover_body(parent1, parent2, rng)
        brain = cls.crossover_brain(parent1, parent2, rng)

        return Genotype(body=body.body, brain=brain.brain)
