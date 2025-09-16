"""Genotype class."""

from __future__ import annotations

import uuid

import numpy as np
import sqlalchemy.orm as orm
from dataclasses import field
from .base import Base

import config
from genotypes.body_genotype_direct import BodyGenotypeDirect
from genotypes.body_genotype_cppn import BodyGenotypeCppn
from genotypes.brain_genotype_simple import BrainGenotype as BrainGenotypeSimple
from genotypes.brain_genotype_cppn_simple import BrainGenotype as BrainGenotypeCppnSimple
from genotypes.brain_genotype_cpg import BrainGenotype as BrainGenotypeCpg
from genotypes.brain_genotype_cppn_cpg import BrainGenotype as BrainGenotypeCppnCpg
from revolve2.experimentation.database import HasId


class Genotype(Base, HasId, BodyGenotypeDirect, BrainGenotypeSimple):
    """SQLAlchemy model for a genotype for a modular robot body and brain."""

    __tablename__ = "genotype"
    experience: list = field(default_factory=lambda: [])
    inherited_experience: list = field(default_factory=lambda: [])
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
        body, mutation_parameter = self.mutate_body_start(rng, brain)

        genotype = Genotype(body=body.body, brain=brain.brain)
        genotype.mutation_parameter = mutation_parameter
        return genotype

    @staticmethod
    def crossover(
        parent1: Genotype,
        parent2: Genotype,
        rng: np.random.Generator,
    ) -> (Genotype, Genotype):
        """
        Perform crossover between two genotypes.

        :param parent1: The first genotype.
        :param parent2: The second genotype.
        :param rng: Random number generator.
        :returns: A newly created genotype.
        """
        child1_body, child2_body = BodyGenotypeDirect.crossover_body(parent1, parent2, rng)

        if config.CONTROLLERS != -1:
            child1_brain, child2_brain = BrainGenotypeSimple.crossover_brain(parent1, parent2, rng)
            return Genotype(body=child1_body.body, brain=child1_brain.brain), Genotype(body=child2_body.body, brain=child2_brain.brain)

        all_brains = {**parent1.brain, **parent2.brain}

        child_1_brain = {key: all_brains[key] for key in child1_body.get_brain_uuids() if key in all_brains}
        child_2_brain = {key: all_brains[key] for key in child2_body.get_brain_uuids() if key in all_brains}

        if len(child_1_brain.keys()) == 0:
            new_uuid = uuid.uuid4()
            child_1_brain = {new_uuid: np.array(rng.random(4))}

        if len(child_2_brain.keys()) == 0:
            new_uuid = uuid.uuid4()
            child_2_brain = {new_uuid: np.array(rng.random(4))}

        return Genotype(body=child1_body.body, brain=child_1_brain), Genotype(body=child2_body.body, brain=child_2_brain)
