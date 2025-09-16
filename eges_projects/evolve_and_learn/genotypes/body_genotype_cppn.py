import uuid

import multineat
import numpy as np
from pyrr import Vector3

import config
from genotypes.body_genotype import BodyGenotype
from genotypes.brain_genotype_simple import BrainGenotype
from revolve2.modular_robot.body.base import ActiveHinge
from revolve2.modular_robot.body.v1 import BodyV1

from revolve2.standards.genotypes.cppnwin.modular_robot.v1 import BodyGenotypeOrmV1


class BodyGenotypeCppn(BodyGenotypeOrmV1, BodyGenotype):
    innovation_database = multineat.InnovationDatabase()

    @classmethod
    def initialize_body(cls, rng, brain: BrainGenotype):
        return cls.random_body(BodyGenotypeCppn.innovation_database, rng)

    def mutate_body_start(
        self,
        rng: np.random.Generator, brain: BrainGenotype
    ):
        return super().mutate_body(BodyGenotypeCppn.innovation_database, rng), 0.0

    def develop_body(self) -> BodyV1:
        body = super().develop_body()

        active_hinges = body.find_modules_of_type(ActiveHinge)
        for active_hinge in active_hinges:
            grid_position = body.grid_position(active_hinge) + Vector3([10, 10, 10])
            active_hinge.map_uuid = uuid.UUID(int=int(grid_position[0] + grid_position[1] * (config.MAX_NUMBER_OF_MODULES * 2) + grid_position[2] * ((config.MAX_NUMBER_OF_MODULES * 2) ** 2)))

        return body
