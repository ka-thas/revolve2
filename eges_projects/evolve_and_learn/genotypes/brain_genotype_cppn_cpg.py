import numpy as np

from genotypes.brain_genotype_cpg import BrainGenotype as BrainGenotypeCpg
from revolve2.modular_robot.body.base import ActiveHinge


class BrainGenotype(BrainGenotypeCpg):

    @classmethod
    def initialize_brain(cls, rng) -> 'BrainGenotype':
        return BrainGenotype(brain={})

    def update_brain_parameters(self, developed_body, rng):
        active_hinges = developed_body.find_modules_of_type(ActiveHinge)
        for active_hinge in active_hinges:
            self.brain[active_hinge.map_uuid] = np.array(rng.random(3))
