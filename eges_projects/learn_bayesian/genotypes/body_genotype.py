from abc import abstractmethod

import numpy as np

from genotypes.brain_genotype_simple import BrainGenotype


class BodyGenotype:

    @classmethod
    @abstractmethod
    def initialize_body(cls, rng, brain: BrainGenotype):
        pass

    @abstractmethod
    def mutate_body_start(
            self,
            rng: np.random.Generator, brain: BrainGenotype
    ):
        pass

    @abstractmethod
    def develop_body(self):
        pass
