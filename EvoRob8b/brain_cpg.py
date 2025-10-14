import math
from copy import copy

import numpy as np
"""Brain genotype for direct CPG parameter encoding.

This module defines a simple genotype representation for CPG-based brains used
in the `cpg_evolve_and_learn` experiments. The genotype stores a mapping from
grid positions to parameter arrays. When a robot body (BodyV1) is developed,
parameters are extracted from this mapping and assembled into a static CPG
network (`BrainCpgNetworkStatic`).

Key concepts:
- `brain` (dict): maps a string key "{x}x{y}" representing an active hinge's
  grid position to a 1D numpy array of parameters.
- `update_brain_parameters`: ensures the genotype has parameter vectors for all
  active hinges present in a developed body (initialises missing entries).
- `mutate_brain`: returns a mutated copy of the genotype.
- `crossover_brain`: currently returns a copy of one parent (placeholder).
- `develop_brain`: converts the genotype -> CPG brain instance used in
  simulation by assembling parameter vectors in the order required by the
  network-structure helper.

The implementation is intentionally small and focused; more advanced
recombination or parameter encoding schemes can replace the simple
copy/mutate operators here.
"""

import math
from copy import copy

import numpy as np
from sqlalchemy import orm

from revolve2.modular_robot.body.v1 import BodyV1, ActiveHingeV1
from revolve2.modular_robot.brain.cpg import (
    BrainCpgNetworkStatic,
    active_hinges_to_cpg_network_structure_neighbor,
    BrainCpgNetworkNeighborRandom,
    BrainCpgNetworkNeighbor,
    BrainCpgInstance
)


class BrainGenotype():

    # The `brain` attribute stores per-hinge parameter arrays keyed by grid
    # coordinates encoded as strings, e.g. "3x-1" -> np.ndarray
    def __init__(self, brain=None):
            """Initialize the BrainGenotype with the given brain dictionary.

            Args:
                brain: Optional; a dictionary mapping grid positions to parameter arrays
            """
            # Initialize the brain with an empty dictionary
            if (brain == None):
                self.brain =  {}
            else:
                self.brain = brain

            self._initial_state = 0
            self._weight_matrix = 0
            self._output_mapping = 0


    def update_brain_parameters(self, developed_body: BodyV1, rng):
        """Ensure genotype contains parameters for every active hinge.

        This inspects `developed_body` to find all ActiveHingeV1 modules and
        creates a brain key for each hinge based on its grid position. If the
        key does not yet exist in `self.brain`, a new parameter vector of
        length 14 is sampled uniformly in [-1, 1] using the provided RNG.

        Args:
            developed_body: a BodyV1 instance already 'developed'
            rng: a numpy-compatible RNG-like object exposing `random()`
                 and `random(n)` (used here to initialise parameters)
        """


    def mutate_brain(self, rng: np.random.Generator):
        """Return a mutated copy of this genotype.

        Mutation is applied element-wise to each parameter with 80% chance;
        changed parameters are perturbed by a Gaussian with sigma=0.5.

        Args:
            rng: numpy random Generator used for sampling mutations

        Returns:
            BrainGenotype: new genotype with mutated parameter arrays
        """
        


    @classmethod
    def crossover_brain(cls, parent1, parent2, rng):
        """Crossover operator for brains.

        Current implementation is a simple copy of parent1. This is a
        placeholder — replace with true recombination (e.g. per-key mixing or
        uniform crossover) when desired.
        """
        return BrainGenotype(brain=copy(parent1.brain))

    def develop_brain(self, body, rng):
        """Convert the genotype into a runnable CPG brain instance.

        Steps:
        1. Find active hinges present in `body`.
        2. Query the helper `active_hinges_to_cpg_network_structure_neighbor`
           to obtain the CPG network topology (`cpg_network_structure`) and
           the `output_mapping` used by `BrainCpgNetworkStatic`.
        """
        active_hinges = body.find_modules_of_type(ActiveHingeV1)
        (
            cpg_network_structure,
            output_mapping,
        ) = active_hinges_to_cpg_network_structure_neighbor(active_hinges)


        brain = BrainCpgNetworkNeighborRandom(body=body, rng=rng)
        brain._make_weights(active_hinges, output_mapping, rng)

        self.brain = brain

        


    def make_instance(self) -> BrainCpgInstance:
        """
        Create an instance of this brain.

        :returns: The created instance.
        """

        return self.brain.make_instance()