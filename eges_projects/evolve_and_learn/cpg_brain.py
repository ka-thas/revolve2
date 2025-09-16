import uuid

import numpy as np
import numpy.typing as npt

from revolve2.modular_robot.body.base import ActiveHinge, Body
from revolve2.modular_robot.brain.cpg import BrainCpgNetworkNeighbor


class CpgBrain(BrainCpgNetworkNeighbor):
    """The Cpg brain."""
    brain: dict[uuid.UUID, npt.NDArray[np.float_]]

    def __init__(self, body, brain):
        self.brain = brain
        super().__init__(body)

    def _make_weights(self, active_hinges: list[ActiveHinge], connections: list[tuple[ActiveHinge, ActiveHinge]],
                      body: Body) -> tuple[list[float], list[float]]:
        internal_weights = [self.brain[active_hinge.map_uuid][0] * 8 - 4 for active_hinge in active_hinges]

        external_weights = []
        for (active_hinge1, active_hinge2) in connections:
            if active_hinge1 in active_hinge2.neighbours(within_range=1) or active_hinge2 in active_hinge1.neighbours(within_range=1):
                external_weights.append(self.brain[active_hinge1.map_uuid][1] * 8 - 4 )
            else:
                external_weights.append(self.brain[active_hinge1.map_uuid][2] * 8 - 4 )

        return (internal_weights, external_weights)

