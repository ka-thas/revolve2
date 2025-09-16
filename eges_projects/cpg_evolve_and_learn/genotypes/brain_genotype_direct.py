import math
from copy import copy

import numpy as np
from sqlalchemy import orm

from revolve2.modular_robot.body.v1 import BodyV1, ActiveHingeV1
from revolve2.modular_robot.brain.cpg import BrainCpgNetworkStatic, active_hinges_to_cpg_network_structure_neighbor


class BrainGenotype(orm.MappedAsDataclass):

    brain: dict

    def update_brain_parameters(self, developed_body: BodyV1, rng):
        active_hinges = developed_body.find_modules_of_type(ActiveHingeV1)
        brain_keys = []
        for active_hinge in active_hinges:
            grid_position = developed_body.grid_position(active_hinge)
            brain_keys.append(str(int(grid_position[0])) + "x" + str(int(grid_position[1])))

        for brain_key in brain_keys:
            if brain_key not in self.brain.keys():
                self.brain[brain_key] = np.array(rng.random(14)) * 2 - 1

    def mutate_brain(self, rng: np.random.Generator):
        new_brain = {}
        for key, values in self.brain.items():
            new_values = np.array([])
            for value in values:
                new_value = value
                if rng.random() < 0.8:
                    new_value = value + rng.normal(loc=0, scale=0.5) # TODO: clip
                new_values = np.append(new_values, new_value)
            new_brain[key] = new_values
        return BrainGenotype(brain=new_brain)


    @classmethod
    def crossover_brain(cls, parent1, parent2, rng):
        return BrainGenotype(brain=copy(parent1.brain))

    def develop_brain(self, body):
        active_hinges = body.find_modules_of_type(ActiveHingeV1)
        (
            cpg_network_structure,
            output_mapping,
        ) = active_hinges_to_cpg_network_structure_neighbor(active_hinges)
        params = []
        for active_hinge in active_hinges:
            grid_position = body.grid_position(active_hinge)
            brain_key = str(int(grid_position[0])) + "x" + str(int(grid_position[1]))
            params.append(self.brain[brain_key][0])
        for pair in cpg_network_structure.connections:
            low_active_hinge = active_hinges[pair.cpg_index_lowest.index]
            high_active_hinge = active_hinges[pair.cpg_index_highest.index]
            low_grid_position = body.grid_position(low_active_hinge)
            high_grid_position = body.grid_position(high_active_hinge)
            low_brain_key = str(int(low_grid_position[0])) + "x" + str(int(low_grid_position[1]))
            params.append(self.brain[low_brain_key][self.grid_positions_to_array_number(low_grid_position, high_grid_position)])

        brain = BrainCpgNetworkStatic.uniform_from_params(
                    params=params,
                    cpg_network_structure=cpg_network_structure,
                    initial_state_uniform=0.5 * math.sqrt(2),
                    output_mapping=output_mapping,
                )
        return brain

    @staticmethod
    def grid_positions_to_array_number(low_grid_position, high_grid_position):
        if low_grid_position[0] - high_grid_position[0] == -2:
            return 0
        if low_grid_position[0] - high_grid_position[0] == -1:
            if low_grid_position[1] - high_grid_position[1] == -1:
                return 1
            if low_grid_position[1] == high_grid_position[1]:
                return 2
            if low_grid_position[1] - high_grid_position[1] == 1:
                return 3
        if low_grid_position[0] == high_grid_position[0]:
            if low_grid_position[1] - high_grid_position[1] == -2:
                return 4
            if low_grid_position[1] - high_grid_position[1] == -1:
                return 5
            if low_grid_position[1] == high_grid_position[1]:
                return 6
            if low_grid_position[1] - high_grid_position[1] == 1:
                return 7
            if low_grid_position[1] - high_grid_position[1] == 2:
                return 8
        if low_grid_position[0] - high_grid_position[0] == 1:
            if low_grid_position[1] - high_grid_position[1] == -1:
                return 9
            if low_grid_position[1] == high_grid_position[1]:
                return 10
            if low_grid_position[1] - high_grid_position[1] == 1:
                return 11
        if low_grid_position[0] - high_grid_position[0] == 2:
            return 13