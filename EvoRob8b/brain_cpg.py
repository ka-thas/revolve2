
import logging
import pickle
import math
from typing import Any

import config
import multineat
import numpy as np
import numpy.typing as npt

from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot_simulation import ModularRobotScene, simulate_scenes
from revolve2.modular_robot.body.v1 import ActiveHingeV1, BodyV1, BrickV1
from revolve2.standards import modular_robots_v1, fitness_functions, terrains
from revolve2.modular_robot.brain.cpg import BrainCpgNetworkNeighborRandom, active_hinges_to_cpg_network_structure_neighbor
from revolve2.experimentation.rng import make_rng_time_seed
from revolve2.simulators.mujoco_simulator import LocalSimulator
from revolve2.standards.simulation_parameters import make_standard_batch_parameters
from revolve2.modular_robot.brain.cpg import (
    BrainCpgNetworkStatic,
    active_hinges_to_cpg_network_structure_neighbor,
)




from revolve2.modular_robot.body.base import ActiveHinge, Body
from revolve2.modular_robot.brain.cpg._brain_cpg_network_neighbor import BrainCpgNetworkNeighbor




class Brain_cpg():




    def __init__(self, body):
        self.body = body
        self.brain: dict

# From Ege

    def initialize_brain(self, body):

        active_hinges = body.find_modules_of_type(ActiveHingeV1)
        (
            cpg_network_structure,
            output_mapping,
        ) = active_hinges_to_cpg_network_structure_neighbor(active_hinges)

        params = []

        # First, append the per-hinge base parameter (index 0 from each key)
        for active_hinge in active_hinges:
            grid_position = body.grid_position(active_hinge)
            brain_key = str(int(grid_position[0])) + "x" + str(int(grid_position[1]))
            params.append(self.brain[brain_key][0])

        # Then append parameters for every connection in the network
        for pair in cpg_network_structure.connections:
            low_active_hinge = active_hinges[pair.cpg_index_lowest.index]
            high_active_hinge = active_hinges[pair.cpg_index_highest.index]

            low_grid_position = body.grid_position(low_active_hinge)
            high_grid_position = body.grid_position(high_active_hinge)

            # Determine which index of the source hinge's parameter array
            # encodes the coupling parameter between these two positions
            low_brain_key = str(int(low_grid_position[0])) + "x" + str(int(low_grid_position[1]))
            array_index = self.grid_positions_to_array_number(low_grid_position, high_grid_position)
            params.append(self.brain[low_brain_key][array_index])

        brain = BrainCpgNetworkStatic.uniform_from_params(
            params=params,
            cpg_network_structure=cpg_network_structure,
            initial_state_uniform=0.5 * math.sqrt(2),
            output_mapping=output_mapping,
        )
        self.brain = brain
        return brain



    def update_brain_parameters(self, body, rng):

        # If new module is added

        active_hinges = body.find_modules_of_type(ActiveHingeV1)
        brain_keys = []
        for active_hinge in active_hinges:
            grid_position = body.grid_position(active_hinge)
            brain_keys.append(str(int(grid_position[0])) + "x" + str(int(grid_position[1])))

        for brain_key in brain_keys:
            if brain_key not in self.brain_dict.keys():
                self.brain_dict[brain_key] = np.array(rng.random(14)) * 2 - 1
        self.brain = self.brain_dict
    

    def mutate(self, rng: np.random.Generator):
        new_brain = {}
        for key, values in self.brain_dict.items():
            new_values = np.array([])
            for value in values:
                new_value = value
                if rng.random() < 0.8:
                    new_value = value + rng.normal(loc=0, scale=0.5) # TODO: clip
                new_values = np.append(new_values, new_value)
            new_brain[key] = new_values
        return new_brain

      


