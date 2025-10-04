
import logging
import pickle
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


from revolve2.modular_robot.body.base import ActiveHinge, Body
from revolve2.modular_robot.brain.cpg._brain_cpg_network_neighbor import BrainCpgNetworkNeighbor




class Brain_cpg():


    def __init__(self, body, brain, rng):
        self.body = body
        self.brain = brain
        self.rng = rng

        self.brain_dict = {}

# From Ege

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

      


