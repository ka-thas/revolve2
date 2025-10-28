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
import copy

import numpy as np
from sqlalchemy import orm
import random
import config

from revolve2.modular_robot.body.v1 import BodyV1, ActiveHingeV1
from revolve2.modular_robot.brain.cpg import (
    BrainCpgNetworkStatic,
    active_hinges_to_cpg_network_structure_neighbor,
    BrainCpgNetworkNeighborRandom,
    BrainCpgNetworkNeighbor,
    BrainCpgInstance
)



import logging
import pickle
from typing import Any

import config
import multineat
import numpy as np
import numpy.typing as npt
import numpy
import time

from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot_simulation import ModularRobotScene, simulate_scenes
from revolve2.modular_robot.body.v1 import ActiveHingeV1, BodyV1, BrickV1
from revolve2.standards import modular_robots_v1, fitness_functions, terrains
from revolve2.modular_robot.brain.cpg import BrainCpgNetworkNeighborRandom
from revolve2.experimentation.rng import make_rng_time_seed
from revolve2.simulators.mujoco_simulator import LocalSimulator
from revolve2.standards.simulation_parameters import make_standard_batch_parameters


from copy import deepcopy

def mutate_brain(weights, rng):

    """Return a mutated copy of this genotype.

    Mutation is applied element-wise to each parameter with 80% chance;
    changed parameters are perturbed by a Gaussian with sigma=0.5.

    Args:
        rng: numpy random Generator used for sampling mutations

    Returns:
        BrainGenotype: new genotype with mutated parameter arrays
    """
    # print("\n Old weights")
    # print(weights)     
    epsilon = config.MUTATION_EPSILON   
    for y in range(len(weights)):
            for x in range(len(weights[y])):
                if (weights[y][x] != 0 and random.random() > 1-config.BRAIN_MUTATION_RATE ):
                    weights[y][x] += + rng.normal(loc=0, scale=config.MUTATION_EPSILON)
    # print("\n new weights")
    # print(weights)        



    return weights

class BrainGenotype():

    """
    def get_weights(self):

    def get_outputmap(self):
    
    def get_initial_state(self):
    """


    def __init__(self, brain=None):
            """Initialize the BrainGenotype with the given brain dictionary.

            Args:
                brain: Optional; a dictionary mapping grid positions to parameter arrays
            """

            self.weights = []
            self.brain = None
            self.fitness = 0




    def update_brain_parameters(self, body, rng):
        """
        Same rng, we can update to account for morphologigal differences
        """
        active_hinges = body.find_modules_of_type(ActiveHingeV1)
        (
            cpg_network_structure,
            output_mapping,
        ) = active_hinges_to_cpg_network_structure_neighbor(active_hinges)


        brain = BrainCpgNetworkNeighborRandom(body=body, rng=rng)
        brain._make_weights(active_hinges, output_mapping, rng)
        self.brain = brain
        self.brain.make_instance()
        self.weights = self.brain.get_weights()
        

    def get_weights(self):
         return self.weights
    
    def update_weights(self, new_weights):
        self.brain.update_weights(new_weights)
        self.weights = new_weights


    def develop_brain(self, body, rng, weights=np.array(0)):
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

        # cpgnetwork structure has connectionsnew_weights
        # Grid brain match grid body, active hinges get pos
        # need 3 weights for each grid pos, or 2
        # one for inetrnal , one for the neighboring tree distance
        if config.DEBUG_BRAIN: print(cpg_network_structure.connections)


        brain = BrainCpgNetworkNeighborRandom(body=body, rng=rng)
        if (weights.any()):
            brain.update_weights(weights)
        else:
            brain._make_weights(active_hinges, cpg_network_structure.connections, rng)
        self.brain = brain
        self.weights = brain.get_weights()

        if config.DEBUG_BRAIN:
            print("weights: ")
            print(brain.get_weights())
            print("\n\n outputmap: ")
        self.weights = brain.get_weights()


            #print(f"weights: " + brain.get_weights() + "\n\n outputmap: " + brain.get_outputmap()+ "\n\n initial state: " + brain.get_initial_state())

        


    def make_instance(self) -> BrainCpgInstance:
        """
        Create an instance of this brain.

        :returns: The created instance.
        """

        return self.brain.make_instance()


    def improve(self, body, iterations, rng, terrain):
        start_time = time.time()
        best_fitness = 0
        best_brain = copy.deepcopy(self.brain)
        best_brain.make_instance()
        iterations_since_update = 0
        self.weights = best_brain.get_weights()


        while (iterations > 0):
            if config.DEBUG_BRAIN:
                print(f" \n iteration: ", {config.INNER_LOOP_ITERATIONS -iterations})
                print("\n Best fitness: ")
                print(best_fitness)               

            old_weights = best_brain.get_weights()

            mutated_weights = mutate_brain(old_weights, rng)

            best_brain.update_weights(mutated_weights)

            robot = ModularRobot(body, best_brain)
            # Create a scene.
            scene = ModularRobotScene(terrain=terrain)
            scene.add_robot(robot)

            # Create a simulator.
            simulator = LocalSimulator(headless=True)
            # Simulate the scene and obtain states sampled during the simulation.
            scene_states = simulate_scenes(
                simulator=simulator,
                batch_parameters=make_standard_batch_parameters(),
                scenes=scene,
            )
            
            # Get the state at the beginning and end of the simulation.
            scene_state_begin = scene_states[0]
            scene_state_end = scene_states[-1]

            # Retrieve the states of the modular robot.
            robot_state_begin = scene_state_begin.get_modular_robot_simulation_state(robot)
            robot_state_end = scene_state_end.get_modular_robot_simulation_state(robot)

            # Calculate the xy displacement of the robot.
            xy_displacement = fitness_functions.xy_displacement(
                robot_state_begin, robot_state_end
            )

            if config.DEBUG_BRAIN: print(xy_displacement)

            iterations_since_update += 1
            # If worse, revert
            if (xy_displacement < best_fitness):
                best_brain.update_weights(old_weights)
            else:
                best_fitness = xy_displacement
                iterations_since_update = 0

            iterations -= 1
            
        self.brain = best_brain
        self.weights = self.brain.get_weights()
        self.brain.make_instance()
        self.fitness = best_fitness

        end_time = time.time()
        if config.DEBUG_BRAIN:
            print(f"\n total_time", end_time-start_time)

    
