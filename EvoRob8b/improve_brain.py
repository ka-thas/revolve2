
import logging
import pickle
from typing import Any

import config
import multineat
import numpy as np
import numpy.typing as npt
from brain_cpg import Brain_cpg

from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot_simulation import ModularRobotScene, simulate_scenes
from revolve2.modular_robot.body.v1 import ActiveHingeV1, BodyV1, BrickV1
from revolve2.standards import modular_robots_v1, fitness_functions, terrains
from revolve2.modular_robot.brain.cpg import BrainCpgNetworkNeighborRandom
from revolve2.experimentation.rng import make_rng_time_seed
from revolve2.simulators.mujoco_simulator import LocalSimulator
from revolve2.standards.simulation_parameters import make_standard_batch_parameters



class Improve_brain():


    def __init__(self, body, brain, rng):
        self.body = body
        self.brain = brain

        self.brain_list = []
        self.fitness_list = [] # 1:1 with brains
        self.best_fitness = 0

    def improve(self, iterations, rng):


        # Create a scene.
        scene = ModularRobotScene(terrain=terrains.crater([4,4], 0.04, 3))



        x = 0

        while (x < iterations):
            new_brain = self.brain.mutate(rng)
            
            self.brain_list.append(new_brain)
            x += 1
        
        for brains in self.brain_list:
            robot = ModularRobot(self.body, brains)                
            scene.add_robot(robot)

            # Create a simulator.
            simulator = LocalSimulator(headless=False)

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

            self.fitness_list.append(xy_displacement)
            if (xy_displacement > self.best_fitness):
                self.best_fitness = xy_displacement
                self.brain = brains

    def get_brain(self):
        return self.brain
            

    




            

