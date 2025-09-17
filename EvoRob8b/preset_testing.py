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
from revolve2.modular_robot.brain.cpg import BrainCpgNetworkNeighborRandom
from revolve2.experimentation.rng import make_rng_time_seed
from revolve2.simulators.mujoco_simulator import LocalSimulator
from revolve2.standards.simulation_parameters import make_standard_batch_parameters


#taken from readme in revolve 2.

# Create a modular robot.
body = modular_robots_v1.snake_v1()
rng = make_rng_time_seed()
brain = BrainCpgNetworkNeighborRandom(body=body, rng=rng)
robot = ModularRobot(body, brain)

# Create a scene.
scene = ModularRobotScene(terrain=terrains.crater([15,15], 0.04, 3))
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

print(xy_displacement)