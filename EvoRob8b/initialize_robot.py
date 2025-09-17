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



# Write jason file for one individual robot 


