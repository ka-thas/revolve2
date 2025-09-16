"""Evaluator class."""

import numpy as np
import numpy.typing as npt
from pyrr import Vector3

import config

from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot.body.base import Body
from revolve2.modular_robot_simulation import (
    ModularRobotScene,
    Terrain,
    simulate_scenes,
)
from revolve2.simulation.scene import Pose
from revolve2.simulators.mujoco_simulator import LocalSimulator

from revolve2.standards import terrains, fitness_functions
from revolve2.standards.simulation_parameters import make_standard_batch_parameters


class Evaluator:
    """Provides evaluation of robots."""

    _simulator: LocalSimulator
    _terrain: Terrain
    _body: Body

    def __init__(
        self,
        headless: bool,
        num_simulators: int,
    ) -> None:
        """
        Initialize this object.

        :param headless: `headless` parameter for the physics simulator.
        :param num_simulators: `num_simulators` parameter for the physics simulator.
        """
        self._simulator = LocalSimulator(
            headless=headless, num_simulators=num_simulators
        )

        if config.ENVIRONMENT == 'flat':
            self._terrain = terrains.flat_thin()
        elif config.ENVIRONMENT == 'hills':
            self._terrain = terrains.hills(height=0.2)
        elif config.ENVIRONMENT == 'steps':
            self._terrain = terrains.steps(height=0.25)
        elif config.ENVIRONMENT == 'notsonoisy':
            self._terrain = terrains.thin_crater((15, 15), 0.1, 0, 0.1)
        elif config.ENVIRONMENT == 'almostnoisy':
            self._terrain = terrains.thin_crater((10, 10), 0.2, 0, 0.1)
        elif config.ENVIRONMENT == 'noisy':
            self._terrain = terrains.thin_crater((10, 10), 0.3, 0, 0.1)

    def evaluate(
        self,
        robot: ModularRobot,
    ) -> float:
        """
        Evaluate multiple robots.

        Fitness is the distance traveled on the xy plane.

        :param robot: robot to evaluate.
        :returns: Fitnesses of the solutions.
        """

        # Create the scenes.
        scenes = []
        scene = ModularRobotScene(terrain=self._terrain)
        scene.add_robot(robot, pose=Pose(position=Vector3([0, 0, 0.1])))
        scenes.append(scene)

        # Simulate all scenes.
        scene_states = simulate_scenes(
            simulator=self._simulator,
            batch_parameters=make_standard_batch_parameters(),
            scenes=scenes,
        )

        if len(fitness_functions.detect_outliers(scene_states[0], robot)) > 1:
            return 0

        return fitness_functions.forward_displacement(
                scene_states[0][0].get_modular_robot_simulation_state(robot),
                scene_states[0][-1].get_modular_robot_simulation_state(robot),
            )
