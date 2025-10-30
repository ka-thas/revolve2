"""Standard fitness functions for modular robots."""

import math

from revolve2.modular_robot_simulation import ModularRobotSimulationState


def xy_displacement(
    begin_state: ModularRobotSimulationState, end_state: ModularRobotSimulationState
) -> float:
    """
    Calculate the distance traveled on the xy-plane by a single modular robot.

    :param begin_state: Begin state of the robot.
    :param end_state: End state of the robot.
    :returns: The calculated fitness.
    """
    begin_position = begin_state.get_pose().position
    end_position = end_state.get_pose().position
    return math.sqrt(
        (begin_position.x - end_position.x) ** 2
        + (begin_position.y - end_position.y) ** 2
    )

def x_displacement(
    begin_state: ModularRobotSimulationState, end_state: ModularRobotSimulationState
) -> float:
    """
    Calculate the distance traveled on the x-axis by a single modular robot.

    :param begin_state: Begin state of the robot.
    :param end_state: End state of the robot.
    :returns: The calculated fitness.
    """
    begin_position = begin_state.get_pose().position
    end_position = end_state.get_pose().position
    return  abs(end_position.x - begin_position.x)

def y_displacement(
    begin_state: ModularRobotSimulationState, end_state: ModularRobotSimulationState
) -> float:
    """
    Calculate the distance traveled on the y-axis by a single modular robot.

    :param begin_state: Begin state of the robot.
    :param end_state: End state of the robot.
    :returns: The calculated fitness.
    """
    begin_position = begin_state.get_pose().position
    end_position = end_state.get_pose().position
    return  abs(end_position.y - begin_position.y)