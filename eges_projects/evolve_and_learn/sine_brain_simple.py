import math

import config
from revolve2.modular_robot import ModularRobotControlInterface
from revolve2.modular_robot.body.base import ActiveHinge
from revolve2.modular_robot.brain import BrainInstance, Brain
from revolve2.modular_robot.sensor_state import ModularRobotSensorState


class SineBrainInstance(BrainInstance):
    """ANN brain instance."""

    active_hinges: list[ActiveHinge]
    t: list[float]
    amplitudes: list[float]
    phases: list[float]
    offsets: list[float]
    energy: float

    def __init__(
            self,
            active_hinges: list[ActiveHinge],
            amplitudes: list[float],
            phases: list[float],
            offsets: list[float],
    ) -> None:
        """
        Initialize the Object.

        :param active_hinges: The active hinges to control.
        """
        self.active_hinges = active_hinges
        self.amplitudes = amplitudes
        self.phases = phases
        self.offsets = offsets
        self.t = [0.0] * len(active_hinges)
        self.energy = config.ENERGY

    def control(
            self,
            dt: float,
            sensor_state: ModularRobotSensorState,
            control_interface: ModularRobotControlInterface,
    ) -> None:
        """
        Control the modular robot.

        :param dt: Elapsed seconds since last call to this function.
        :param sensor_state: Interface for reading the current sensor state.
        :param control_interface: Interface for controlling the robot.
        """
        if self.energy < 0:
            return
        i = 0
        for active_hinge, amplitude, phase, offset in zip(self.active_hinges, self.amplitudes, self.phases, self.offsets):
            phase = phase + active_hinge.reverse_phase
            target = amplitude * math.sin(self.t[i] + phase) + offset
            control_interface.set_active_hinge_target(active_hinge, target)
            self.t[i] += dt * config.FREQUENCY
            i += 1

        # self.energy -= control_interface.get_actuator_force()


class SineBrain(Brain):
    """The Sine brain."""

    active_hinges: list[ActiveHinge]
    amplitudes: list[float]
    phases: list[float]
    offsets: list[float]

    def __init__(
        self,
        active_hinges: list[ActiveHinge],
        amplitudes: list[float],
        phases: list[float],
        offsets: list[float],
    ) -> None:
        """
        Initialize the Object.

        :param active_hinges: The active hinges to control.
        """
        self.active_hinges = active_hinges
        self.amplitudes = amplitudes
        self.phases = phases
        self.offsets = offsets

    def make_instance(self) -> BrainInstance:
        """
        Create an instance of this brain.

        :returns: The created instance.
        """
        return SineBrainInstance(
            active_hinges=self.active_hinges,
            amplitudes=self.amplitudes,
            phases=self.phases,
            offsets=self.offsets
        )
