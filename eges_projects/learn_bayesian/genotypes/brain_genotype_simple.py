import math

import numpy as np
import numpy.typing as npt
import sqlalchemy.orm as orm
from sqlalchemy import event
from sqlalchemy.engine import Connection

import uuid

from sine_brain import SineBrain
import config
from revolve2.modular_robot.body.base import ActiveHinge
from revolve2.modular_robot.body.v1 import BodyV1


class BrainGenotype(orm.MappedAsDataclass):
    """
    An SQLAlchemy mixing that provides a distribution column that is a tuple of floats.

    The distribution are saved in the database as string of semicolon seperated floats.
    """

    brain: dict[uuid.UUID, npt.NDArray[np.float_]]
    _serialized_brain: orm.Mapped[str] = orm.mapped_column(
        "serialized_brain", init=False, nullable=False
    )

    def __init__(self, brain: dict[uuid.UUID, npt.NDArray[np.float_]]):
        self.brain = brain

    def develop_brain(self, body: BodyV1):
        active_hinges = body.find_modules_of_type(ActiveHinge)

        amplitudes = []
        phases = []
        offsets = []
        for active_hinge in active_hinges:
            amplitudes.append(self.brain[active_hinge.map_uuid][0])
            phases.append(self.brain[active_hinge.map_uuid][1] * 2 * math.pi)
            offsets.append(self.brain[active_hinge.map_uuid][2] - 0.5)

        brain = SineBrain(
            active_hinges=active_hinges,
            amplitudes=amplitudes,
            phases=phases,
            offsets=offsets
        )

        return brain


@event.listens_for(BrainGenotype, "before_update", propagate=True)
@event.listens_for(BrainGenotype, "before_insert", propagate=True)
def _update_serialized_brain(
        mapper: orm.Mapper[BrainGenotype],
        connection: Connection,
        target: BrainGenotype,
) -> None:
    target._serialized_brain = ''

    for hinge_uuid, values in target.brain.items():
        target._serialized_brain += str(hinge_uuid) + ":" + ','.join(map(str, values)) + ";"

    target._serialized_brain = target._serialized_brain[:-1]


@event.listens_for(BrainGenotype, "load", propagate=True)
def _deserialize_brain(target: BrainGenotype, context: orm.QueryContext) -> None:
    target.brain = {}
    for value in target._serialized_brain.split(';'):
        new_uuid, values = value.split(':')
        string_list = values.split(',')
        if values == "":
            target.brain[uuid.UUID(new_uuid)] = np.array([])
            continue
        float_list = [float(value) for value in string_list]
        target.brain[uuid.UUID(new_uuid)] = np.array(float_list)

