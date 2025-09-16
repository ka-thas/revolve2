from abc import abstractmethod

import numpy as np
import numpy.typing as npt
import sqlalchemy.orm as orm
from sqlalchemy import event
from sqlalchemy.engine import Connection

import uuid

import config
from revolve2.modular_robot.body.v1 import BodyV1


class BrainGenotype(orm.MappedAsDataclass):
    """
    An SQLAlchemy mixing that provides a distribution column that is a tuple of floats.

    The distribution are saved in the database as string of semicolon seperated floats.
    """
    number_of_parameters_per_active_hinge = 4
    brain: dict[uuid.UUID, npt.NDArray[np.float_]]
    _serialized_brain: orm.Mapped[str] = orm.mapped_column(
        "serialized_brain", init=False, nullable=False
    )

    def __init__(self, brain: dict[uuid.UUID, npt.NDArray[np.float_]]):
        self.brain = brain

    @classmethod
    def initialize_brain(cls, rng) -> 'BrainGenotype':
        number_of_brains = config.CONTROLLERS
        if config.CONTROLLERS == -1:
            number_of_brains = 0

        brain = {}
        for i in range(number_of_brains):
            new_uuid = uuid.UUID(int=i)
            brain[new_uuid] = np.array(rng.random(cls.number_of_parameters_per_active_hinge))

        return BrainGenotype(brain=brain)

    def update_brain_parameters(self, brain_uuids, rng):
        pass

    def add_new(self, rng):
        random_int = (int(rng.integers(0, 2 ** 64, dtype=np.uint64)) << 64) | int(
            rng.integers(0, 2 ** 64, dtype=np.uint64))
        new_uuid = uuid.UUID(int=random_int)
        self.brain[new_uuid] = np.array(rng.random(self.number_of_parameters_per_active_hinge))
        return new_uuid

    def remove_unused(self, used_uuids, rng):
        difference = [item for item in list(self.brain.keys()) if item not in list(used_uuids)]

        for remove_item in difference:
            self.brain.pop(remove_item)

        if len(self.brain.keys()) == 0:
            self.add_new(rng)

    def mutate_brain(self, rng: np.random.Generator):
        brain = BrainGenotype(brain=self.brain.copy())

        for key, value in brain.brain.items():
            noise = rng.normal(loc=0, scale=config.MUTATION_STD, size=len(value))
            noisy_values = [v + n for v, n in zip(value, noise)]
            brain.brain[key] = np.clip(noisy_values, 0, 1)

        return brain

    @classmethod
    def crossover_brain(cls, parent1, parent2, rng):
        child1_brain, child2_brain = BrainGenotype.initialize_brain(rng), BrainGenotype.initialize_brain(rng)

        for i in range(config.CONTROLLERS):
            if rng.random() > 0.5:
                child1_brain.brain[uuid.UUID(int=i)] = parent1.brain[uuid.UUID(int=i)]
                child2_brain.brain[uuid.UUID(int=i)] = parent2.brain[uuid.UUID(int=i)]
            else:
                child1_brain.brain[uuid.UUID(int=i)] = parent2.brain[uuid.UUID(int=i)]
                child2_brain.brain[uuid.UUID(int=i)] = parent1.brain[uuid.UUID(int=i)]
        return child1_brain, child2_brain

    @abstractmethod
    def develop_brain(self, body: BodyV1):
        pass

    @abstractmethod
    def get_p_bounds(self):
        pass

    @abstractmethod
    def get_evolutionary_search_next_point(self):
        pass

    @abstractmethod
    def next_point_to_brain(self, next_point, brain_uuids):
        pass

    @abstractmethod
    def update_values_with_genotype(self, sorted_inherited_experience):
        pass


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
        if value == '':
            continue
        new_uuid, values = value.split(':')
        string_list = values.split(',')
        if values == "":
            target.brain[uuid.UUID(new_uuid)] = np.array([])
            continue
        float_list = [float(value) for value in string_list]
        target.brain[uuid.UUID(new_uuid)] = np.array(float_list)

