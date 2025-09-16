import copy
import json
import random
import uuid
from abc import abstractmethod

import numpy as np
import sqlalchemy.orm as orm
from pyrr import Vector3

from brain_genotype import BrainGenotype
import config
from revolve2.modular_robot.body import RightAngles

from revolve2.modular_robot.body.v1 import BodyV1, BrickV1, ActiveHingeV1

from sqlalchemy import event
from sqlalchemy.engine import Connection


class BodyDeveloper:
    queue: list
    grid: list
    body: BodyV1
    def __init__(self, initial_module):
        self.grid = [Vector3([0, 0, 0])]
        self.body = BodyV1()

        initial_module.body_module = self.body.core_v1

        self.queue = []
        self.queue.append((initial_module, initial_module.body_module))


    def develop(self):
        while len(self.queue) > 0:
            current_module, current_body_module = self.queue.pop(0)
            for direction, new_module in current_module.children.items():
                new_body_module = new_module.get_body_module()
                setattr(current_body_module, direction, new_body_module)

                grid_position = self.body.grid_position(new_body_module)
                if grid_position in self.grid or len(self.grid) > config.MAX_NUMBER_OF_MODULES:
                    setattr(current_body_module, direction, None)
                    continue
                self.grid.append(grid_position)

                self.queue.append((new_module, new_body_module))

            if isinstance(current_body_module, ActiveHingeV1):
                BodyDeveloper.develop_hinge(current_module, current_body_module)

    @staticmethod
    def develop_hinge(hinge_module, hinge_body_module):
        hinge_body_module.map_uuid = hinge_module.module_id
        hinge_body_module.reverse_phase = 0


class Module:
    possible_children: dict
    module_id: uuid.UUID
    type: str

    def __init__(self, module_id: uuid.UUID):
        self.module_id = module_id

    def serialize(self):
        return str(self.module_id)


class CoreModule(Module):
    possible_children = [['left'], ['right'], ['front'], ['back']]
    type = 'core'


class BrickModule(Module):
    possible_children = [['left'], ['right'], ['front'], ['up'], ['down']]
    type = 'brick'


class HingeModule(Module):
    possible_children = [['attachment']]
    type = 'hinge'


class Rule:
    start: Module
    end: dict[str, list]

    def __init__(self, start: Module, end: dict[str, list]):
        self.start = start
        self.end = end

    def remove_module(self, module: Module):
        for place, child in self.end.items():
            if child[0] is module:
                self.end[place][0] = None
            elif isinstance(child[0], Rule):
                if child[0].start is module:
                    self.end[place][0] = None
                else:
                    self.end[place][0].remove_module(module)

    def serialize(self):
        serialized = {'start': self.start.serialize(), 'end': {}}
        for place, child in self.end.items():
            serialized['end'][place] = []
            if isinstance(child[0], Module):
                serialized['end'][place].append(child[0].serialize())
            elif isinstance(child[0], Rule):
                serialized['end'][place].append(child[0].serialize())
            else:
                serialized['end'][place].append('None')
            serialized['end'][place].append(child[1])
        return serialized

    def deserialize_end(self, serialized_end, modules):
        for key, value in serialized_end.items():
            if value[0] == 'None':
                self.end[key] = [None, value[1]]
            elif isinstance(value[0], dict):
                self.end[key] = [Rule(modules[value[0]['start']], {}), value[1]]
                self.end[key][0].deserialize_end(value[0]['end'], modules)
            else:
                self.end[key] = [modules[value[0]], value[1]]


class ModuleGenotype:
    rotation: float
    children: dict
    type: str
    module_id: uuid.UUID
    iteration: int

    def __init__(self, rotation, iteration):
        self.rotation = rotation
        self.children = {}
        self.iteration = iteration

    def replace(self, rule: Rule, iteration, check_start=True):
        if (rule.start.module_id == self.module_id and iteration != self.iteration) or not check_start:
            for name, module in rule.end.items():
                if module[0] is None:
                    continue
                elif isinstance(module[0], Module):
                    module_to_check = module[0]
                elif isinstance(module[0], Rule):
                    module_to_check = module[0].start
                else:
                    raise Exception("Oopsiefloopsie")

                if module_to_check.type == 'brick':
                    new_module = BrickGenotype(module[1], iteration)
                else:
                    new_module = HingeGenotype(module[1], iteration)
                new_module.module_id = module_to_check.module_id
                self.children[name] = new_module

                if isinstance(module[0], Rule):
                    new_module.replace(module[0], iteration, False)
        for module in self.children.values():
            module.replace(rule, iteration)


class CoreGenotype(ModuleGenotype):
    type = 'core'


class BrickGenotype(ModuleGenotype):
    type = 'brick'

    def get_body_module(self):
        return BrickV1(self.rotation)



class HingeGenotype(ModuleGenotype):
    type = 'hinge'

    def get_body_module(self):
        return ActiveHingeV1(self.rotation)

class BodyGenotype:
    """SQLAlchemy model for a direct encoding body genotype."""

    first_module: Module | None
    modules: list[Module]
    first_rule: Rule
    rules: dict[uuid.UUID, Rule]
    creation: list[uuid.UUID]

    def __init__(self):
        self.first_module = None
        self.modules = []
        self.rules = {}
        self.creation = []

    @classmethod
    def initialize_body(cls, rng: np.random.Generator, brain: BrainGenotype):
        core = CoreModule(uuid.uuid4())
        brick = BrickModule(uuid.uuid4())
        hinge_uuid = uuid.uuid4()
        hinge = HingeModule(hinge_uuid)
        brain.add_new(hinge_uuid, rng)

        genotype = BodyGenotype()
        genotype.first_module = core
        genotype.modules = [brick, hinge]

        rule1 = genotype.create_rule(core, rng)
        rule2 = genotype.create_rule(brick, rng)
        rule3 = genotype.create_rule(hinge, rng)

        genotype.first_rule = rule1
        genotype.rules[brick.module_id] = rule2
        genotype.rules[hinge.module_id] = rule3
        genotype.creation = [brick.module_id, hinge.module_id]
        return genotype

    def create_rule(self, start, rng):
        end = {}
        for child_pair in start.possible_children:
            module = self.rando_mando(rng)
            rotation = self.choose_rotations(rng)
            for child in child_pair:
                end[child] = [module, rotation.pop()]
        return Rule(start, end)

    def choose_rotations(self, rng):
        return list(rng.choice([
            [RightAngles.DEG_0.value, RightAngles.DEG_180.value],
            [RightAngles.DEG_90.value, RightAngles.DEG_270.value],
            [RightAngles.DEG_180.value, RightAngles.DEG_0.value],
            [RightAngles.DEG_270.value, RightAngles.DEG_90.value]
        ]))

    def rando_mando(self, rng: np.random.Generator):
        if rng.random() > config.MODULE_IN_NEW_SPOT:
            return None

        result = rng.choice(self.modules)
        if rng.random() < config.RULE_IN_NEW_SPOT:
            return self.create_rule(result, rng)

        return result

    def mutate_body(self, rng: np.random.Generator, brain: BrainGenotype):
        genotype = copy.deepcopy(self)

        mutation_chooser = rng.random()

        if mutation_chooser < 0.25:
            genotype.add_new_module(rng, brain)
        elif mutation_chooser < 0.5:
            genotype.delete_module(rng, brain)
        elif mutation_chooser < 0.75:
            genotype.update_rule(rng)
        else:
            rng.shuffle(genotype.creation)

        return genotype

    def add_new_module(self, rng: np.random.Generator, brain: BrainGenotype):
        if random.random() < 0.5:
            new_module = BrickModule(uuid.uuid4())
        else:
            hinge_uuid = uuid.uuid4()
            new_module = HingeModule(hinge_uuid)
            brain.add_new(hinge_uuid, rng)

        self.modules.append(new_module)
        self.add_module_to_existing_rules(new_module, rng)
        self.rules[new_module.module_id] = self.create_rule(new_module, rng)
        self.creation.insert(rng.integers(0, len(self.creation)), new_module.module_id)

    def add_module_to_existing_rules(self, new_module: Module, rng: np.random.Generator):
        for rule_uuid, rule in self.rules.items():
            self.add_module_to_existing_rules_recursive(new_module, rule, rng)
        self.add_module_to_existing_rules_recursive(new_module, self.first_rule, rng)

    def add_module_to_existing_rules_recursive(self, new_module: Module, rule: Rule, rng: np.random.Generator):
        for child_pair in rule.start.possible_children:
            if rng.random() < config.NEW_MODULE_PLACEMENT:
                rotation = self.choose_rotations(rng)
                for child in child_pair:
                    rule.end[child] = [new_module, rotation.pop()]
                continue

            for child in child_pair:
                if isinstance(rule.end[child][0], Rule):
                    self.add_module_to_existing_rules_recursive(new_module, rule.end[child][0], rng)

    def delete_module(self, rng: np.random.Generator, brain: BrainGenotype):
        module_to_delete = self.choose_module_to_delete(rng)

        if module_to_delete is None:
            return

        self.first_rule.remove_module(module_to_delete)
        for rule_uuid, rule in self.rules.items():
            rule.remove_module(module_to_delete)

        self.rules.pop(module_to_delete.module_id)
        self.creation.remove(module_to_delete.module_id)
        self.modules.remove(module_to_delete)

        if module_to_delete.type == 'hinge':
            brain.remove(module_to_delete.module_id)

    def choose_module_to_delete(self, rng: np.random.Generator) -> Module | None:
        hinges = []
        bricks = []
        for module in self.modules:
            if module.type == 'hinge':
                hinges.append(module)
            elif module.type == 'brick':
                bricks.append(module)

        possible_modules = []
        if len(hinges) > 1:
            possible_modules = possible_modules + hinges
        if len(bricks) > 1:
            possible_modules = possible_modules + bricks

        if len(possible_modules) == 0:
            return None

        return rng.choice(possible_modules)

    def update_rule(self, rng: np.random.Generator):
        rules = list(self.rules.values()) + [self.first_rule]
        rule_to_update = rng.choice(rules)

        new_rule = self.create_rule(rule_to_update.start, rng)

        if rule_to_update.start == self.first_rule.start:
            self.first_rule = new_rule
        else:
            self.rules[rule_to_update.start.module_id] = new_rule

    def develop_body(self):
        direct_encoding_body = CoreGenotype(0.0, 0)
        direct_encoding_body.module_id = self.first_module.module_id
        direct_encoding_body.replace(self.first_rule, -1)
        for i in self.creation:
            direct_encoding_body.replace(self.rules[i], i)

        body_developer = BodyDeveloper(direct_encoding_body)
        body_developer.develop()
        return body_developer.body

    def serialize(self):
        serialized = {'first_module': {
            'id': self.first_module.serialize(),
            'type': self.first_module.type
        }, 'first_rule': self.first_rule.serialize(),
            'creation': [str(value) for value in self.creation], 'modules': [], 'rules': {}}

        for module in self.modules:
            serialized['modules'].append(
                {
                    'id': module.serialize(),
                    'type': module.type
                }
            )

        for rule_uuid, rule in self.rules.items():
            serialized['rules'][str(rule_uuid)] = rule.serialize()

        return serialized

    def deserialize(self, serialized):
        self.first_module = CoreModule(uuid.UUID(serialized['first_module']['id']))
        self.modules = []
        module_dict = {serialized['first_module']['id']: self.first_module}
        for module in serialized['modules']:
            if module['type'] == 'hinge':
                new_module = HingeModule(uuid.UUID(module['id']))
            elif module['type'] == 'brick':
                new_module = BrickModule(uuid.UUID(module['id']))
            else:
                raise Exception("Mama mia that is left")
            self.modules.append(new_module)
            module_dict[module['id']] = new_module

        self.first_rule = Rule(module_dict[serialized['first_rule']['start']], {})
        self.first_rule.deserialize_end(serialized['first_rule']['end'], module_dict)
        for module_uuid, rule in serialized['rules'].items():
            self.rules[uuid.UUID(module_uuid)] = Rule(module_dict[rule['start']], {})
            self.rules[uuid.UUID(module_uuid)].deserialize_end(rule['end'], module_dict)

        for rule in serialized['creation']:
            self.creation.append(uuid.UUID(rule))

        return self


class BodyGenotypeGenerative(orm.MappedAsDataclass):

    body: BodyGenotype
    _serialized_body: orm.Mapped[str] = orm.mapped_column(
        "serialized_body", init=False, nullable=False
    )

    @classmethod
    def initialize_body(cls, rng: np.random.Generator, brain: BrainGenotype):
        return BodyGenotype.initialize_body(rng, brain)


@event.listens_for(BodyGenotypeGenerative, "before_update", propagate=True)
@event.listens_for(BodyGenotypeGenerative, "before_insert", propagate=True)
def _update_serialized_body(
        mapper: orm.Mapper[BodyGenotypeGenerative],
        connection: Connection,
        target: BodyGenotypeGenerative,
) -> None:
    target._serialized_body = str(target.body.serialize()).replace("'", "\"")


@event.listens_for(BodyGenotypeGenerative, "load", propagate=True)
def _deserialize_body(target: BodyGenotypeGenerative, context: orm.QueryContext) -> None:
    serialized_dict = json.loads(target._serialized_body)
    target.body = BodyGenotype().deserialize(serialized_dict)
