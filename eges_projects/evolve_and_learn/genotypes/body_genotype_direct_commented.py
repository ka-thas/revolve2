import copy
import math
import uuid
from pyrr import Vector3

import numpy as np
import sqlalchemy.orm as orm
import json
from sqlalchemy import event
from sqlalchemy.engine import Connection

from genotypes.body_genotype import BodyGenotype
from genotypes.brain_genotype import BrainGenotype
import config
from revolve2.modular_robot.body import RightAngles
from revolve2.modular_robot.body.v1 import BodyV1, ActiveHingeV1, BrickV1


class BodyDeveloper:
    """
    Helper that takes a CoreGenotype (the root module genotype) and
    constructs a runtime BodyV1 object by traversing the genotype tree.

    It maintains a queue of modules to process and a `grid` of occupied
    positions to avoid overlapping modules. It also handles mirroring
    logic and hinge-specific initialization.
    """
    queue: list
    grid: list
    body: BodyV1

    # direction-mapping dicts used to transform directions when mirroring
    reverse_direction = {
        'left': 'right',
        'up': 'up',
        'front': 'front',
        'right': 'left',
        'down': 'down',
        'back': 'back',
        'attachment': 'attachment'
    }
    # funky mapping used to complicate orientation for certain mirrored cases
    to_make_it_even_more_complicated = {
        'left': 'left',
        'up': 'down',
        'front': 'front',
        'right': 'right',
        'down': 'up',
        'back': 'back',
        'attachment': 'attachment'
    }
    # mapping used when geometry isn't straight (used for central modules)
    not_straight_direction = {
        'left': 'up',
        'up': 'left',
        'front': 'front',
        'right': 'down',
        'down': 'right',
        'back': 'back',
        'attachment': 'attachment'
    }

    def __init__(self, initial_module):
        # grid stores Vector3 positions already occupied by modules (prevents overlap)
        self.grid = [Vector3([0, 0, 0])]
        self.body = BodyV1()

        # attach the root genotype's body module to the body's core
        initial_module.body_module = self.body.core_v1

        # queue holds tuples: (module_genotype, body_module, mirrored_bool, straight_bool)
        self.queue = []
        self.queue.append((initial_module, initial_module.body_module, False, True))


    def develop(self):
        """
        Breadth-first traversal of the genotype tree; for each child genotype:
         - compute the new body module via get_body_module()
         - determine the correct direction considering mirroring/central logic
         - attach the new body module to the parent's body module
         - check the spatial grid for overlaps or exceeding max modules
         - enqueue child for further processing
         - handle mirror flip for the next sibling
        """
        while len(self.queue) > 0:
            current_module, current_body_module, module_mirror, straight = self.queue.pop(0)

            # iterate over all possible child direction-sets and corresponding genotypes
            for directions, new_module in current_module.children.items():
                direction_mirror = module_mirror
                # a directions value may contain multiple directions (like ['front','back'])
                for direction in directions:
                    # create the body module for the child (rotation may be reversed)
                    new_body_module = new_module.get_body_module(direction_mirror)

                    # determine correct direction to attach to parent body module
                    correct_direction = direction
                    if current_module.central and not straight:
                        # central modules that aren't straight use a different mapping
                        correct_direction = self.not_straight_direction[direction]
                    elif module_mirror and not straight:
                        # mirrored and not straight — use the "complicated" mapping
                        correct_direction = self.to_make_it_even_more_complicated[direction]
                    elif module_mirror:
                        # mirrored straight case — use reverse_direction mapping
                        correct_direction = self.reverse_direction[correct_direction]

                    # attach the child body module to the parent in the chosen direction
                    setattr(current_body_module, correct_direction, new_body_module)

                    # compute grid (world) position of the new module to detect collisions
                    grid_position = self.body.grid_position(new_body_module)

                    # if the position is already occupied or we exceeded max modules, undo attach
                    if grid_position in self.grid or len(self.grid) > config.MAX_NUMBER_OF_MODULES:
                        setattr(current_body_module, correct_direction, None)
                        continue

                    # otherwise mark that position as occupied
                    self.grid.append(grid_position)

                    # compute new 'straight' flag for the child:
                    # if child rotation is 0 or child not central -> maintain straight flag,
                    # otherwise flip straight. There's also a special case for 'down' from core.
                    new_straight = straight if new_module.rotation == 0.0 or not new_module.central else not straight
                    new_straight = not new_straight if current_module.type == 'core' and direction == 'down' else new_straight

                    # enqueue the child for processing; direction_mirror will flip after each direction
                    self.queue.append((new_module, new_body_module, direction_mirror, new_straight))

                    # next sibling direction uses flipped mirror
                    direction_mirror = not direction_mirror

            # if the body module is an active hinge, call static hinge setup
            if isinstance(current_body_module, ActiveHingeV1):
                BodyDeveloper.develop_hinge(current_module, current_body_module, module_mirror)

    @staticmethod
    def develop_hinge(hinge_module, hinge_body_module, mirror):
        """
        Configure hinge-specific runtime fields:
         - map_uuid is used to associate hinge with a brain/actuator map
         - reverse_phase controls phase reversal for hinge actuation when mirrored
        """
        hinge_body_module.map_uuid = hinge_module.brain_index
        if mirror:
            hinge_body_module.reverse_phase = hinge_module.reverse_phase_value
        else:
            hinge_body_module.reverse_phase = 0


class ModuleGenotype:
    """
    Base genotype class for a module (brick or hinge). Handles tree structure,
    random addition/removal of submodules, serialization/deserialization, and
    some traversal utilities.
    """
    rotation: float
    temp_rotation: float
    children: dict
    possible_children: list
    type: str
    body_module = None
    central = 0

    # mapping used to reverse rotation when mirrored (RightAngles constants -> values)
    reverse_rotation = {
        RightAngles.DEG_0.value: 0.0,
        RightAngles.DEG_90.value: RightAngles.DEG_270.value,
        RightAngles.DEG_180.value: RightAngles.DEG_180.value,
        RightAngles.DEG_270.value: RightAngles.DEG_90.value
    }

    def __init__(self, rotation):
        self.rotation = rotation
        self.children = {}

    def get_body_module(self, reverse):
        """
        Prepare and return the runtime body module for this genotype.
        Subclasses override to construct specific BodyV1 module objects.
        This base function sets temp_rotation based on mirroring.
        """
        self.temp_rotation = self.rotation
        if reverse:
            self.temp_rotation = self.reverse_rotation[self.rotation]
        # Note: subclasses should create the body object and return it.

    def add_random_module_to_connection(self, index: int, rng: np.random.Generator, brain: BrainGenotype):
        """
        Recursively walk possible child connections and add a randomly chosen
        module at the specified (1-based) index. Returns 0 when insertion done,
        otherwise returns the decremented index to continue recursion.
        """
        for directions in self.get_possible_children():
            if index == 1:
                # place a new module here
                module_to_add = self.choose_random_module(rng, brain)
                # preserve central-ness when adding to single-direction central slots
                if self.central and len(directions) == 1:
                    module_to_add.central = True

                existing_module = None
                if tuple(directions) in list(self.children.keys()):
                    existing_module = self.children[tuple(directions)]
                self.children[tuple(directions)] = module_to_add

                # if something already existed, reattach it under attachment/front depending on type
                if existing_module is not None:
                    if isinstance(module_to_add, HingeGenotype):
                        module_to_add.children[tuple(['attachment'])] = existing_module
                    if isinstance(module_to_add, BrickGenotype):
                        module_to_add.children[tuple(['front'])] = existing_module

                return 0
            index -= 1
            # recurse into existing child if present
            if tuple(directions) in self.children.keys():
                index = self.children[tuple(directions)].add_random_module_to_connection(index, rng, brain)
            if index == 0:
                return 0
        return index

    def get_amount_possible_connections(self):
        """Count the total number of possible (empty or filled) connection slots in this subtree."""
        possible_connections = 0
        for directions in self.get_possible_children():
            possible_connections += 1
            if tuple(directions) in self.children.keys():
                possible_connections += self.children[tuple(directions)].get_amount_possible_connections()
        return possible_connections

    def get_amount_nodes(self):
        """Return number of nodes in this subtree (counts genotypes, not directional multiplicity)."""
        nodes = 1
        for module in self.children.values():
            nodes += module.get_amount_nodes()
        return nodes

    def get_amount_modules(self):
        """
        Return a count of modules weighted by the number of directions each child
        occupies. This is used when controlling population size by module count.
        """
        nodes = 1
        for directions, module in self.children.items():
            nodes += module.get_amount_modules() * len(directions)
        return nodes

    def get_amount_hinges(self):
        """Return number of hinge modules in this subtree (counts multiplicity of direction sets)."""
        nodes = 0
        for directions, module in self.children.items():
            nodes += module.get_amount_hinges() * len(directions)
        return nodes

    def add_random_module_to_random_connection(self, rng: np.random.Generator, brain: BrainGenotype):
        """
        Choose one of the possible child directions at random and add a random module there.
        If the chosen slot is occupied, recurse into that child to continue.
        """
        direction_chooser = rng.choice(range(len(self.get_possible_children())))
        chosen_direction = self.get_possible_children()[direction_chooser]
        if tuple(chosen_direction) in self.children.keys():
            self.children[tuple(chosen_direction)].add_random_module_to_random_connection(rng, brain)
            return
        module_to_add = self.choose_random_module(rng, brain)
        if self.central and len(chosen_direction) == 1:
            module_to_add.central = True
        self.children[tuple(chosen_direction)] = module_to_add

    def remove_node(self, index):
        """
        Remove the node at the given 1-based index in a pre-order traversal.
        When removing, reattach the removed node's children where possible
        to keep the structure valid.
        """
        for direction, module in self.children.items():
            if index == 1:
                # pop this child and try to reattach its children into parent's possible slots
                child = self.children.pop(direction)

                for child_direction in child.children.keys():
                    temp_key = list(child_direction)

                    # special-case mappings to maintain consistent connection semantics
                    if self.type == 'hinge' and list(child_direction) == ['front']:
                        temp_key = ['attachment']
                    elif self.type in ['core', 'brick'] and list(child_direction) == ['attachment']:
                        temp_key = list(direction)
                    elif self.type == 'core' and list(child_direction) in [['front'], ['back']]:
                        temp_key = ['front', 'back']

                    if temp_key in self.get_possible_children() and tuple(temp_key) not in self.children.keys():
                        self.children[tuple(temp_key)] = child.children[child_direction]

                return 0
            index = module.remove_node(index - 1)
            if index == 0:
                return 0
        return index

    def check_for_brains(self):
        """
        Collect UUIDs (or identifiers) of brains used in this subtree.
        Used to find and remove unused brains from the global BrainGenotype.
        """
        uuids = []
        for module in self.children.values():
            recursive_uuids = module.check_for_brains()
            for recursive_uuid in recursive_uuids:
                if recursive_uuid not in uuids:
                    uuids.append(recursive_uuid)
        return uuids

    def switch_brain(self, rng: np.random.Generator, brain: BrainGenotype):
        """Recursively attempt to switch brain association for hinges inside subtree."""
        for module in self.children.values():
            module.switch_brain(rng, brain)

    def reverse_phase_function(self, value):
        """Propagate a reverse-phase value to hinges in the subtree."""
        for module in self.children.values():
            module.reverse_phase_function(value)

    def serialize(self):
        """Return a JSON-serializable dict representing this module and its children."""
        serialized = {'type': self.type, 'rotation': self.rotation, 'central': int(self.central), 'children': {}}

        for directions, module in self.children.items():
            direction_string = ",".join(directions)
            serialized['children'][direction_string] = module.serialize()

        return serialized

    def deserialize(self, serialized):
        """
        Create a module genotype (and subtree) from a serialized dict.
        Note: this base method will create either a BrickGenotype or HingeGenotype
        for children depending on the 'type' field.
        """
        self.type = serialized['type']
        self.central = False
        if 'central' in serialized.keys():
            self.central = serialized['central']
        for direction, child in serialized['children'].items():
            # direction may be stored as CSV string; convert back to tuple of str
            if isinstance(direction, str):
                direction = tuple(map(str, direction.split(',')))
            if child['type'] == 'brick':
                child_object = BrickGenotype(child['rotation'])
            else:
                child_object = HingeGenotype(child['rotation'])
            # recursively deserialize and attach
            self.children[direction] = child_object.deserialize(child)

        return self

    def choose_random_module(self, rng: np.random.Generator, brain: BrainGenotype):
        """
        Randomly choose between creating a brick (70%) or hinge (30%).
        If a new hinge and config allows new brains, assign a new brain UUID.
        """
        module_chooser = rng.random()

        if module_chooser < 0.7:
            module = BrickGenotype(0.0)
        else:
            module = HingeGenotype(rotation = rng.choice([RightAngles.DEG_0.value, RightAngles.DEG_90.value]))

            new_brain_chooser = rng.random()
            if config.CONTROLLERS == -1 and new_brain_chooser < config.NEW_HINGE_NEW_BRAIN:
                # create a new brain in the BrainGenotype if config allows (-1 indicates dynamic)
                module.brain_index = brain.add_new(rng)
            else:
                # otherwise pick an existing brain at random
                module.brain_index = rng.choice(list(brain.brain.keys()))

        return module

    def get_possible_children(self):
        """Return the list of possible child direction-sets for this module type."""
        return self.possible_children


class CoreGenotype(ModuleGenotype):
    # core-specific possible child slots
    possible_children = [['left'], ['right'], ['front', 'back']]
    type = 'core'
    rotation = 0.0
    possible_phase_differences = [0, math.pi]
    reverse_phase = 0
    central = 1

    def get_amount_nodes(self):
        """For core, node counting excludes the core itself (used by some logic)."""
        nodes = 0
        for module in self.children.values():
            nodes += module.get_amount_nodes()
        return nodes

    def serialize(self):
        """Extend base serialization with mapping for reverse_phase into small ints."""
        serialized = super().serialize()

        phase_difference_to_value = {
            0: 0,
            0.5 * math.pi: 2,
            math.pi: 1,
            1.5 * math.pi: 3,
        }

        serialized['reverse_phase'] = phase_difference_to_value[self.reverse_phase]

        return serialized

    def deserialize(self, serialized):
        """Deserialize and map integer back to reverse_phase radian value."""
        super().deserialize(serialized)

        value_to_phase_difference = {
            0: 0,
            2: 0.5 * math.pi,
            1: math.pi,
            3: 1.5 * math.pi,
        }

        self.reverse_phase = value_to_phase_difference[serialized['reverse_phase']]

        return self


class BrickGenotype(ModuleGenotype):
    """Genotype for a passive brick module."""
    type = 'brick'
    central = 0

    def get_body_module(self, reverse):
        # set temp_rotation via base and then create a BrickV1 runtime object
        super().get_body_module(reverse)
        self.body_module = BrickV1(self.temp_rotation)
        return self.body_module

    def get_possible_children(self):
        # Central bricks have a special slot set
        if self.central:
            return [['left', 'right'], ['front'], ['up'] , ['down']]
        else:
            return [['left'], ['right'], ['front'], ['up'], ['down']]


class HingeGenotype(ModuleGenotype):
    """Genotype for an active hinge (actuated) module."""
    possible_children = [['attachment']]
    brain_index = -1  # UUID or identifier for which brain controls this hinge
    reverse_phase_value = 0
    type = 'hinge'

    def get_body_module(self, reverse):
        # create an ActiveHingeV1 runtime object with the computed temp_rotation
        super().get_body_module(reverse)
        self.body_module = ActiveHingeV1(self.temp_rotation)
        return self.body_module

    def check_for_brains(self):
        # collect all child brain UUIDs and include our own brain_index
        uuids = super().check_for_brains()
        if self.brain_index not in uuids:
            uuids.append(self.brain_index)
        return uuids

    def switch_brain(self, rng: np.random.Generator, brain: BrainGenotype):
        """
        Potentially switch this hinge's brain to a different existing brain
        depending on config.SWITCH_BRAIN probability.
        """
        if rng.random() > config.SWITCH_BRAIN:
            self.brain_index = rng.choice(list(brain.brain.keys()))

        # recurse to children
        super().switch_brain(rng, brain)

    def reverse_phase_function(self, value):
        # set local reverse-phase value and propagate to children
        self.reverse_phase_value = value
        super().reverse_phase_function(value)

    def serialize(self):
        serialized = super().serialize()
        # store brain_index as string to ensure JSON serializable (UUID -> str)
        serialized['brain_index'] = str(self.brain_index)
        return serialized

    def deserialize(self, serialized):
        super().deserialize(serialized)
        # parse the stored string back into a UUID
        self.brain_index = uuid.UUID(serialized['brain_index'])
        return self

    def get_amount_hinges(self):
        """Count hinges in subtree; this hinge contributes 1 plus children's counts."""
        nodes = 1
        for module in self.children.values():
            nodes += module.get_amount_hinges()
        return nodes


class BodyGenotypeDirect(orm.MappedAsDataclass, BodyGenotype):
    """SQLAlchemy dataclass model for a direct-encoding body genotype."""
    body: CoreGenotype

    # backing column in DB storing serialized body as JSON string
    _serialized_body: orm.Mapped[str] = orm.mapped_column(
        "serialized_body", init=False, nullable=False
    )

    def __init__(self, body: CoreGenotype):
        self.body = body

    @classmethod
    def initialize_body(cls, rng: np.random.Generator, brain: BrainGenotype):
        """
        Create a random initial core body by repeatedly adding random modules until
        a target number of modules (random between INIT_MIN_MODULES and INIT_MAX_MODULES)
        is reached.
        """
        number_of_modules = rng.integers(config.INIT_MIN_MODULES, config.INIT_MAX_MODULES)
        body = CoreGenotype(0.0)
        current_number_of_modules = 0
        while current_number_of_modules < number_of_modules:
            # choose a random empty connection and add a module there
            body.add_random_module_to_random_connection(rng, brain)
            current_number_of_modules = body.get_amount_modules()

        # pick a random reverse_phase for the core (used to invert hinge phases)
        body.reverse_phase = rng.choice(body.possible_phase_differences)

        return BodyGenotypeDirect(body)

    def mutate_body_start(self, rng: np.random.Generator, brain: BrainGenotype):
        """
        Perform mutation attempts on a deep-copied body until a mutation is accepted.
        Returns the mutated BodyGenotypeDirect and the chooser value that selected the mutation type.
        Mutation types include:
         - add modules (45% chance)
         - remove modules (45% chance)
         - change reverse_phase
         - (else) switch brain assignments
        """
        body = None
        mutation_chooser = 0
        mutation_accepted = False
        while not mutation_accepted:
            body = copy.deepcopy(self.body)
            mutation_chooser = rng.random()

            if mutation_chooser < 0.45:
                # add between 1 and MAX_ADD_MODULES modules
                for _ in range(rng.integers(1, config.MAX_ADD_MODULES + 1)):
                    body.add_random_module_to_random_connection(rng, brain)
                # accept only if body size remains reasonable (< 110% of max)
                mutation_accepted = body.get_amount_modules() < config.MAX_NUMBER_OF_MODULES * 1.1
            elif mutation_chooser <= 0.9:
                # delete between 1 and MAX_DELETE_MODULES nodes
                for _ in range(rng.integers(1, config.MAX_DELETE_MODULES + 1)):
                    amount_nodes = body.get_amount_nodes()
                    if amount_nodes == 0:
                        break
                    node_to_remove = rng.integers(1, amount_nodes + 1)
                    body.remove_node(node_to_remove)

                # if brains are dynamic, remove unused ones
                if config.CONTROLLERS == -1:
                    used_brains = body.check_for_brains()
                    brain.remove_unused(used_brains, rng)
                mutation_accepted = True
            elif mutation_chooser <= 1:
                # flip reverse phase to a different allowed value
                new_phase = body.reverse_phase
                while new_phase == body.reverse_phase:
                    new_phase = rng.choice(body.possible_phase_differences)
                body.reverse_phase = new_phase
                mutation_accepted = True
            else:
                # fallback: switch brain assignments in subtree
                body.switch_brain(rng, brain)

                if config.CONTROLLERS == -1:
                    used_brains = body.check_for_brains()
                    brain.remove_unused(used_brains, rng)
                mutation_accepted = True
        return BodyGenotypeDirect(body), mutation_chooser

    def get_brain_uuids(self):
        """Return the list of brain UUIDs used in this body genotype."""
        return self.body.check_for_brains()

    @staticmethod
    def crossover_body(parent1: 'BodyGenotypeDirect', parent2: 'BodyGenotypeDirect', rng: np.random.Generator):
        """
        Simple subtree crossover: choose a random branch from each parent's core children
        and swap them. If a parent has no children, return copies unchanged.
        """
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)

        if len(parent1.body.children) == 0 or len(parent2.body.children) == 0:
            return child1, child2

        parent_1_branch_chooser = rng.choice(list(parent1.body.children))
        parent_2_branch_chooser = rng.choice(list(parent2.body.children))

        # swap subtrees between children
        child1.body.children[parent_1_branch_chooser] = copy.deepcopy(parent2.body.children[parent_2_branch_chooser])
        child2.body.children[parent_2_branch_chooser] = copy.deepcopy(parent1.body.children[parent_1_branch_chooser])

        # randomize rotation of the swapped-in branches to add variation
        child1.body.children[parent_1_branch_chooser].rotation = (
            rng.choice([RightAngles.DEG_0.value, RightAngles.DEG_90.value, RightAngles.DEG_180.value,
                        RightAngles.DEG_270.value]))
        child2.body.children[parent_2_branch_chooser].rotation = (
            rng.choice([RightAngles.DEG_0.value, RightAngles.DEG_90.value, RightAngles.DEG_180.value,
                        RightAngles.DEG_270.value]))

        return child1, child2


    def develop_body(self):
        """
        Convert the genotype tree into a runtime BodyV1 instance.
        If REVERSE_PHASE is enabled, propagate phase reversal information first.
        Returns the built BodyV1.
        """
        if config.REVERSE_PHASE:
            self.body.reverse_phase_function(self.body.reverse_phase)

        body_developer = BodyDeveloper(self.body)
        body_developer.develop()
        return body_developer.body


# SQLAlchemy event listeners --------------------------------------------------

@event.listens_for(BodyGenotypeDirect, "before_update", propagate=True)
@event.listens_for(BodyGenotypeDirect, "before_insert", propagate=True)
def _update_serialized_body(
        mapper: orm.Mapper[BodyGenotypeDirect],
        connection: Connection,
        target: BodyGenotypeDirect,
) -> None:
    """
    Before saving to DB, serialize the body to a JSON string and store it in _serialized_body.
    We stringify using Python's dict -> str then replace single quotes to double quotes
    so the DB column gets valid JSON. (Could also use json.dumps for clarity.)
    """
    target._serialized_body = str(target.body.serialize()).replace("'", "\"")


@event.listens_for(BodyGenotypeDirect, "load", propagate=True)
def _deserialize_body(target: BodyGenotypeDirect, context: orm.QueryContext) -> None:
    """
    When loading from DB, parse the JSON string and reconstruct the CoreGenotype subtree.
    """
    serialized_dict = json.loads(target._serialized_body)
    target.body = CoreGenotype(0.0).deserialize(serialized_dict)