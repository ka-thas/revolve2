import uuid

import numpy as np
from cpg_brain import CpgBrain
from revolve2.modular_robot.body.v1 import BodyV1
from genotypes.brain_genotype import BrainGenotype as AbstractBrainGenotype


class BrainGenotype(AbstractBrainGenotype):

    def develop_brain(self, body: BodyV1):
        cpg_brain = CpgBrain(
            body,
            self.brain
        )
        return cpg_brain

    def get_p_bounds(self):
        brain_uuids = list(self.brain.keys())
        pbounds = {}
        for key in brain_uuids:
            pbounds['internal_' + str(key)] = [0, 1]
            pbounds['first_neighbour_' + str(key)] = [0, 1]
            pbounds['second_neighbour_' + str(key)] = [0, 1]
        return pbounds

    def get_evolutionary_search_next_point(self):
        brain_uuids = list(self.brain.keys())
        next_point = {}
        for key in brain_uuids:
            next_point['internal_' + str(key)] = self.brain[key][0]
            next_point['first_neighbour_' + str(key)] = self.brain[key][1]
            next_point['second_neighbour_' + str(key)] = self.brain[key][2]
        return next_point

    def next_point_to_brain(self, next_point, brain_uuids):
        for brain_uuid in brain_uuids:
            self.brain[brain_uuid] = np.array(
                [
                    next_point['internal_' + str(brain_uuid)],
                    next_point['first_neighbour_' + str(brain_uuid)],
                    next_point['second_neighbour_' + str(brain_uuid)],
                ]
            )

    def update_values_with_genotype(self, sorted_inherited_experience):
        current_genotype_keys = [str(key) for key in self.brain.keys()]

        for values, _, _ in sorted_inherited_experience:
            keys_to_remove = []
            unique_keys = set()

            for full_key, value in values.items():
                real_key = full_key.split("_")[-1]
                unique_keys.add(real_key)
                if real_key not in current_genotype_keys:
                    keys_to_remove.append(full_key)

            for key in keys_to_remove:
                del values[key]

            for genotype_key in current_genotype_keys:
                if genotype_key not in unique_keys:
                    brain_data = self.brain[uuid.UUID(genotype_key)]
                    values[f'internal_{genotype_key}'] = brain_data[0]
                    values[f'first_neighbour_{genotype_key}'] = brain_data[1]
                    values[f'second_neighbour_{genotype_key}'] = brain_data[2]
        return sorted_inherited_experience