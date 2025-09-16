import math
import uuid

import numpy as np
from sine_brain_simple import SineBrain
from revolve2.modular_robot.body.base import ActiveHinge
from revolve2.modular_robot.body.v1 import BodyV1
from genotypes.brain_genotype import BrainGenotype as AbstractBrainGenotype


class BrainGenotype(AbstractBrainGenotype):

    @staticmethod
    def sin_cos_to_angle(sin_val, cos_val):
        angle = np.arctan2(sin_val, cos_val)
        if angle < 0:
            angle += 2 * np.pi  # Adjust to the range [0, 2π]
        return angle

    def develop_brain(self, body: BodyV1):
        active_hinges = body.find_modules_of_type(ActiveHinge)

        amplitudes = []
        phases = []
        offsets = []
        for active_hinge in active_hinges:
            amplitudes.append(self.brain[active_hinge.map_uuid][0])
            phases.append(BrainGenotype.sin_cos_to_angle(
                self.brain[active_hinge.map_uuid][1] * 2 - 1,
                self.brain[active_hinge.map_uuid][2] * 2 - 1
            ))
            offsets.append(self.brain[active_hinge.map_uuid][3] - 0.5)

        brain = SineBrain(
            active_hinges=active_hinges,
            amplitudes=amplitudes,
            phases=phases,
            offsets=offsets
        )

        return brain

    def get_p_bounds(self):
        brain_uuids = list(self.brain.keys())
        pbounds = {}
        for key in brain_uuids:
            pbounds['amplitude_' + str(key)] = [0, 1]
            pbounds['phase_sin_' + str(key)] = [0, 1]
            pbounds['phase_cos_' + str(key)] = [0, 1]
            pbounds['offset_' + str(key)] = [0, 1]
        return pbounds

    def get_evolutionary_search_next_point(self):
        brain_uuids = list(self.brain.keys())
        next_point = {}
        for key in brain_uuids:
            next_point['amplitude_' + str(key)] = self.brain[key][0]
            next_point['phase_sin_' + str(key)] = self.brain[key][1]
            next_point['phase_cos_' + str(key)] = self.brain[key][2]
            next_point['offset_' + str(key)] = self.brain[key][3]
        return next_point

    def get_random_next_point(self, rng):
        brain_uuids = list(self.brain.keys())
        next_point = {}
        for key in brain_uuids:
            next_point['amplitude_' + str(key)] = rng.random()
            next_point['phase_sin_' + str(key)] = rng.random()
            next_point['phase_cos_' + str(key)] = rng.random()
            next_point['offset_' + str(key)] = rng.random()
        return next_point

    def next_point_to_brain(self, next_point, brain_uuids):
        for brain_uuid in brain_uuids:
            self.brain[brain_uuid] = np.array(
                [
                    next_point['amplitude_' + str(brain_uuid)],
                    next_point['phase_sin_' + str(brain_uuid)],
                    next_point['phase_cos_' + str(brain_uuid)],
                    next_point['offset_' + str(brain_uuid)],
                ]
            )

    def update_values_with_genotype(self, sorted_inherited_experience):
        current_genotype_keys = [str(key) for key in self.brain.keys()]

        for values, _, _ in sorted_inherited_experience:
            keys_to_remove = []
            unique_keys = set()

            for full_key, value in values.items():
                real_key = full_key.split("_")[1]
                unique_keys.add(real_key)
                if real_key not in current_genotype_keys:
                    keys_to_remove.append(full_key)

            for key in keys_to_remove:
                del values[key]

            for genotype_key in current_genotype_keys:
                if genotype_key not in unique_keys:
                    brain_data = self.brain[uuid.UUID(genotype_key)]
                    values[f'amplitude_{genotype_key}'] = brain_data[0]
                    values[f'phase_sin_{genotype_key}'] = brain_data[1]
                    values[f'phase_cos_{genotype_key}'] = brain_data[2]
                    values[f'offset_{genotype_key}'] = brain_data[3]
        return sorted_inherited_experience
