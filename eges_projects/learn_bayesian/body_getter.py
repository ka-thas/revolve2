import config

from revolve2.experimentation.database import open_database_sqlite, OpenMethod

from database_components.learn_genotype import LearnGenotype
from database_components.learn_individual import LearnIndividual
from database_components.learn_population import LearnPopulation
from database_components.learn_generation import LearnGeneration
from database_components.genotype import Genotype
from database_components.individual import Individual
from database_components.population import Population
from database_components.generation import Generation

from sqlalchemy import select
from sqlalchemy.orm import Session

from revolve2.modular_robot.body import RightAngles
from revolve2.modular_robot.body.v2 import BodyV2, ActiveHingeV2, BrickV2


def get_best_genotype(file_name):
    dbengine = open_database_sqlite(
        file_name, open_method=OpenMethod.OPEN_IF_EXISTS
    )
    with Session(dbengine) as ses:
        row = ses.execute(
            select(Genotype)
            .join_from(LearnGenotype, LearnIndividual, LearnGenotype.id == LearnIndividual.genotype_id)
            .join_from(LearnIndividual, Genotype, LearnIndividual.morphology_genotype_id == Genotype.id)
            .order_by(LearnIndividual.objective_value.desc())
        ).first()
        assert row is not None

    return row[0]


def make_body() -> BodyV2:
    """
    Create a body for the robot.

    :returns: The created body.
    """
    # A modular robot body follows a 'tree' structure.
    # The 'Body' class automatically creates a center 'core'.
    # From here, other modular can be attached.
    # Modules can be attached in a rotated fashion.
    # This can be any angle, although the original design takes into account only multiples of 90 degrees.
    body = BodyV2()
    body.core_v2.left_face.middle_left = ActiveHingeV2(RightAngles.DEG_90)
    body.core_v2.left_face.middle_left.attachment = ActiveHingeV2(RightAngles.DEG_270)
    body.core_v2.left_face.middle_left.attachment.attachment = BrickV2(RightAngles.DEG_180)
    body.core_v2.right_face.middle_right = ActiveHingeV2(RightAngles.DEG_270)
    body.core_v2.right_face.middle_right.attachment = ActiveHingeV2(RightAngles.DEG_90)
    body.core_v2.right_face.middle_right.attachment.attachment = BrickV2(RightAngles.DEG_0)
    # body.core_v2.front_face.middle = ActiveHingeV2(RightAngles.DEG_180)
    # body.core_v2.front_face.middle.attachment = ActiveHingeV2(RightAngles.DEG_0)
    # body.core_v2.front_face.middle.attachment.attachment = BrickV2(RightAngles.DEG_0)

    return body