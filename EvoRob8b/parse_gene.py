""" 
build_body(gene)
    Takes JSON gene and renders in 3D

json needs to be saved in ./genes_warderobe/gene_[i].json
"""

import logging
import pickle
from typing import Any

# import multineat
import numpy as np
import numpy.typing as npt
import json
import config
from brain_cpg import BrainGenotype



from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot_simulation import ModularRobotScene, simulate_scenes
from revolve2.modular_robot.body.v1 import ActiveHingeV1, BodyV1, BrickV1
from revolve2.standards import modular_robots_v1, fitness_functions, terrains
from revolve2.modular_robot.brain.cpg import BrainCpgNetworkNeighborRandom
from revolve2.experimentation.rng import make_rng_time_seed
from revolve2.simulators.mujoco_simulator import LocalSimulator
from revolve2.standards.simulation_parameters import make_standard_batch_parameters

from revolve2.standards import fitness_functions, modular_robots_v1, terrains

def build_body(gene):
    body = BodyV1()
    
    build_body_recursive(body.core_v1, gene["core"])

    return body
    
def build_body_recursive(body, node):
    if not isinstance(node, dict) or not node:  # leaf node
        return

    # TODO Detect collisions
    
    for key, value in node.items():
        if key == "front":
            if "hinge" in value.keys():
                rotation = value["hinge"]["rotation"]
                body.front = ActiveHingeV1(rotation)
                body.front.attachment = BrickV1(0.0)
                build_body_recursive(body.front.attachment, node["front"]["hinge"]["brick"])

        if key == "left":
            if "hinge" in value.keys():
                body.left = ActiveHingeV1(0.0)
                body.left.attachment = BrickV1(0.0)
                build_body_recursive(body.left.attachment, node["left"]["hinge"]["brick"])
        
        if key == "right":
            if "hinge" in value.keys():
                body.right = ActiveHingeV1(0.0)
                body.right.attachment = BrickV1(0.0)
                build_body_recursive(body.right.attachment, node["right"]["hinge"]["brick"])

        if key == "back":
            if "hinge" in value.keys():
                body.back = ActiveHingeV1(0.0)
                body.back.attachment = BrickV1(0.0)
                build_body_recursive(body.back.attachment, node["back"]["hinge"]["brick"])
    return body




def print_json_gene(node, depth=0):
    """Recursively parses the gene structure and prints it."""
    indent = "  " * depth
    if not isinstance(node, dict) or not node:  # leaf node
        return
    
    for key, value in node.items():
        print(f"{indent}- {key}")
        print_json_gene(value, depth + 1)


if __name__ == "__main__":
    
    nr = input("gene nr.")
    with open(f"./genes_wardrobe/gene_{nr}.json", "r") as f:
        gene = json.load(f)
    
    print("Gene Structure:")
    print_json_gene(gene)

    body = build_body(gene) # Most important function here
    
    rng = make_rng_time_seed()
    print(rng)
    brain = BrainGenotype()
    brain.develop_brain(body=body, rng=rng)

    brain.improve(body, config.INNER_LOOP_ITERATIONS, rng)
    config.BRAIN_MUTATION_RATE = 0


    robot = ModularRobot(body, brain)

    input("ready")

    # Create a scene.
    scene = ModularRobotScene(terrain=terrains.flat())
    scene.add_robot(robot)

    # Create a simulator.
    simulator = LocalSimulator(headless=False)

    # Simulate the scene and obtain states sampled during the simulation.
    scene_states = simulate_scenes(
        simulator=simulator,
        batch_parameters=make_standard_batch_parameters(),
        scenes=scene,
    )

    # Get the state at the beginning and end of the simulation.
    scene_state_begin = scene_states[0]
    scene_state_end = scene_states[-1]

    # Retrieve the states of the modular robot.
    robot_state_begin = scene_state_begin.get_modular_robot_simulation_state(robot)
    robot_state_end = scene_state_end.get_modular_robot_simulation_state(robot)

    # Calculate the xy displacement of the robot.
    xy_displacement = fitness_functions.xy_displacement(
        robot_state_begin, robot_state_end
    )

    print(xy_displacement)



    # Test the brain for other bodies
    print(f"Testing new brain for body spider")

    #tester med ny kropp
    spide_body = modular_robots_v1.gecko_v1()
    new_robot = ModularRobot(spide_body, brain)


    new_simulator = LocalSimulator(headless=False, num_simulators=1)
    new_scene = ModularRobotScene(terrain=terrains.flat())

    new_scene.add_robot(new_robot)

    new_scene_states = simulate_scenes(
        simulator = new_simulator,
        batch_parameters = make_standard_batch_parameters(),
        scenes = new_scene,
    )


    scene_state_begin = new_scene_states[0]
    scene_state_end = new_scene_states[-1]

    # Retrieve the states of the modular robot.
    robot_state_begin = scene_state_begin.get_modular_robot_simulation_state(robot)
    robot_state_end = scene_state_end.get_modular_robot_simulation_state(robot)

    xy_displacement = fitness_functions.xy_displacement(
        robot_state_begin, robot_state_end
    )

    print(xy_displacement)