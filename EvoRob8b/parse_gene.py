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
                rotation = value["hinge"]["rotation"]
                body.left = ActiveHingeV1(rotation)
                body.left.attachment = BrickV1(0.0)
                build_body_recursive(body.left.attachment, node["left"]["hinge"]["brick"])
        
        if key == "right":
            if "hinge" in value.keys():
                rotation = value["hinge"]["rotation"]
                body.right = ActiveHingeV1(rotation)
                body.right.attachment = BrickV1(0.0)
                build_body_recursive(body.right.attachment, node["right"]["hinge"]["brick"])

        if key == "back":
            if "hinge" in value.keys():
                rotation = value["hinge"]["rotation"]
                body.back = ActiveHingeV1(rotation)
                body.back.attachment = BrickV1(0.0)
                build_body_recursive(body.back.attachment, node["back"]["hinge"]["brick"])
    return body




def print_json_gene(node, depth=0):
    """Recursively parses the gene structure and prints it."""
    indent = "  " * depth
    if not isinstance(node, dict) or not node:  # leaf node
        return
    
    for key, value in node.items():
        if not value:
            continue
        if key != "hinge" and key != "brick" and key != "rotation":
            print(f"{indent}- {key}")
        print_json_gene(value, depth + 1)

def load_brain():
    pass

def load_body_and_brain(file):

    content = file.read()

    # Find the index of the last closing brace for the dictionary
    dict_end = content.rindex('}')  # Find the last '}'
    array_start = dict_end + 1  # Everything after that is part of the array

    # Extract the dictionary part and array part
    dict_str = content[:array_start].strip()  # Dictionary portion
    array_str = content[array_start:].strip()  # Array portion

    # Load the dictionary and array as Python objects
    dict_data = json.loads(dict_str)
    array_data = json.loads(array_str)
    id_value = dict_data.pop("id", None)  # Extract and remove 'id' from the dictionary
    gene = dict_data


    print_json_gene(gene)

    return gene, array_data, id


if __name__ == "__main__":

    folders = ["experiments", "genes_wardrobe"]

    print("\n-----| Available folders |-----")
    for i in range(len(folders)):
        print(i, ": ", folders[i])
    foldernr = input("> Select folder [number]: ")
    # default to experiments
    folder = folders[int(foldernr)] if foldernr else "experiments" 

    gene_name = input("> Gene [name]: ")
    # default to final_best_individual
    gene_name = "final_best_individual" if not gene_name else gene_name

    if folder == "genes_wardrobe":
        gene_name = "gene_" + gene_name
    

    with open(f"./{folder}/{gene_name}.json", "r") as f:
        gene = json.load(f)

    if config.DEBUGGING: 
        print("\n-----| Gene Structure |-----")
        print_json_gene(gene)


    body = build_body(gene) # Renders body into revolve2
    rng = make_rng_time_seed()
    brain = BrainGenotype()
    weights = gene.get("brain_weights", [])
    weights = np.array(weights)
    brain.develop_brain(body=body, rng=rng, weights=weights)

    robot = ModularRobot(body, brain)

    input("> ready [press enter]: ")

    # Create a scene.
    scene = ModularRobotScene(terrain=terrains.flat())
    scene.add_robot(robot)

    # Create a simulator.
    simulator = LocalSimulator(headless=True)

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

    print(f"\n->> xy_displacement: {xy_displacement}")

