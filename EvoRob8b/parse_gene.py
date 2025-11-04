""" 
build_body(gene)
    Takes JSON gene and renders in 3D

json needs to be saved in ./genes_warderobe/gene_[i].json
"""
import os
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
from revolve2.experimentation.rng import make_rng
from revolve2.simulators.mujoco_simulator import LocalSimulator
from revolve2.standards.simulation_parameters import make_standard_batch_parameters


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
                if "rotation" in value["hinge"]["brick"].keys():
                    brickrotation = value["hinge"]["brick"]["rotation"]
                else:
                    brickrotation = 0.0
                body.front = ActiveHingeV1(rotation)
                body.front.attachment = BrickV1(brickrotation)
                build_body_recursive(body.front.attachment, node["front"]["hinge"]["brick"])

        if key == "left":
            if "hinge" in value.keys():
                rotation = value["hinge"]["rotation"]
                if "rotation" in value["hinge"]["brick"].keys():
                    brickrotation = value["hinge"]["brick"]["rotation"]
                else:
                    brickrotation = 0.0
                body.left = ActiveHingeV1(rotation)
                body.left.attachment = BrickV1(brickrotation)
                build_body_recursive(body.left.attachment, node["left"]["hinge"]["brick"])
        
        if key == "right":
            if "hinge" in value.keys():
                rotation = value["hinge"]["rotation"]
                if "rotation" in value["hinge"]["brick"].keys():
                    brickrotation = value["hinge"]["brick"]["rotation"]
                else:
                    brickrotation = 0.0
                body.right = ActiveHingeV1(rotation)
                body.right.attachment = BrickV1(brickrotation)
                build_body_recursive(body.right.attachment, node["right"]["hinge"]["brick"])

        if key == "back":
            if "hinge" in value.keys():
                rotation = value["hinge"]["rotation"]
                if "rotation" in value["hinge"]["brick"].keys():
                    brickrotation = value["hinge"]["brick"]["rotation"]
                else:
                    brickrotation = 0.0
                body.back = ActiveHingeV1(rotation)
                body.back.attachment = BrickV1(brickrotation)
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



def json_file_select():
    folders = ["experiments", "genes_wardrobe"]

    print("\n-----| Available folders |-----")
    for i, f in enumerate(folders):
        print(f"{i}: {f}")

    # Select folder
    foldernr = input("> Select folder [number, default=0]: ").strip()
    folder = folders[int(foldernr)] if foldernr.isdigit() and int(foldernr) < len(folders) else "experiments"

    if folder == "experiments":
        folder = os.path.join(folder, input("> Enter experiment id [xxxxxx]: ").strip())

    # List JSON files in the folder
    files = [f for f in os.listdir(folder) if f.endswith(".json")]
    if not files:
        print(f"\n(no JSON files found in '{folder}')")
        return None

    print(f"\n-----| JSON files in '{folder}' |-----")
    for i, f in enumerate(files):
        print(f"{i}: {f}")

    # Select file by name or index
    selection = input("> Select file [number or name, default=0]: ").strip()
    if selection.isdigit() and int(selection) < len(files):
        filename = files[int(selection)]
    elif selection and selection in files:
        filename = selection
    else:
        filename = files[0]

    jsonfile = os.path.join(folder, filename)
    print(f"\nSelected file: {jsonfile}")
    return jsonfile

def run(robot, terrain):

    # Create a scene.
    scene = ModularRobotScene(terrain=terrain)
    scene.add_robot(robot)

    # Create a simulator.
    simulator = LocalSimulator(headless=headless)

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
    return xy_displacement


if __name__ == "__main__":

    jsonfile = json_file_select()

    with open(jsonfile, "r") as f:
        gene = json.load(f)

    if config.DEBUGGING: 
        print("\n-----| Gene Structure |-----")
        print_json_gene(gene)

    if "runID" in gene.keys():
        seed = int(gene["runID"])
        print(f"Using seed from gene: {seed}")
    else:
        seed = int(input("> Enter seed [int]: "))

    rng = make_rng(seed)
    body = build_body(gene) # Renders body into revolve2
    brain_flat = BrainGenotype()
    brain_uneven = BrainGenotype()
    brain_crater = BrainGenotype()

    weights_flat = gene.get("brain_weights_flat", [])
    weights_flat = np.array(weights_flat)
    brain_flat.develop_brain(body=body, rng=rng, weights=weights_flat)

    weights_uneven = gene.get("brain_weights_uneven", [])
    weights_uneven = np.array(weights_uneven)
    brain_uneven.develop_brain(body=body, rng=rng, weights=weights_uneven)

    weights_crater = gene.get("brain_weights_crater", [])
    weights_crater = np.array(weights_crater)
    brain_crater.develop_brain(body=body, rng=rng, weights=weights_crater)

    robot_flat = ModularRobot(body, brain_flat)
    robot_uneven = ModularRobot(body, brain_uneven)
    robot_crater = ModularRobot(body, brain_crater)

    headless_s =     input("> Headless? [y/ n]: ")
    if (headless_s == "y"):
        headless = True
    else:        
        headless = False



    input("> ready [press enter]: ")


    xy_displacement_flat = run(robot_flat, terrains.flat())
    xy_displacement_uneven = run(robot_flat, terrain=terrains.crater([20.0, 20.0], 0.1, 0.1))
    xy_displacement_crater = run(robot_crater,  terrain=terrains.crater([20.0, 20.0], 0.03, 10))

    print(f"\n->> xy displacement flat: {xy_displacement_flat}")
    print(f"\n->> xy displacement uneven: {xy_displacement_uneven}")
    print(f"\n->> xy displacement crater: {xy_displacement_crater}")


