""" 
build_body(gene)
    Takes JSON gene and renders in 3D

json needs to be saved in ./genes_warderobe/gene_[i].json
"""
import os
import logging
import pickle
import random
from typing import Any, Dict


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

class Individual:
    """Represents an individual in the population."""
    def __init__(self, gene):
        self.gene = gene  # Build the robot body from the gene
        self.body = None
        self.brain = None
        self.weights = None
        self.num_bricks: int = 0

        self.fitness = -float("inf")
        self.fitness_flat = -float("inf")
        self.fitness_uneven = -float("inf")
        self.fitness_crater = -float("inf")

def count_bricks(gene_dict: Dict[str, Any]) -> int:
        def count_recursive(node):
            if not isinstance(node, dict) or not node:
                return 0

            count = 0
            for key, value in node.items():
                if key in ["front", "left", "right", "back"] and "hinge" in str(value):
                    # count += 1  # Count the hinge
                    if (
                        isinstance(value, dict)
                        and "hinge" in value
                        and "brick" in value["hinge"]
                    ):
                        count += 1  # Count the brick
                        count += count_recursive(value["hinge"]["brick"])
                elif isinstance(value, dict):
                    count += count_recursive(value)

            return count

        return count_recursive(gene_dict.get("core", {}))

def build_body(gene):
    body = BodyV1()
    
    build_body_recursive(body.core_v1, gene["core"])

    return body
    
def build_body_recursive(body, node):
    if not isinstance(node, dict) or not node:  # leaf node
        return
    
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

def save_individual(individual, seed, filepath):
    """Saves the individual to a file."""
    fitness_flat = individual.fitness_flat
    fitness_uneven = individual.fitness_uneven
    fitness_crater = individual.fitness_crater
    fitness_total = individual.fitness

    to_save = {
        "seed": seed,
        "fitness_flat": fitness_flat,
        "fitness_uneven": fitness_uneven,
        "fitness_crater": fitness_crater,
        "fitness_total": fitness_total,
        "core": individual.gene["core"],
        "brain_weights": individual.weights.tolist()
    }

    with open(filepath, "w") as f:
        json.dump(to_save, f, indent=2)

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

    # get gene
    folder = "genes_wardrobe"
    filename = "vebjorn.json"
    
    jsonfile = os.path.join(folder, filename)
    print(f"\nSelected file: {jsonfile}")

    with open(jsonfile, "r") as f:
        gene = json.load(f)

    if config.DEBUGGING: 
        print("\n-----| Gene Structure |-----")
        print_json_gene(gene)

    # get seed
    seed = random.randint(0, 100000)

    # init robot
    rng = make_rng(seed)
    body = build_body(gene) # Renders body into revolve2
    brain = BrainGenotype()
    individual = Individual(gene)
    individual.body = body
    individual.brain = brain
    individual.num_bricks = count_bricks(gene)

    weights = np.array([])
    brain.develop_brain(body=body, rng=rng, weights=weights)

    # Sequential training on terrains
    print("Sequential training on terrains")
    
    # Flat
    brain.improve(body, config.INNER_LOOP_ITERATIONS, rng, terrain=terrains.flat(), fitness=individual.fitness_flat)
    fitness_flat = brain.fitness
    print(f"Flat fitness: {fitness_flat}")

    # Uneven
    brain.improve(body, config.INNER_LOOP_ITERATIONS, rng, terrain=terrains.crater([20.0, 20.0], 0.13, 0.1), fitness=individual.fitness_uneven)
    fitness_uneven = brain.fitness
    print(f"Uneven fitness: {fitness_uneven}")
    
    # Crater
    brain.improve(body, config.INNER_LOOP_ITERATIONS, rng, terrain=terrains.crater([20.0, 20.0], 0.03, 10), fitness=individual.fitness_crater)
    fitness_crater = brain.fitness
    print(f"Crater fitness: {fitness_crater}")

    # Update variables
    fitness_total = fitness_flat + fitness_uneven + fitness_crater
    individual.fitness = fitness_total
    individual.fitness_flat = fitness_flat
    individual.fitness_uneven = fitness_uneven
    individual.fitness_crater = fitness_crater

    # Log this run's fitnesses
    print("fitness_total, fitness_flat, fitness_uneven, fitness_crater")
    print(fitness_total, fitness_flat, fitness_uneven, fitness_crater, sep=", ")
    
    # save_individual(individual, seed, jsonfile)