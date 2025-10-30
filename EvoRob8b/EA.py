"""
Evolutionary Algorithm for JSON gene representation of modular robots.
This EA handles evolution of robot body structures represented as nested JSON.
"""

import json
import os
import random
import copy
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import time
from dataclasses import dataclass
import logging

from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot_simulation import ModularRobotScene, simulate_scenes
from revolve2.modular_robot.brain.cpg import BrainCpgNetworkNeighborRandom
from revolve2.experimentation.rng import make_rng
from revolve2.simulators.mujoco_simulator import LocalSimulator
from revolve2.standards.simulation_parameters import make_standard_batch_parameters
from revolve2.standards import fitness_functions, terrains

import config
from parse_gene import build_body, print_json_gene
from gene_generator import Gene_Generator
from brain_cpg import BrainGenotype
from plotter import Plotter


@dataclass
class Individual:
    """Represents an individual in the population."""
    def __init__(self, gene):
        self.gene = gene  # Build the robot body from the gene
        self.body = None
        self.brain = None
        self.weights = None
        self.fitness: float = -float("inf")
        self.num_bricks: int = 0


class JSONGeneEA:
    """Evolutionary Algorithm for JSON gene representation."""

    def __init__(self, population_size: int = None, offspring_size: int = None):
        """Initialize the EA with configuration parameters."""
        self.population_size = population_size or config.POPULATION_SIZE
        self.offspring_size = offspring_size or config.OFFSPRING_SIZE
        self.max_modules = config.MAX_BRICKS
        self.tournament_size = config.PARENT_TOURNAMENT_SIZE
        self.function_evaluations = config.FUNCTION_EVALUATIONS
        self.mutation_rate = config.MUTATION_RATE
        self.population: List[Individual] = []
        self.generation = 0
        self.evaluations = 0

        # Initialize random number generator
        self.rng = make_rng(config.SEED)
        self.generator = Gene_Generator(self.rng)
        i = 0
        while True:
            self.runID = str(i)  # To not overwrite logs
            self.runID = self.runID.zfill(6)
            self.log_folder = config.LOG_FOLDER + f"{self.runID}/"
            try:
                os.makedirs(self.log_folder)
                break
            except FileExistsError:
                i += 1
                continue

        if config.VERBOSE_PRINTS:
            print(f"Logging to folder: {self.log_folder} with runID: {self.runID}")

        self.start_time = None
        self.plotter = Plotter(
            filename=self.log_folder + "progress.csv", runID=self.runID
        )

        # Setup logging
        logging.basicConfig(level=logging.WARNING)
        self.logger = logging.getLogger(__name__)

    def write_run_info(self) -> None:
        """Write run configuration info to a log file."""
        filename = self.log_folder + "run_info.txt"
        with open(filename, "w") as f:
            with open("config.py", "r") as config_file:
                f.write(f"Run ID: {self.runID}\n")
                f.write(
                    f"Start Time: {time.strftime('%Y-%m-%d %H:%M', time.localtime())}\n"
                )
                f.write("----- config.py -----\n")
                f.write(config_file.read())
                f.write("\n")

    def debug_dump(self, gene: Dict[str, Any], header: str) -> None:
        filename = self.log_folder + "debug.txt"
        with open(filename, "w") as f:
            f.write(f"{header}\n")
            f.write(json.dumps(gene, indent=4))

    def initialize_population(self) -> None:
        """Initialize the population with random individuals."""
        self.logger.info(f"Initializing population of size {self.population_size}")

        for i in range(self.population_size):
            # Generate random gene
            # gene_generator was refactored into a class - use it here
            gene_dict = self.generator.make_core()

            individual = Individual(gene=gene_dict)
            self.population.append(individual)

        self.logger.info("Population initialized successfully")

    def count_modules(self, gene_dict: Dict[str, Any]) -> int:
        """Count the total number of modules in a gene."""

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

    def evaluate_individual(self, individual: Individual) -> float:
        """Evaluate the fitness of an individual."""
        try:

            individual.body = build_body(individual.gene)
            body = individual.body

            rng = self.rng
            # Create brain
            brain = BrainGenotype()

            brain.develop_brain(body, rng=rng)
            brain.improve(body, config.INNER_LOOP_ITERATIONS, rng)

            individual.brain = brain
            individual.weights = individual.brain.get_weights()

            robot = ModularRobot(body, brain)

            # Create scene
            scene = ModularRobotScene(terrain=terrains.flat())
            scene.add_robot(robot)

            # Simulate
            simulator = LocalSimulator(headless=True)
            scene_states = simulate_scenes(
                simulator=simulator,
                batch_parameters=make_standard_batch_parameters(),
                scenes=scene,
            )

            # Get the state at the beginning and end of the simulation.
            scene_state_begin = scene_states[0]
            scene_state_end = scene_states[-1]

            # Retrieve the states of the modular robot.
            robot_state_begin = scene_state_begin.get_modular_robot_simulation_state(
                robot
            )
            robot_state_end = scene_state_end.get_modular_robot_simulation_state(robot)

            # Calculate the xy displacement of the robot.
            select_fitness_function = {
                "x_displacement": fitness_functions.x_displacement,
                "y_displacement": fitness_functions.y_displacement,
                "xy_displacement": fitness_functions.xy_displacement,
            }

            fitness = select_fitness_function[config.FITNESS_FUNCTION](
                robot_state_begin, robot_state_end
            )


            # Penalize for having too many modules
            individual.num_bricks = self.count_modules(individual.gene)
            if individual.num_bricks > self.max_modules:
                fitness *= 1 + ((self.max_modules - individual.num_bricks) * 0.02)

            return fitness

        except Exception as e:
            self.logger.warning(f"Error evaluating individual : {str(e)}")
            return -1000.0  # Very low fitness for invalid individuals

    def evaluate_population(self) -> None:
        """Evaluate all individuals in the population."""
        self.logger.info(f"Evaluating population (generation {self.generation})")

        for individual in self.population:
            if individual.fitness == -float(
                "inf"
            ):  # Only evaluate if not already evaluated
                individual.fitness = self.evaluate_individual(individual)

                if config.VERBOSE_PRINTS:
                    print(
                        time.strftime("%H:%M:%S", time.gmtime(time.time())),
                        f"Evaluated individual with fitness: {individual.fitness:.3f}",
                    )
                self.evaluations += 1

        # Sort population by fitness (descending)
        self.population.sort(key=lambda x: x.fitness, reverse=True)

    def log_generation_stats(self) -> None:
        """Log statistics of the current generation."""

        self.plotter.log_generation(
            generation=self.generation,
            best=self.population[0].fitness,
            worst=self.population[-1].fitness,
            mean=sum(individual.fitness for individual in self.population)
            / len(self.population),
            median=self.population[len(self.population) // 2].fitness,
            std=np.std([individual.fitness for individual in self.population]),
            num_modules=self.population[0].num_bricks,
            total_elapsed_time=time.time() - self.start_time,
        )

        best_fitness = self.population[0].fitness
        mean_fitness = sum(individual.fitness for individual in self.population) / len(
            self.population
        )
        # Compute values for logging/printing (use the same values we logged to plotter)
        median_fitness = self.population[len(self.population) // 2].fitness
        std_fitness = np.std([individual.fitness for individual in self.population])
        num_modules_fitness = self.population[0].num_bricks

        self.logger.info(
            f"Generation {self.generation}: Best={best_fitness:.3f}, Mean={mean_fitness:.3f}, Median={median_fitness:.3f}, Std={std_fitness:.3f}, NumModules={num_modules_fitness:.3f}"
        )
        if config.VERBOSE_PRINTS:
            elapsed = time.time() - self.start_time
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
            print(
                f"\n----- Generation {self.generation} -----",
                f"RunID={self.runID}",
                f"Elapsed={elapsed_str}",
                f"Best={best_fitness:.3f}",
                f"Mean={mean_fitness:.3f}",
                f"Median={median_fitness:.3f}",
                f"Std={std_fitness:.3f}",
                f"NumModules={num_modules_fitness:.3f}",
                end="\n",
                sep="\n",
            )

    def tournament_selection(self, tournament_size: int = None) -> Individual:
        """Select an individual using tournament selection."""
        tournament_size = tournament_size or self.tournament_size

        # Select random individuals for tournament
        tournament = self.rng.choice(
            self.population, min(tournament_size, len(self.population))
        )

        # Return the best individual from the tournament
        return max(tournament, key=lambda x: x.fitness)

    def mutate_gene(self, gene: Dict[str, Any]) -> Dict[str, Any]:
        """Mutate a gene with various mutation operators."""
        mutated = copy.deepcopy(gene)

        def mutate_recursive(node, depth=0):
            if config.DEBUG_EA:
                print(node.keys())
            if (
                not isinstance(node, dict) or depth > 6
            ):  # Limit depth to prevent infinite growth
                return node

            for key, value in list(node.items()):
                if key in ["front", "right", "left", "back"]:
                    pmutate = 0.15 + 0.05 * depth
                    pskip = 0.7 - 0.10 * depth
                    # Mutation operations
                    mutation_type = self.rng.choice(
                        ["add_hinge", "remove_hinge", "modify_existing"],
                        p=[pmutate, pmutate, pskip],
                    )

                    if mutation_type == "add_hinge" and (not value or value == {}):
                        if config.DEBUG_EA:
                            print(node.keys(), "add")
                        new_brick = {
                        "front": {}, 
                        "right": {},
                        "left": {},
                        "rotation" : self.rng.integers(0,4) * np.pi/2}
                        rotation = 0.0
                        if self.rng.random() < config.CHANCE_TO_ROTATE:
                            rotation = self.rng.integers(1, 4) * np.pi / 2
                        node[key] = {
                            "hinge": {"brick": new_brick, "rotation": rotation}
                        }
                        if config.DEBUG_EA:
                            print(node.keys(), "added")

                    elif (
                        mutation_type == "remove_hinge"
                        and isinstance(value, dict)
                        and "hinge" in value
                    ):
                        # Remove hinge structure
                        if config.DEBUG_EA:
                            print(node.keys(), "remove")
                        node[key] = {}
                        if config.DEBUG_EA:
                            print(node.keys(), "removed")

                    elif (
                        mutation_type == "modify_existing"
                        and isinstance(value, dict)
                        and "hinge" in value
                    ):
                        # Recursively mutate the brick structure
                        if "brick" in value["hinge"]:
                            if config.DEBUG_EA:
                                print(node.keys(), "modify")
                            mutate_recursive(value["hinge"]["brick"], depth + 1)
                            if config.DEBUG_EA:
                                print(node.keys(), "modified")


            return node

        # Mutate the core and ensure symmetry
        if "core" in mutated:
            if self.rng.random() < config.MUTATION_RATE:
                mutate_recursive(mutated["core"])
            self.generator.spine_symmetry(mutated["core"])

        return mutated

    def crossover_genes(
        self, parent1: Individual, parent2: Individual
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Perform crossover between two parent genes.
        see ./images/sub-tree-crossover.png
        """

        # Create offspring as copies of parents
        offspring1 = copy.deepcopy(parent1.gene)  # Dict[str, Any]
        offspring2 = copy.deepcopy(parent2.gene)

        def recursive(node):
            """
            input: hinge: {brick: {...}}
            output: hinge: {brick: {...}}
            """
            faces = ["front", "right", "left"]
            face = self.rng.choice(faces)
            if not face in node["brick"]:
                return None
            if node["brick"][face]:  # eg. if node["front"] has content
                if self.rng.random() < config.CROSSOVER_CHANCE_TO_DIVE:
                    child = recursive(node["brick"][face]["hinge"])
                    if child:
                        return child
                    else:
                        return None
            return node

        # get subtrees to swap
        face = self.rng.choice(["front", "back", "right"])
        subtree1 = 0
        subtree2 = 0

        if "hinge" in offspring1["core"][face] and "hinge" in offspring2["core"][face]:
            subtree1 = recursive(offspring1["core"][face]["hinge"])
            if subtree1 is None:
                print("Crossover failed: no subtree found in offspring1")
                if config.DEBUGGING:
                    self.debug_dump(offspring1, "offspring1:")
            subtree2 = recursive(offspring2["core"][face]["hinge"])
            if subtree2 is None:
                print("Crossover failed: no subtree found in offspring2")
                if config.DEBUGGING:
                    self.debug_dump(offspring2, "offspring2:")

        if subtree1 and subtree2:  # Values 0 and None indicate failure to find subtree
            subtree1["brick"], subtree2["brick"] = subtree2["brick"], subtree1["brick"]

        return offspring1, offspring2

    def create_offspring(self) -> List[Individual]:
        """Create offspring using selection, crossover, and mutation."""

        if config.VERBOSE_PRINTS:
            print(time.time(),"Creating offspring")

        offspring = []

        for _ in range(self.offspring_size // 2):
            # Select parents
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()

            # Crossover
            if self.rng.random() < config.CROSSOVER_RATE:
                child1_gene, child2_gene = self.crossover_genes(parent1, parent2)
            else:
                child1_gene = copy.deepcopy(parent1.gene)
                child2_gene = copy.deepcopy(parent2.gene)

            # Mutation
            child1_gene = self.mutate_gene(child1_gene)
            child2_gene = self.mutate_gene(child2_gene)

            # Create offspring individuals
            offspring.extend(
                [Individual(gene=child1_gene), Individual(gene=child2_gene)]
            )

        # If odd offspring size, create one more
        if self.offspring_size % 2 == 1:
            parent = self.tournament_selection()
            child_gene = self.mutate_gene(copy.deepcopy(parent.gene))
            offspring.append(Individual(gene=child_gene))
            
        if config.VERBOSE_PRINTS:
            print(time.strftime("%H:%M:%S", time.gmtime(time.time())), "Created offspring")

        return offspring

    def survival_selection(self, offspring: List[Individual]) -> None:
        """Combine population and offspring, then select survivors."""
        # Combine population and offspring
        combined = self.population + offspring

        if config.VERBOSE_PRINTS:
            print(time.strftime("%H:%M:%S", time.gmtime(time.time())), "Evaluating offspring")
        # Evaluate offspring
        for individual in offspring:
            individual.fitness = self.evaluate_individual(individual)
            
            if config.VERBOSE_PRINTS:
                print(time.strftime("%H:%M:%S", time.gmtime(time.time())), f"Evaluated individual with fitness {individual.fitness:.3f} and {individual.num_bricks} bricks")
                
            
            self.evaluations += 1

        if config.VERBOSE_PRINTS:
            print(time.strftime("%H:%M:%S", time.gmtime(time.time())), "Finished evaluating offspring")
            print(time.strftime("%H:%M:%S", time.gmtime(time.time())), "population size:", len(self.population), "offspring size:", len(offspring), "combined size:", len(combined))

        # Sort by fitness (descending) and keep the best
        combined.sort(key=lambda x: x.fitness, reverse=True)
        self.population = combined[: self.population_size]

    def save_best_individual(self, filename: str = None) -> None:
        """Save the best individual to a file."""
        if not self.population:
            return

        best = self.population[0]
        best.gene["runID"] = self.runID
        best.gene["fitness"] = best.fitness
        best.gene["generation"] = self.generation
        best.gene["brain_weights"] = best.brain.weights.tolist()

        filename = filename or f"{self.log_folder}/best_gen_{self.generation}.json"

        with open(filename, "w") as f:
            json.dump(best.gene, f, indent=2)

        self.logger.info(
            f"Best individual saved to {filename} (fitness: {best.fitness:.3f})"
        )

    def run(self) -> Individual:
        print(time.strftime("%H:%M:%S", time.gmtime(time.time())), "Running EA")
        best_fitness = -float("inf")
        self.start_time = time.time()

        self.initialize_population()
        self.evaluate_population()  # Evaluate init
        self.log_generation_stats()  # Log init

        while self.evaluations < self.function_evaluations:
            self.generation += 1
            if config.VERBOSE_PRINTS:
                print(time.strftime("%H:%M:%S", time.gmtime(time.time())), "Starting generation", self.generation)

            offspring = self.create_offspring()
            self.survival_selection(
                offspring
            )  # only evaluates offspring and selects survivors from both parents and offspring
            self.log_generation_stats()  # Log the new generation and update plotter

            # Save new individual if improved
            if best_fitness < self.population[0].fitness:
                best_fitness = self.population[0].fitness
                self.save_best_individual()

            # Append last 5 logged generations to a progress CSV
            if self.generation % 5 == 0:
                if config.VERBOSE_PRINTS:
                    print(time.strftime("%H:%M:%S", time.gmtime(time.time())), "Appending last 5 generations to progress CSV")
                try:
                    self.plotter.append_last_n_to_csv(
                        self.log_folder + "progress.csv", n=5
                    )
                except Exception:
                    self.logger.exception("Failed to append generation progress to CSV")

            # Check termination condition
            if self.evaluations >= self.function_evaluations:
                break
            if config.DEBUGGING:
                print(f"->> Evaluation:{self.evaluations}")

        if self.generation % 5 != 0:
            # Ensure final data is saved if not already done
            try:
                self.plotter.append_last_n_to_csv(
                    self.log_folder + "progress.csv", n=self.generation % 5
                )
            except Exception:
                self.logger.exception(
                    "Failed to append final generation progress to CSV"
                )
        print(
            f"EA completed after {self.generation} generations and {self.evaluations} evaluations"
        )
        self.save_best_individual(
            self.log_folder + "final_best_individual.json"
        )  # Save final best individual

        return self.population[0]


def main():
    """Main function to run the evolutionary algorithm."""

    
    ea = JSONGeneEA()
    ea.write_run_info()
    best_individual = ea.run()

    print(f"\nBest individual found:")
    print(f"Fitness: {best_individual.fitness:.3f}")
    print(f"Modules: {ea.count_modules(best_individual.gene)}")

    # Print gene structure
    print("\nGene structure:")
    print_json_gene(best_individual.gene)


if __name__ == "__main__":
    main()
