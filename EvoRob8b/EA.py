"""
Evolutionary Algorithm for JSON gene representation of modular robots.
This EA handles evolution of robot body structures represented as nested JSON.
"""

import json
import random
import copy
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import logging

import config
from parse_gene import build_body, print_json_gene
from gene_generator import Gene_Generator

from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot_simulation import ModularRobotScene, simulate_scenes
from revolve2.modular_robot.brain.cpg import BrainCpgNetworkNeighborRandom
from revolve2.experimentation.rng import make_rng_time_seed
from revolve2.simulators.mujoco_simulator import LocalSimulator
from revolve2.standards.simulation_parameters import make_standard_batch_parameters
from revolve2.standards import fitness_functions, terrains


@dataclass
class Individual:
    """Represents an individual in the population."""
    gene: Dict[str, Any]
    fitness: float = -float('inf')
    id: int = 0
    
    def __post_init__(self):
        if self.id == 0:
            self.id = random.randint(1, 1000000)


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

        self.generator = Gene_Generator()
        
        # Initialize random number generator
        self.rng = make_rng_time_seed()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def initialize_population(self) -> None:
        """Initialize the population with random individuals."""
        self.logger.info(f"Initializing population of size {self.population_size}")
        
        for i in range(self.population_size):
            # Generate random gene
            # gene_generator was refactored into a class - use it here
            gene_dict = self.generator.make_core()
            gene_dict["id"] = i + 1
            gene_dict["brain"] = {}
            
            individual = Individual(gene=gene_dict, id=i + 1)
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
                    count += 1  # Count the hinge
                    if isinstance(value, dict) and "hinge" in value and "brick" in value["hinge"]:
                        count += 1  # Count the brick
                        count += count_recursive(value["hinge"]["brick"])
                elif isinstance(value, dict):
                    count += count_recursive(value)
            
            return count
        
        return count_recursive(gene_dict.get("core", {}))
    
    def evaluate_individual(self, individual: Individual) -> float:
        """Evaluate the fitness of an individual."""
        try:
            # Build the robot body from the gene
            body = build_body(individual.gene)
            
            # Create brain
            brain = BrainCpgNetworkNeighborRandom(body=body, rng=self.rng)
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
            robot_state_begin = scene_state_begin.get_modular_robot_simulation_state(robot)
            robot_state_end = scene_state_end.get_modular_robot_simulation_state(robot)

            # Calculate the xy displacement (fitness) of the robot.
            fitness = fitness_functions.xy_displacement(
                robot_state_begin, robot_state_end
            )
            
            # Penalize for having too many modules
            module_count = self.count_modules(individual.gene)
            if module_count > self.max_modules:
                fitness *= 1+((self.max_modules-module_count)*0.02)
            
            return fitness
            
        except Exception as e:
            self.logger.warning(f"Error evaluating individual {individual.id}: {str(e)}")
            return -1000.0  # Very low fitness for invalid individuals
    
    def evaluate_population(self) -> None:
        """Evaluate all individuals in the population."""
        self.logger.info(f"Evaluating population (generation {self.generation})")
        
        for individual in self.population:
            if individual.fitness == -float('inf'):  # Only evaluate if not already evaluated
                individual.fitness = self.evaluate_individual(individual)
                self.evaluations += 1
        
        # Sort population by fitness (descending)
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        best_fitness = self.population[0].fitness
        avg_fitness = sum(individual.fitness for individual in self.population) / len(self.population)
        
        self.logger.info(f"Generation {self.generation}: Best={best_fitness:.3f}, Avg={avg_fitness:.3f}")
    
    def tournament_selection(self, tournament_size: int = None) -> Individual:
        """Select an individual using tournament selection."""
        tournament_size = tournament_size or self.tournament_size
        
        # Select random individuals for tournament
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        
        # Return the best individual from the tournament
        return max(tournament, key=lambda x: x.fitness)
    
    def mutate_gene(self, gene_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Mutate a gene with various mutation operators."""
        mutated = copy.deepcopy(gene_dict)
        
        def mutate_recursive(node, depth=0):
            if not isinstance(node, dict) or depth > 6:  # Limit depth to prevent infinite growth
                return node
            
            for key, value in list(node.items()):
                if key in ["front", "right", "back"]:
                    if random.random() < self.mutation_rate:
                        pmutate = 0.1+0.03*depth
                        pskip = 0.7-0.09*depth
                        # Mutation operations
                        mutation_type = np.random.choice([
                            "add_hinge", "remove_hinge", "modify_existing", "swap_sides"
                        ], p=[pmutate, pmutate, pskip, pmutate])
                        # TODO add orientation fix
                        if mutation_type == "add_hinge" and (not value or value == {}):
                            # Add a new hinge+brick structure
                            max_depth = random.randint(1, 3)
                            # Use the Gene_Generator class to create a brick subtree
                            node[key] = {
                                "hinge": {
                                    "brick": self.generator.make_brick(),
                                    "orientation": {}
                                }
                            }
                        
                        elif mutation_type == "remove_hinge" and isinstance(value, dict) and "hinge" in value:
                            # Remove hinge structure
                            node[key] = {}
                        
                        elif mutation_type == "modify_existing" and isinstance(value, dict) and "hinge" in value:
                            # Recursively mutate the brick structure
                            if "brick" in value["hinge"]:
                                mutate_recursive(value["hinge"]["brick"], depth + 1)
                        
                        elif mutation_type == "swap_sides" and key in ["front", "left", "right"]:
                            # Swap with another side
                            other_sides = [s for s in ["front", "left", "right", "back"] if s != key]
                            other_key = random.choice(other_sides)
                            if other_key in node:
                                node[key], node[other_key] = node[other_key], node[key]
                    
                    elif isinstance(value, dict):
                        mutate_recursive(value, depth + 1)
            
            return node
        
        # Mutate the core and ensure symmetry
        if "core" in mutated:
            mutate_recursive(mutated["core"])
            self.generator.spine_symmetry(mutated["core"])

        return mutated
    
    def crossover_genes(self, parent1: Individual, parent2: Individual) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Perform crossover between two parent genes."""
        
        def get_subtrees(node, path=""):
            """Get all subtrees with their paths."""
            subtrees = []
            if not isinstance(node, dict):
                return subtrees
            
            for key, value in node.items():
                if key in ["front", "left", "right", "back"] and isinstance(value, dict) and "hinge" in value:
                    current_path = f"{path}.{key}" if path else key
                    subtrees.append((current_path, value))
                    if "brick" in value.get("hinge", {}):
                        subtrees.extend(get_subtrees(value["hinge"]["brick"], current_path))
                elif isinstance(value, dict):
                    subtrees.extend(get_subtrees(value, path))
            
            return subtrees
        
        # Create offspring as copies of parents
        offspring1 = copy.deepcopy(parent1.gene)
        offspring2 = copy.deepcopy(parent2.gene)
        
        # Get subtrees from both parents
        subtrees1 = get_subtrees(parent1.gene.get("core", {}))
        subtrees2 = get_subtrees(parent2.gene.get("core", {}))
        
        if subtrees1 and subtrees2:
            # Select random subtrees to swap
            path1, subtree1 = random.choice(subtrees1)
            path2, subtree2 = random.choice(subtrees2)
            
            # Function to set value at path
            def set_at_path(gene_dict, path, value):
                keys = path.split('.')
                current = gene_dict["core"]
                
                for key in keys[:-1]:
                    if key not in current:
                        return False
                    if "hinge" in current[key] and "brick" in current[key]["hinge"]:
                        current = current[key]["hinge"]["brick"]
                    else:
                        current = current[key]
                
                if keys[-1] in current:
                    current[keys[-1]] = value
                    return True
                return False
            
            # Perform crossover by swapping subtrees
            set_at_path(offspring1, path1, subtree2)
            set_at_path(offspring2, path2, subtree1)
        
        return offspring1, offspring2
    
    def create_offspring(self) -> List[Individual]:
        """Create offspring using selection, crossover, and mutation."""
        offspring = []
        
        for _ in range(self.offspring_size // 2):
            # Select parents
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()
            
            # Crossover
            if random.random() < config.CROSSOVER_RATE:
                child1_gene, child2_gene = self.crossover_genes(parent1, parent2)
            else:
                child1_gene = copy.deepcopy(parent1.gene)
                child2_gene = copy.deepcopy(parent2.gene)
            
            # Mutation
            child1_gene = self.mutate_gene(child1_gene)
            child2_gene = self.mutate_gene(child2_gene)
            
            # Create offspring individuals
            offspring.extend([
                Individual(gene=child1_gene),
                Individual(gene=child2_gene)
            ])
        
        # If odd offspring size, create one more
        if self.offspring_size % 2 == 1:
            parent = self.tournament_selection()
            child_gene = self.mutate_gene(copy.deepcopy(parent.gene))
            offspring.append(Individual(gene=child_gene))
        
        return offspring
    
    def survival_selection(self, offspring: List[Individual]) -> None:
        """Combine population and offspring, then select survivors."""
        # Combine population and offspring
        combined = self.population + offspring
        
        # Evaluate offspring
        for individual in offspring:
            individual.fitness = self.evaluate_individual(individual)
            self.evaluations += 1
        
        # Sort by fitness (descending) and keep the best
        combined.sort(key=lambda x: x.fitness, reverse=True)
        self.population = combined[:self.population_size]
    
    def save_best_individual(self, filename: str = None) -> None:
        """Save the best individual to a file."""
        if not self.population:
            return
        
        best = self.population[0]
        filename = filename or config.LOG_FOLDER+f"best_individual_gen_{self.generation}.json"
        
        with open(filename, 'w') as f:
            json.dump(best.gene, f, indent=2)
        
        self.logger.info(f"Best individual saved to {filename} (fitness: {best.fitness:.3f})")
    
    def run(self) -> Individual:
        """Run the evolutionary algorithm."""
        self.logger.info("Starting evolutionary algorithm")
        print(1)
        # Initialize population
        self.initialize_population()
        print(2)
        # Evaluate initial population
        self.evaluate_population()
        print(3)
        # Evolution loop
        while self.evaluations < self.function_evaluations:
            self.generation += 1
            print(4)
            # Create offspring
            offspring = self.create_offspring()
            print(5)
            # Survival selection
            self.survival_selection(offspring)
            print(6)
            # Log progress
            if self.generation % 10 == 0:
                self.save_best_individual()
            print(7)
            # Check termination condition
            if self.evaluations >= self.function_evaluations:
                break
            print(8)
        self.logger.info(f"Evolution completed after {self.generation} generations and {self.evaluations} evaluations")
        print(9)
        # Save final best individual
        self.save_best_individual("final_best_individual.json")
        
        return self.population[0]


def main():
    """Main function to run the evolutionary algorithm."""
    ea = JSONGeneEA()
    best_individual = ea.run()
    
    print(f"\nBest individual found:")
    print(f"Fitness: {best_individual.fitness:.3f}")
    print(f"Modules: {ea.count_modules(best_individual.gene)}")
    
    # Print gene structure
    print("\nGene structure:")
    print_json_gene(best_individual.gene)


if __name__ == "__main__":
    main()
