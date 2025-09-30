"""
Simple Evolutionary Algorithm for JSON gene representation.
This version focuses on the EA mechanics without simulation dependencies.
"""

import json
import random
import copy
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import logging

import config
from gene_generator import make_core, save_gene


@dataclass
class Individual:
    """Represents an individual in the population."""
    gene: Dict[str, Any]
    fitness: float = -float('inf')
    id: int = 0
    
    def __post_init__(self):
        if self.id == 0:
            self.id = random.randint(1, 1000000)


class SimpleJSONGeneEA:
    """Simple Evolutionary Algorithm for JSON gene representation."""
    
    def __init__(self, population_size: int = None, offspring_size: int = None):
        """Initialize the EA with configuration parameters."""
        self.population_size = population_size or config.POPULATION_SIZE
        self.offspring_size = offspring_size or config.OFFSPRING_SIZE
        self.max_modules = config.MAX_NUMBER_OF_MODULES
        self.tournament_size = config.PARENT_TOURNAMENT_SIZE
        self.max_generations = 100  # Simplified termination
        
        self.population: List[Individual] = []
        self.generation = 0
        self.evaluations = 0
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def initialize_population(self) -> None:
        """Initialize the population with random individuals."""
        self.logger.info(f"Initializing population of size {self.population_size}")
        
        for i in range(self.population_size):
            # Generate random gene
            max_depth = random.randint(2, 4)
            gene_dict = make_core(max_depth)
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
                if key in ["front", "left", "right", "back"] and isinstance(value, dict) and "hinge" in value:
                    count += 1  # Count the hinge
                    if "brick" in value.get("hinge", {}):
                        count += 1  # Count the brick
                        count += count_recursive(value["hinge"]["brick"])
                elif isinstance(value, dict):
                    count += count_recursive(value)
            
            return count
    
    def calculate_complexity_score(self, gene_dict: Dict[str, Any]) -> float:
        """Calculate a complexity score based on structure."""
        def complexity_recursive(node, depth=0):
            if not isinstance(node, dict) or not node:
                return 0
            
            score = 0
            for key, value in node.items():
                if key in ["front", "left", "right", "back"] and isinstance(value, dict) and "hinge" in value:
                    # Reward for having limbs, but penalize excessive depth
                    score += max(0, 10 - depth * 2)
                    if "brick" in value.get("hinge", {}):
                        score += complexity_recursive(value["hinge"]["brick"], depth + 1)
                elif isinstance(value, dict):
                    score += complexity_recursive(value, depth)
            
            return score
        
        return complexity_recursive(gene_dict.get("core", {}))
    
    def evaluate_individual_simple(self, individual: Individual) -> float:
        """Simple evaluation function based on structure metrics."""
        try:
            module_count = self.count_modules(individual.gene)
            complexity = self.calculate_complexity_score(individual.gene)
            
            # Balance between having modules and not being too complex
            if module_count == 0:
                return 0.0
            
            # Fitness based on complexity but penalize excessive modules
            fitness = complexity
            if module_count > self.max_modules:
                fitness *= 0.1  # Heavy penalty for oversized robots
            
            # Add some randomness to simulate environmental variation
            fitness += random.gauss(0, 5)
            
            return max(0, fitness)
            
        except Exception as e:
            self.logger.warning(f"Error evaluating individual {individual.id}: {str(e)}")
            return 0.0
    
    def evaluate_population(self) -> None:
        """Evaluate all individuals in the population."""
        self.logger.info(f"Evaluating population (generation {self.generation})")
        
        for individual in self.population:
            if individual.fitness == -float('inf'):  # Only evaluate if not already evaluated
                individual.fitness = self.evaluate_individual_simple(individual)
                self.evaluations += 1
        
        # Sort population by fitness (descending)
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        best_fitness = self.population[0].fitness
        avg_fitness = sum(ind.fitness for ind in self.population) / len(self.population)
        
        self.logger.info(f"Generation {self.generation}: Best={best_fitness:.3f}, Avg={avg_fitness:.3f}")
    
    def tournament_selection(self, tournament_size: int = None) -> Individual:
        """Select an individual using tournament selection."""
        tournament_size = tournament_size or self.tournament_size
        
        # Select random individuals for tournament
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        
        # Return the best individual from the tournament
        return max(tournament, key=lambda x: x.fitness)
    
    def mutate_gene(self, gene_dict: Dict[str, Any], mutation_rate: float = 0.3) -> Dict[str, Any]:
        """Mutate a gene with various mutation operators."""
        mutated = copy.deepcopy(gene_dict)
        
        def mutate_recursive(node, depth=0):
            if not isinstance(node, dict) or depth > 5:  # Limit depth
                return node
            
            for key, value in list(node.items()):
                if key in ["front", "left", "right", "back"]:
                    if random.random() < mutation_rate:
                        # Mutation operations
                        mutation_type = random.choice([
                            "add_hinge", "remove_hinge", "modify_existing"
                        ])
                        
                        if mutation_type == "add_hinge" and (not value or value == {}):
                            # Add a new hinge+brick structure
                            max_depth = random.randint(1, 2)
                            from gene_generator import make_brick
                            node[key] = {
                                "hinge": {
                                    "brick": make_brick(0, max_depth),
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
                    
                    elif isinstance(value, dict):
                        mutate_recursive(value, depth + 1)
            
            return node
        
        # Mutate the core
        if "core" in mutated:
            mutate_recursive(mutated["core"])
        
        return mutated
    
    def crossover_genes(self, parent1: Individual, parent2: Individual) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Perform single-point crossover between two parent genes."""
        offspring1 = copy.deepcopy(parent1.gene)
        offspring2 = copy.deepcopy(parent2.gene)
        
        # Simple crossover: swap random limbs between parents
        if "core" in offspring1 and "core" in offspring2:
            limbs = ["front", "left", "right", "back"]
            swap_limb = random.choice(limbs)
            
            if swap_limb in offspring1["core"] and swap_limb in offspring2["core"]:
                offspring1["core"][swap_limb], offspring2["core"][swap_limb] = \
                    offspring2["core"][swap_limb], offspring1["core"][swap_limb]
        
        return offspring1, offspring2
    
    def create_offspring(self) -> List[Individual]:
        """Create offspring using selection, crossover, and mutation."""
        offspring = []
        
        for _ in range(self.offspring_size // 2):
            # Select parents
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()
            
            # Crossover
            if random.random() < 0.8:  # 80% crossover rate
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
        # Evaluate offspring
        for individual in offspring:
            individual.fitness = self.evaluate_individual_simple(individual)
            self.evaluations += 1
        
        # Combine population and offspring
        combined = self.population + offspring
        
        # Sort by fitness (descending) and keep the best
        combined.sort(key=lambda x: x.fitness, reverse=True)
        self.population = combined[:self.population_size]
    
    def save_best_individual(self, filename: str = None) -> None:
        """Save the best individual to a file."""
        if not self.population:
            return
        
        best = self.population[0]
        filename = filename or f"best_individual_gen_{self.generation}.json"
        
        with open(filename, 'w') as f:
            json.dump(best.gene, f, indent=2)
        
        self.logger.info(f"Best individual saved to {filename} (fitness: {best.fitness:.3f})")
    
    def get_statistics(self) -> Dict[str, float]:
        """Get population statistics."""
        if not self.population:
            return {}
        
        fitnesses = [ind.fitness for ind in self.population]
        modules = [self.count_modules(ind.gene) for ind in self.population]
        
        return {
            "best_fitness": max(fitnesses),
            "avg_fitness": sum(fitnesses) / len(fitnesses),
            "worst_fitness": min(fitnesses),
            "avg_modules": sum(modules) / len(modules),
            "best_modules": modules[0]  # Since population is sorted by fitness
        }
    
    def run(self) -> Individual:
        """Run the evolutionary algorithm."""
        self.logger.info("Starting evolutionary algorithm")
        
        # Initialize population
        self.initialize_population()
        
        # Evaluate initial population
        self.evaluate_population()
        
        # Evolution loop
        for generation in range(self.max_generations):
            self.generation = generation + 1
            
            # Create offspring
            offspring = self.create_offspring()
            
            # Survival selection
            self.survival_selection(offspring)
            
            # Log progress and save best
            if self.generation % 10 == 0:
                stats = self.get_statistics()
                self.logger.info(f"Gen {self.generation}: "
                               f"Best={stats['best_fitness']:.1f}, "
                               f"Avg={stats['avg_fitness']:.1f}, "
                               f"Modules={stats['best_modules']:.0f}")
                self.save_best_individual()
        
        self.logger.info(f"Evolution completed after {self.generation} generations")
        
        # Save final best individual
        self.save_best_individual("final_best_simple.json")
        
        return self.population[0]


def demo_simple_ea():
    """Demo function to test the simple EA."""
    print("Running Simple JSON Gene EA Demo")
    print("=" * 40)
    
    # Create and run EA with smaller parameters for demo
    ea = SimpleJSONGeneEA(population_size=20, offspring_size=10)
    ea.max_generations = 50
    
    best_individual = ea.run()
    
    print(f"\nBest individual found:")
    print(f"Fitness: {best_individual.fitness:.3f}")
    print(f"Modules: {ea.count_modules(best_individual.gene)}")
    
    # Print final statistics
    stats = ea.get_statistics()
    print(f"\nFinal Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.2f}")
    
    # Show gene structure (simplified)
    print(f"\nBest gene structure preview:")
    core = best_individual.gene.get("core", {})
    for limb in ["front", "left", "right", "back"]:
        has_limb = limb in core and core[limb] and "hinge" in str(core[limb])
        print(f"  {limb}: {'✓' if has_limb else '✗'}")


if __name__ == "__main__":
    demo_simple_ea()