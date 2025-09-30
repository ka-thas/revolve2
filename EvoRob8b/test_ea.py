"""
Test script for the JSON Gene Evolutionary Algorithm.
"""

import json
import sys
import os

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from EA_simple import SimpleJSONGeneEA, Individual
from gene_generator import make_core
import config


def test_individual_creation():
    """Test creating individuals with genes."""
    print("Testing individual creation...")
    
    # Create a simple gene
    gene = make_core(max_depth=3)
    gene["id"] = 1
    gene["brain"] = {}
    
    individual = Individual(gene=gene)
    print(f"Created individual with ID: {individual.id}")
    print(f"Gene keys: {list(individual.gene.keys())}")
    
    return individual


def test_module_counting():
    """Test module counting functionality."""
    print("\nTesting module counting...")
    
    ea = SimpleJSONGeneEA(population_size=5, offspring_size=2)
    
    # Load existing test gene
    try:
        with open("testgene.json", "r") as f:
            test_gene = json.load(f)
        
        count = ea.count_modules(test_gene)
        print(f"Modules in testgene.json: {count}")
        
    except FileNotFoundError:
        print("testgene.json not found, creating simple test gene")
        test_gene = make_core(max_depth=2)
        count = ea.count_modules(test_gene)
        print(f"Modules in generated gene: {count}")
    
    return count


def test_mutation():
    """Test mutation operators."""
    print("\nTesting mutation...")
    
    ea = SimpleJSONGeneEA()
    
    # Create a base gene
    original_gene = make_core(max_depth=2)
    original_gene["id"] = 999
    original_gene["brain"] = {}
    
    print("Original gene structure:")
    core = original_gene.get("core", {})
    for limb in ["front", "left", "right", "back"]:
        has_limb = limb in core and core[limb] and "hinge" in str(core[limb])
        print(f"  {limb}: {'✓' if has_limb else '✗'}")
    
    # Apply mutation multiple times
    mutated_gene = original_gene
    for i in range(3):
        mutated_gene = ea.mutate_gene(mutated_gene, mutation_rate=0.5)
    
    print("After mutation:")
    core = mutated_gene.get("core", {})
    for limb in ["front", "left", "right", "back"]:
        has_limb = limb in core and core[limb] and "hinge" in str(core[limb])
        print(f"  {limb}: {'✓' if has_limb else '✗'}")


def test_crossover():
    """Test crossover operation."""
    print("\nTesting crossover...")
    
    ea = SimpleJSONGeneEA()
    
    # Create two parent genes
    parent1_gene = make_core(max_depth=2)
    parent1_gene["id"] = 1
    parent1_gene["brain"] = {}
    parent1 = Individual(gene=parent1_gene)
    
    parent2_gene = make_core(max_depth=2)
    parent2_gene["id"] = 2
    parent2_gene["brain"] = {}
    parent2 = Individual(gene=parent2_gene)
    
    print("Parent 1:")
    core1 = parent1.gene.get("core", {})
    for limb in ["front", "left", "right", "back"]:
        has_limb = limb in core1 and core1[limb] and "hinge" in str(core1[limb])
        print(f"  {limb}: {'✓' if has_limb else '✗'}")
    
    print("Parent 2:")
    core2 = parent2.gene.get("core", {})
    for limb in ["front", "left", "right", "back"]:
        has_limb = limb in core2 and core2[limb] and "hinge" in str(core2[limb])
        print(f"  {limb}: {'✓' if has_limb else '✗'}")
    
    # Perform crossover
    child1_gene, child2_gene = ea.crossover_genes(parent1, parent2)
    
    print("Child 1:")
    core_c1 = child1_gene.get("core", {})
    for limb in ["front", "left", "right", "back"]:
        has_limb = limb in core_c1 and core_c1[limb] and "hinge" in str(core_c1[limb])
        print(f"  {limb}: {'✓' if has_limb else '✗'}")


def test_mini_evolution():
    """Test a mini evolution run."""
    print("\nTesting mini evolution...")
    
    # Create EA with very small parameters
    ea = SimpleJSONGeneEA(population_size=5, offspring_size=4)
    ea.max_generations = 5
    
    # Run evolution
    best = ea.run()
    
    print(f"Best individual fitness: {best.fitness:.3f}")
    print(f"Best individual modules: {ea.count_modules(best.gene)}")
    
    return best


def main():
    """Run all tests."""
    print("JSON Gene EA Test Suite")
    print("=" * 50)
    
    try:
        # Test individual creation
        individual = test_individual_creation()
        
        # Test module counting
        module_count = test_module_counting()
        
        # Test mutation
        test_mutation()
        
        # Test crossover
        test_crossover()
        
        # Test mini evolution
        best = test_mini_evolution()
        
        print("\n" + "=" * 50)
        print("All tests completed successfully! ✓")
        
        # Save the best individual from the mini run
        with open("test_best.json", "w") as f:
            json.dump(best.gene, f, indent=2)
        print("Test results saved to test_best.json")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()