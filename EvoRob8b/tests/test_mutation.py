import sys
import os
import json

# ensure the package root (EvoRob8b folder) is on sys.path so we can import EA.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from EA import JSONGeneEA, Individual

nr = 1

ea = JSONGeneEA()

with open(f"../genes_wardrobe/gene_{nr}.json", "r") as f:
    gene = json.load(f)

individual = Individual(gene)

# Apply mutation to the gene
mutated = ea.mutate_gene(individual.gene)

# Save mutated gene to a file for inspection
with open("mutated_gene.json", "w") as f:
    json.dump(mutated, f, indent=4)

print("Mutated gene written to mutated_gene.json")
