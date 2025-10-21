import sys
import os
# ensure the package root (EvoRob8b folder) is on sys.path so we can import EA.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from EA import JSONGeneEA, Individual
import json

nr = 4
nr2 = 2

ea = JSONGeneEA()

with open(f"../genes_wardrobe/gene_{nr}.json", "r") as f:
    gene1 = json.load(f)

individual1 = Individual(gene1)

with open(f"../genes_wardrobe/gene_{nr2}.json", "r") as f:
    gene2 = json.load(f)

individual2 = Individual(gene2)

with open("test.json", "w") as f:
    json.dump(individual1.gene["core"], f)
    json.dump(individual2.gene["core"], f)

individual1, individual2 = ea.crossover_genes(individual1, individual2)

with open("test2.json", "w") as f:
    json.dump(individual1.gene["core"], f)
    json.dump(individual2.gene["core"], f)
