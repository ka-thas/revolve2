import sys
import os
# ensure the package root (EvoRob8b folder) is on sys.path so we can import EA.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from EA import JSONGeneEA, Individual
import json

nr = 1
nr2 = 5

ea = JSONGeneEA()

with open(f"../genes_wardrobe/gene_{nr}.json", "r") as f:
    gene1 = json.load(f)


with open(f"../genes_wardrobe/gene_{nr2}.json", "r") as f:
    gene2 = json.load(f)

individual1 = Individual(gene1)
individual2 = Individual(gene2)

with open("test.json", "w") as f:
    todump = {"core1": individual1.gene["core"], "core2": individual2.gene["core"]}
    json.dump(todump, f, indent=4)

gene1, gene2 = ea.crossover_genes(individual1, individual2)

with open("test2.json", "w") as f:
    todump = {"core1": gene1["core"], "core2": gene2["core"]}
    json.dump(todump, f, indent=4)
