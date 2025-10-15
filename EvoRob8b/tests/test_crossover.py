import sys
import os
# ensure the package root (EvoRob8b folder) is on sys.path so we can import EA.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from EA import JSONGeneEA
import json

nr = 1
nr2 = 2

ea = JSONGeneEA()

with open(f"./genes_wardrobe/gene_{nr}.json", "r") as f:
    gene = json.load(f)

with open(f"./genes_wardrobe/gene_{nr2}.json", "r") as f:
    gene2 = json.load(f)

ea.crossover(gene, gene2)