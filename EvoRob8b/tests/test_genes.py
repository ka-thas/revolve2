import sys
import os
# ensure the package root (EvoRob8b folder) is on sys.path so we can import EA.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gene_generator
import json

def test_genes(gene):
    pass


def test_hinge_brick(gene):
    # test if all hinges have bricks
    # Also makes sure that hinges are not leaves
    pass

def test_faces(gene):
    # test if all bricks have front, left, right faces
    pass


if __name__ == "__main__":
    nr = 1

    generator = gene_generator.Gene_Generator()
    gene = generator.make_core()

    test_genes(gene)
