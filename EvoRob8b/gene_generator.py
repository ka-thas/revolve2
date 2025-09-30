import json
import random
import uuid


def random_orientation():
    """Return a random orientation value."""
    return {"orientation": 1}


def make_brick(depth, max_depth):
    """Create a Brick with random children (front, left, right)."""
    if depth >= max_depth:
        return {}  # stop recursion

    brick = {}
    for side in ["front", "left", "right"]:
        if random.random() < 0.5:  # 50% chance to place hinge
            brick[side] = {
                "hinge": {
                    "brick": make_brick(depth + 1, max_depth)
                }
            }
        else:
            brick[side] = {}
    return brick


def make_core(max_depth=5):
    """Create a Core with random children (front, left, right, back)."""
    core = {}
    for side in ["front", "left", "right", "back"]:
        if random.random() < 0.7:  # 70% chance to place hinge
            core[side] = {
                "hinge": {
                    "brick": make_brick(1, max_depth)
                }
            }
        else:
            core[side] = {}
    return {"core": core}


def save_gene(gene, filename):
    with open(filename, "w") as f:
        json.dump(gene, f, indent=4)


if __name__ == "__main__":
    # Generate 5 random genes and save them
    for i in range(5):
        gene = make_core(max_depth=3)  # deeper robots if you want
        save_gene(gene, f"gene_{i+1}.json")
        print(f"Saved gene_{i+1}.json")
