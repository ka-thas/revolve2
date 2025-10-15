import json
import random
import uuid
import numpy as np
import config

class Gene_Generator:
    queue: list
    def __init__(self):
        self.brick_count = 0
        self.queue = []


    def mirror_right(self, left_brick, right_brick):
        """
        input: brick from left arm, brick from right arm

        output: left_arm mirrored from right. recursively
        """
        for key, value in right_brick.items():
            if key == "front":
                if "hinge" in value.keys():
                    new_left_brick = {}
                    rotation = right_brick["front"]["hinge"]["rotation"] 
                    left_brick["front"] = {"hinge": {"brick": new_left_brick, "rotation": -rotation}}
                    new_right_brick = right_brick["front"]["hinge"]["brick"]
                    self.mirror_right(new_left_brick, new_right_brick)

            if key == "left":
                if "hinge" in value.keys():
                    rotation = right_brick["left"]["hinge"]["rotation"]
                    new_left_brick = {}
                    left_brick["right"] = {"hinge": {"brick": new_left_brick, "rotation": -rotation }}
                    new_right_brick = right_brick["left"]["hinge"]["brick"]
                    self.mirror_right(new_left_brick, new_right_brick)

            if key == "right":
                if "hinge" in value.keys():
                    rotation = right_brick["right"]["hinge"]["rotation"]
                    new_left_brick = {}
                    left_brick["left"] = {"hinge": {"brick": new_left_brick, "rotation": -rotation }}
                    new_right_brick = right_brick["right"]["hinge"]["brick"]
                    self.mirror_right(new_left_brick, new_right_brick)

        return

    # recursively goes through spine mirroring the right side !!based on perspective.
    # meaning from core it mirrors rightside on the front and left side on the back
    def spine_symmetry(self, spinebrick):
        """
        spinebrick: inserted with core. and recursively called using spinebrick
        
        output: none but changes the input dict
        """

        spinebrick["left"] = {}
        if spinebrick["right"]:
            new_left_brick = {}
            rotation = spinebrick["right"]["hinge"]["rotation"] 
            spinebrick["left"] = {"hinge": {"brick": new_left_brick, "rotation": -rotation}}
            new_right_brick = spinebrick["right"]["hinge"]["brick"]
            self.mirror_right(new_left_brick, new_right_brick)

        if spinebrick["front"]:
            self.spine_symmetry(spinebrick["front"]["hinge"]["brick"])

        if "back" in spinebrick.keys():
            if spinebrick["back"]: # not empty
                self.spine_symmetry(spinebrick["back"]["hinge"]["brick"])
        return

    # Main Function to make the core
    # BFS to shorten limbs and not exhaust all bricks on one branch
    def make_core(self):
        """Create a Core with random children (front, left, right, back)."""
        core = {}
        self.queue = [(core, True)]
        while self.queue:
            current_module, spine = self.queue.pop(0)
            
            #Only uses back on the spine. and doesnt make a left on the side
            if spine:
                sides = ["front", "right", "back"]
            else:
                sides = ["front", "left", "right"]
            for side in sides:
                #can have up to 38 bricks with a limit of 20. because it doesnt count the doubling when its copied from right to left. other than the first time
                # Might make the body wider than longer
                if spine and side == "right":
                    self.brick_count += 2
                else:
                    self.brick_count +=1

                if random.random() < config.CHANCE_TO_PLACE_BRICK and self.brick_count <= config.MAX_BRICKS:
                    new_module = {
                        "front": {},
                        "right": {},
                        "left": {}
                    }
                    rotation = 0.0
                    if random.random() < config.CHANCE_TO_ROTATE:
                        rotation = random.randint(1,3) * np.pi/2
                    current_module[side] = {"hinge": {"brick": new_module, "rotation": rotation}
                                            
                    }
                else:
                    current_module[side] = {}
                    continue
                if side == "front" or side == "back" and spine == True:
                    self.queue.append((new_module, True))
                else:
                    self.queue.append((new_module, False))
        
        

        self.spine_symmetry(core)
        self.brick_count = 0
        return {"core" : core}

    #json dump to save gene
    def save_gene(self, gene, filename):
        with open(filename, "w") as f:
            json.dump(gene, f, indent=4)


if __name__ == "__main__":
    random.seed(7449)

    generator = Gene_Generator()

    # Generate 5 random genes and save them
    for i in range(6):
        print(i)
        gene = generator.make_core()  # deeper robots if you want
        generator.save_gene(gene, f"./genes_wardrobe/gene_{i}.json")
        print(f"Saved gene_{i}.json")
