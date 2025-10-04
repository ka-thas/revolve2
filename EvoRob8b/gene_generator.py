import json
import random
import uuid
import config


class Gene_Generator:
    def __init__(self):
        self.max_bricks = config.MAX_BRICKS
        self.brick_count = 0

    def random_orientation(self):
        """Return a random orientation value."""
        return {"orientation": 1}

    def make_brick(self):
        """Create a Brick with random children (front, left, right)."""      
        brick = {}
        for side in ["front", "left", "right"]:
            if (
                random.random() < config.CHANCETOPLACEBRICK
            ):  # chance to place hinge from config
                if self.brick_count >= 20:
                    brick[side] = {}
                else:
                    brick[side] = {
                        "hinge": {"brick": self.make_brick()}
                    }
                self.brick_count += 1
            else:
                brick[side] = {}
        return brick

    def mirror_right(self, left_brick, right_brick):
        """
        input: brick from left arm, brick from right arm

        output: left_arm mirrored from right. recursively
        """
        for key, value in right_brick.items():
            if key == "front":
                if "hinge" in value.keys():
                    new_left_brick = {}
                    left_brick["front"] = {"hinge": {"brick": new_left_brick}}
                    new_right_brick = right_brick["front"]["hinge"]["brick"]
                    self.mirror_right(new_left_brick, new_right_brick)

            if key == "left":
                if "hinge" in value.keys():
                    new_left_brick = {}
                    left_brick["right"] = {"hinge": {"brick": new_left_brick}}
                    new_right_brick = right_brick["left"]["hinge"]["brick"]
                    self.mirror_right(new_left_brick, new_right_brick)

            if key == "right":
                if "hinge" in value.keys():
                    new_left_brick = {}
                    left_brick["left"] = {"hinge": {"brick": new_left_brick}}
                    new_right_brick = right_brick["right"]["hinge"]["brick"]
                    self.mirror_right(new_left_brick, new_right_brick)

        return

    # optional
    def spine_symmetry(self, spinebrick):
        spinebrick["left"] = {}

        if spinebrick["right"]:
            new_left_brick = {}
            spinebrick["left"] = {"hinge": {"brick": new_left_brick}}
            new_right_brick = spinebrick["right"]["hinge"]["brick"]
            self.mirror_right(new_left_brick, new_right_brick)

        if spinebrick["front"]:
            self.spine_symmetry(spinebrick["front"]["hinge"]["brick"])

        if "back" in spinebrick.keys():
            print("back")
            if spinebrick["back"]:
                print("hinge")
                self.spine_symmetry(spinebrick["back"]["hinge"]["brick"])
        return

    def make_core(self):
        """Create a Core with random children (front, left, right, back)."""
        core = {}
        module = self.body.core_v1
        self.queue = [(module, core, True)]
        while self.queue:
            module, current_module, spine = self.queue.pop(0)

            if spine:
                sides = ["front", "right", "back"]
            else:
                sides = ["front", "left", "right"]
            for side in sides:
                grid_position = self.body.grid_position()
                if grid_position in self.grid or len(self.grid) > config.MAX_BRICKS:
                    core[side] = {}
                    continue

                #can have up to 38 bricks with a limit of 20. because it doesnt count the doubling when its copied from right to left. other than the first time
                # Might make the body wider than longer
                if spine and side == "right":
                    self.brick_count += 2
                else:
                    self.brick_count +=1
                self.grid.append(grid_position)

                new_module = {}
                current_module[side] = {"hinge": {"brick": new_module}}
                
                if side == "front" or side == "back" and spine == True:
                    self.queue.append((new_module, True))
                else:
                    self.queue.append((new_module, False))

    def save_gene(self, gene, filename):
        with open(filename, "w") as f:
            json.dump(gene, f, indent=4)


if __name__ == "__main__":
    random.seed(743)

    generator = Gene_Generator()

    # Generate 5 random genes and save them
    for i in range(5):
        print(i)
        gene = generator.make_core()  # deeper robots if you want
        generator.save_gene(gene, f"./genes_wardrobe/gene_{i}.json")
        print(f"Saved gene_{i}.json")
