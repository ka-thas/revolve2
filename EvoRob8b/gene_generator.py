import json
import random
import uuid
import config

class Gene_Generator:
    def __init__ (self):
        self.max_bricks = config.MAX_BRICKS
        self.brick_count = 0

    def random_orientation(self):
        """Return a random orientation value."""
        return {"orientation": 1}


    def make_brick(self, depth, max_depth):
        """Create a Brick with random children (front, left, right)."""
        if depth >= max_depth:
            return {}  # stop recursion

        brick = {}
        for side in ["front", "left", "right"]:
            if random.random() < config.CHANCETOPLACEBRICK:  #chance to place hinge from config
                brick[side] = {
                    "hinge": {
                        "brick": make_brick(depth + 1, max_depth)
                    }
                }
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
                    left_brick["front"] = {
                        "hinge": {
                            "brick": new_left_brick
                        }
                    }
                    new_right_brick = right_brick["front"]["hinge"]["brick"]
                    mirror_right(new_left_brick, new_right_brick)
                    
                    

            if key == "left":
                if "hinge" in value.keys():
                    new_left_brick = {}
                    left_brick["right"] = {
                        "hinge": {
                            "brick": new_left_brick
                        }
                    }
                    new_right_brick = right_brick["left"]["hinge"]["brick"]
                    mirror_right(new_left_brick, new_right_brick)                
            
            if key == "right":
                if "hinge" in value.keys():
                    new_left_brick = {}
                    left_brick["left"] = {
                        "hinge": {
                            "brick": new_left_brick
                        }
                    }
                    new_right_brick = right_brick["right"]["hinge"]["brick"]
                    mirror_right(new_left_brick, new_right_brick)                

        return

    #optional
    def spine_symmetri(self, spinebrick):
        spinebrick["left"] = {}

        if spinebrick["right"]:
            new_left_brick = {}
            spinebrick["left"] = {
                "hinge": {
                    "brick": new_left_brick
                }
            }
            new_right_brick = spinebrick["right"]["hinge"]["brick"]
            mirror_right(new_left_brick, new_right_brick)                
        
        if spinebrick["front"]:
            spine_symmetri(spinebrick["front"]["hinge"]["brick"])

        if "back" in spinebrick.keys():
            print("back")
            if spinebrick["back"]:
                print("hinge")    
                spine_symmetri(spinebrick["back"]["hinge"]["brick"])
        return



    def make_core(self, max_depth=5):
        """Create a Core with random children (front, left, right, back)."""
        core = {}
        for side in ["front", "right", "back"]:
            if random.random() < config.CHANCETOPLACECORE:  #chance to place hinge from config
                core[side] = {
                    "hinge": {
                        "brick": make_brick(1, max_depth)
                    }
                }
            else:
                core[side] = {}
        
        #mirror right
        left = {}
        core["left"] = left

        """ #legacy system implementert i spine symmetri
        if core["right"] != {}:
            brick = {}
            right_brick = core["right"]["hinge"]["brick"]
            left["hinge"] = {
                "brick": brick}
            mirror_right(brick, right_brick) # Recursive
        """    
        spine_symmetri(core)


        return {"core": core}





    def save_gene(self, gene, filename):
        with open(filename, "w") as f:
            json.dump(gene, f, indent=4)


if __name__ == "__main__":
    random.seed(743)
    # Generate 5 random genes and save them
    for i in range(5):
        print(i)
        gene = make_core(max_depth=3)  # deeper robots if you want
        save_gene(gene, f"./genes_wardrobe/gene_{i}.json")
        print(f"Saved gene_{i+1}.json")
