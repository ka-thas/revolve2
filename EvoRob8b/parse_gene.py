import json

from revolve2.modular_robot.body.v1 import ActiveHingeV1, BodyV1, BrickV1

def build_body():
    body = BodyV1
    

def print_gene(node, depth=0):
    """Recursively parses the gene structure and prints it."""
    indent = "  " * depth
    if not isinstance(node, dict) or not node:  # leaf node
        return
    
    for key, value in node.items():
        print(f"{indent}- {key}")
        parse_gene(value, depth + 1)


if __name__ == "__main__":
    
    with open("./testgene.json", "r") as f:
        gene = json.load(f)
    
    print("Gene Structure:")

    print_gene(gene)