import json

def parse_gene(node, depth=0):
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

    parse_gene(gene)