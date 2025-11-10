import numpy as np

FILENAME = "./experiments/human_made/ka.csv"

with open(FILENAME, "r") as f:
    lines = f.readlines()

fitnesses = []
for line in lines[1:]:
    parts = line.strip().split(",")
    fitness_flat = float(parts[1])
    fitness_uneven = float(parts[2])
    fitness_crater = float(parts[3])
    fitness_total = float(parts[4])
    fitnesses.append([fitness_flat, fitness_uneven, fitness_crater, fitness_total])

avg_fitness = np.mean(fitnesses, axis=0)
print(f"\n->> Average fitness over 10 runs: flat={avg_fitness[0]}, uneven={avg_fitness[1]}, crater={avg_fitness[2]}, total={avg_fitness[3]}")
