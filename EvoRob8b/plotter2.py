""" 
Takes averaged fitness data over multiple runs and plots it

usage: python plotter_avg.py < runIDs.txt

runIDs.txt contains a list of runIDs to average over, one per line
"""

import csv
import config
import sys
import matplotlib.pyplot as plt


def plot_average_fitness(runIDs):

    # Read header from the first file to get indices
    filename = f"{config.LOG_FOLDER}{runIDs[0]}/progress.csv"
    with open(filename, "r") as f:
        reader = csv.reader(f)
        header_lookup = next(reader)
    best_fitness_idx = header_lookup.index("best_fitness")
    mean_fitness_idx = header_lookup.index("mean_fitness")
    worst_fitness_idx = header_lookup.index("worst_fitness")

    all_data_best = []
    all_data_mean = []
    all_data_worst = []

    for runID in runIDs:
        filename = f"{config.LOG_FOLDER}{runID}/progress.csv"
        with open(filename, "r") as f:
            reader = csv.reader(f)
            next(reader)  # skip header

            data_best = []
            data_mean = []
            data_worst = []
            for row in reader:
                data_best.append(float(row[best_fitness_idx]))
                data_mean.append(float(row[mean_fitness_idx]))
                data_worst.append(float(row[worst_fitness_idx]))

            all_data_best.append(data_best)
            all_data_mean.append(data_mean)
            all_data_worst.append(data_worst)

    # Transpose and average
    avg_fitness_best = [sum(gen)/len(gen) for gen in zip(*all_data_best)]
    avg_fitness_mean = [sum(gen)/len(gen) for gen in zip(*all_data_mean)]
    avg_fitness_worst = [sum(gen)/len(gen) for gen in zip(*all_data_worst)]

    # Plotting
    plt.plot(avg_fitness_best)
    plt.plot(avg_fitness_mean)
    plt.plot(avg_fitness_worst)
    plt.xlabel("Generation")
    plt.ylabel("Average Fitness")
    plt.legend(["Best Fitness", "Mean Fitness", "Worst Fitness"])
    plt.title("Average Fitness over Generations")
    plt.savefig(config.LOG_FOLDER + "plots/avg_progression.png")
    plt.close()


if __name__ == "__main__":

    runIDs = []
    for line in sys.stdin:
        runIDs.append(line.strip())

    # Read header from the first file
    first_filename = f"{config.LOG_FOLDER}{runIDs[0]}/progress.csv"
    with open(first_filename, "r") as f:
        reader = csv.reader(f)
        header_lookup = next(reader)  # skip header
    
    plot_average_fitness(runIDs)
