""" 
Takes averaged fitness data over multiple runs and plots it
Takes both sequential and parallel runs
One line per terrain type

usage: python plotter2.py < runIDs.txt

runIDs.txt contains a list of runIDs to average over, one per line
Half of them need to be sequential runs, the other half parallel runs
The first line is the name of the plot to be saved as
"""

import csv
import config
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem

def plot_average_fitness(runIDs):

    # Read header from the first file to get indices
    filename = f"{config.LOG_FOLDER}{runIDs[0]}/progress.csv"
    with open(filename, "r") as f:
        reader = csv.reader(f)
        header_lookup = next(reader)

    best_fitness_flat_idx = header_lookup.index("best_fitness_flat")
    best_fitness_uneven_idx = header_lookup.index("best_fitness_uneven")
    best_fitness_crater_idx = header_lookup.index("best_fitness_crater")

    n = len(runIDs) // 2  # Assuming half sequential, half parallel
    sequential_runs = runIDs[:n]
    parallel_runs = runIDs[n:]

    data = [sequential_runs, parallel_runs]
    labels = ['S', 'P']

    for i in range(2):
        all_data_best_flat = []
        all_data_best_uneven = []
        all_data_best_crater = []

        for runID in data[i]:
            filename = f"{config.LOG_FOLDER}{runID}/progress.csv"
            with open(filename, "r") as f:
                reader = csv.reader(f)
                next(reader)  # skip header

                data_best_flat = []
                data_best_uneven = []
                data_best_crater = []

                gen = 0
                for row in reader:
                    data_best_flat.append(float(row[best_fitness_flat_idx]))
                    data_best_uneven.append(float(row[best_fitness_uneven_idx]))
                    data_best_crater.append(float(row[best_fitness_crater_idx]))
                    gen += 1
                    if gen >= generations:
                        break  # Limit to 41 generations

                all_data_best_flat.append(data_best_flat)
                all_data_best_uneven.append(data_best_uneven)
                all_data_best_crater.append(data_best_crater)

        avg_fitness_best_flat = [sum(gen)/len(gen) for gen in zip(*all_data_best_flat)] # Transpose and average
        avg_fitness_best_uneven = [sum(gen)/len(gen) for gen in zip(*all_data_best_uneven)] # Transpose and average
        avg_fitness_best_crater = [sum(gen)/len(gen) for gen in zip(*all_data_best_crater)] # Transpose and average
        # Plotting
        x = np.arange(generations)

        plt.plot(x, avg_fitness_best_flat, "#e83b20", label=labels[i] + ' Flat', ls='-' if i == 0 else '--')
        plt.plot(x, avg_fitness_best_uneven, "#27eb4b", label=labels[i] + ' Uneven', ls='-' if i == 0 else '--')
        plt.plot(x, avg_fitness_best_crater, "#586fdf", label=labels[i] + ' Crater', ls='-' if i == 0 else '--')


    plt.xlabel("Generation")
    plt.ylabel("Average Fitness")
    plt.legend()
    plt.axis([0, 41, 0, 6])

    plt.savefig(config.LOG_FOLDER + "plots/" + plotname + ".svg")
    plt.close()


if __name__ == "__main__":

    runIDs = []
    for line in sys.stdin:
        runIDs.append(line.strip())

    plotname = runIDs[0]
    generations = 41
    
    plot_average_fitness(runIDs[1:])
