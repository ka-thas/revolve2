""" 
Takes averaged fitness data over multiple runs and plots it

usage: python plotter2.py < runIDs.txt

runIDs.txt contains a list of runIDs to average over, one per line
"""

import csv
import config
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem


def boxplot_fitness(runID, best_fitness):

        data = [best_fitness]

        plt.figure(figsize=(10, 6))
        plt.boxplot(data, labels=['Best Fitness', 'Worst Fitness', 'Mean Fitness'])

        plt.ylabel('Fitness')
        plt.title('Fitness Distribution')
        plt.grid()
        plt.savefig(config.LOG_FOLDER + f"{runID}/fitness_boxplot.png")
        plt.show()

def plot_average_fitness(runIDs, plotname):

    # Read header from the first file to get indices
    filename = f"{config.LOG_FOLDER}{runIDs[0]}/progress.csv"
    with open(filename, "r") as f:
        reader = csv.reader(f)
        header_lookup = next(reader)
    best_fitness_idx = header_lookup.index("best_fitness")
    mean_fitness_idx = header_lookup.index("mean_fitness")
    worst_fitness_idx = header_lookup.index("worst_fitness")

    n = len(runIDs) // 2  # Assuming half sequential, half parallel
    sequential_runs = runIDs[:n]
    parallel_runs = runIDs[n:]

    data = [sequential_runs, parallel_runs]
    labels = ['Sequential', 'Parallel']

    for i in range(2):
        all_data_best = []
        all_data_mean = []
        all_data_worst = []

        for runID in data[i]:
            filename = f"{config.LOG_FOLDER}{runID}/progress.csv"
            with open(filename, "r") as f:
                reader = csv.reader(f)
                next(reader)  # skip header

                data_best = []
                data_mean = []
                data_worst = []
                gen = 0
                for row in reader:
                    data_best.append(float(row[best_fitness_idx]))
                    data_mean.append(float(row[mean_fitness_idx]))
                    data_worst.append(float(row[worst_fitness_idx]))
                    gen += 1
                    if gen >= 26:
                        break

                all_data_best.append(data_best)
                all_data_mean.append(data_mean)
                all_data_worst.append(data_worst)

        avg_fitness_best = [sum(gen)/len(gen) for gen in zip(*all_data_best)] # Transpose and average

        std_error_best = [sem(gen) for gen in zip(*all_data_best)] # Calculate standard error for each fitness

        # Plotting
        x = np.arange(len(avg_fitness_best))

        plt.plot(x, avg_fitness_best, 'g' if i == 0 else 'b', label=labels[i])
        plt.fill_between(x, 
                        np.array(avg_fitness_best) - np.array(std_error_best), 
                        np.array(avg_fitness_best) + np.array(std_error_best), 
                        color='g' if i == 0 else 'b', alpha=0.2)


    plt.xlabel("Generation")
    plt.ylabel("Average Fitness")
    plt.legend()

    plt.savefig(config.LOG_FOLDER + "plots/" + plotname + ".svg")
    plt.close()


if __name__ == "__main__":

    runIDs = []
    for line in sys.stdin:
        runIDs.append(line.strip())

    plotname = runIDs[0]

    # Read header from the first file
    first_filename = f"{config.LOG_FOLDER}{runIDs[1]}/progress.csv"
    with open(first_filename, "r") as f:
        reader = csv.reader(f)
        header_lookup = next(reader)  # skip header
    
    plot_average_fitness(runIDs[1:], plotname)
