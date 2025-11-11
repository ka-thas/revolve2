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


def boxplot_fitness(runID):

        data = [self.best_fitness]

        plt.figure(figsize=(10, 6))
        plt.boxplot(data, labels=['Best Fitness', 'Worst Fitness', 'Mean Fitness'])

        plt.ylabel('Fitness')
        plt.title('Fitness Distribution')
        plt.grid()
        plt.savefig(config.LOG_FOLDER + f"{self.runID}/fitness_boxplot.png")
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
            gen = 0
            for row in reader:
                data_best.append(float(row[best_fitness_idx]))
                data_mean.append(float(row[mean_fitness_idx]))
                data_worst.append(float(row[worst_fitness_idx]))
                gen += 1

            all_data_best.append(data_best)
            all_data_mean.append(data_mean)
            all_data_worst.append(data_worst)

    # Transpose and average
    avg_fitness_best = [sum(gen)/len(gen) for gen in zip(*all_data_best)]
    avg_fitness_mean = [sum(gen)/len(gen) for gen in zip(*all_data_mean)]
    avg_fitness_worst = [sum(gen)/len(gen) for gen in zip(*all_data_worst)]

    # Calculate standard error for each fitness
    std_error_best = [sem(gen) for gen in zip(*all_data_best)]
    std_error_mean = [sem(gen) for gen in zip(*all_data_mean)]
    std_error_worst = [sem(gen) for gen in zip(*all_data_worst)]

    # Plotting
    x = np.arange(len(avg_fitness_best ))
    y = np.arange(len(avg_fitness_mean ))
    z = np.arange(len(avg_fitness_worst))

    plt.plot(x, avg_fitness_best, 'g', label="Best Fitness")
    plt.plot(x, avg_fitness_mean, 'b', label="Mean Fitness")
    plt.plot(x, avg_fitness_worst, 'r', label="Worst Fitness")
    plt.fill_between(x, 
                     np.array(avg_fitness_best) - np.array(std_error_best), 
                     np.array(avg_fitness_best) + np.array(std_error_best), 
                     color='g', alpha=0.2)

    plt.fill_between(x, 
                     np.array(avg_fitness_mean) - np.array(std_error_mean), 
                     np.array(avg_fitness_mean) + np.array(std_error_mean), 
                     color='b', alpha=0.2)

    plt.fill_between(x, 
                     np.array(avg_fitness_worst) - np.array(std_error_worst), 
                     np.array(avg_fitness_worst) + np.array(std_error_worst), 
                     color='r', alpha=0.2)
    plt.plot(y, avg_fitness_mean, 'b', label="Mean fitness")
    plt.plot(z, avg_fitness_worst, 'r', label=" Worst fitness")
    plt.ylim(0, 10)
    plt.xlim(0, 32)
    plt.xlabel("Generation")
    plt.ylabel("Average Fitness")

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
