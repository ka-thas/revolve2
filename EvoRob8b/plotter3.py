""" 
Takes averaged fitness data over multiple runs and plots it
Both sequential and parallel runs

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
                    if gen >= generations:
                        break

                all_data_best.append(data_best)
                all_data_mean.append(data_mean)
                all_data_worst.append(data_worst)

        avg_fitness_best = [sum(gen)/len(gen) for gen in zip(*all_data_best)] # Transpose and average
        avg_fitness_mean = [sum(gen)/len(gen) for gen in zip(*all_data_mean)] 
        avg_fitness_worst = [sum(gen)/len(gen) for gen in zip(*all_data_worst)]

        std_error_best = [sem(gen) for gen in zip(*all_data_best)] # Calculate standard error for each fitness
        std_error_mean = [sem(gen) for gen in zip(*all_data_mean)]
        std_error_worst = [sem(gen) for gen in zip(*all_data_worst)]

        # Plotting
        x = np.arange(len(avg_fitness_best))

        plt.plot(x, avg_fitness_best, "#e71515" if i == 0 else "#2555e5", label=labels[i], ls='-' if i == 0 else '--')
        plt.fill_between(x, 
                        np.array(avg_fitness_best) - np.array(std_error_best), 
                        np.array(avg_fitness_best) + np.array(std_error_best), 
                        color="#e71515" if i == 0 else "#2555e5", alpha=0.2)
        
        plt.plot(x, avg_fitness_mean, '#c82828' if i == 0 else '#3131b9', label=labels[i] + ' Mean', ls='-' if i == 0 else '--')
        plt.fill_between(x, 
                        np.array(avg_fitness_mean) - np.array(std_error_mean), 
                        np.array(avg_fitness_mean) + np.array(std_error_mean), 
                        color='#c82828' if i == 0 else '#3131b9', alpha=0.2)
        
        """ plt.plot(x, avg_fitness_worst, 'y' if i == 0 else 'm', label=labels[i] + ' Worst')
        plt.fill_between(x, 
                        np.array(avg_fitness_worst) - np.array(std_error_worst), 
                        np.array(avg_fitness_worst) + np.array(std_error_worst), 
                        color='y' if i == 0 else 'm', alpha=0.2) """

    plt.xlabel("Generation")
    plt.ylabel("Average Fitness")
    plt.legend()
    plt.axis([0, generations, 0, None])

    plt.savefig(config.LOG_FOLDER + "plots/" + plotname + ".svg")
    plt.close()


if __name__ == "__main__":

    runIDs = []
    for line in sys.stdin:
        runIDs.append(line.strip())

    plotname = runIDs[0]


    least_generations = float('inf')
    # First, determine the least number of generations across all runs
    for runID in runIDs[1:]:
        filename = f"{config.LOG_FOLDER}{runID}/progress.csv"
        with open(filename, "r") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            gen = sum(1 for row in reader)
            if gen < least_generations:
                least_generations = gen
    generations = least_generations
    
    plot_average_fitness(runIDs[1:], plotname)
