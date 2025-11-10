import csv
import os
import config
import time
import matplotlib.pyplot as plt

class Plotter:
    """ gathers data for plotting after ea """
    def __init__(self, filename: str = "", runID: str = ""):
        self.filename = filename
        self.runID = runID

        self.generations = []
        self.best_fitness = []
        self.best_fitness_flat = []
        self.best_fitness_uneven = []
        self.best_fitness_crater = []

        self.worst_fitness = []
        self.worst_fitness_flat = []
        self.worst_fitness_uneven = []
        self.worst_fitness_crater = []

        self.mean_fitness = []
        self.mean_fitness_flat  = []
        self.mean_fitness_uneven = []
        self.mean_fitness_crater = []


        self.median_fitness = []
        self.median_fitness_flat   = []
        self.median_fitness_uneven = []
        self.median_fitness_crater = []

        self.std = []
        self.num_modules_in_best_individual = []
        self.total_elapsed_time = []
        self.time_per_generation = []


    def log_generation(
            self, generation: int, 
            best: float,
            best_flat  : float,
            best_uneven: float,
            best_crater: float,

            worst: float, 
            worst_flat  : float, 
            worst_uneven: float, 
            worst_crater: float, 

            mean: float, 
            mean_flat  : float, 
            mean_uneven: float, 
            mean_crater: float, 

            median: float,
            median_flat  : float,
            median_uneven: float,
            median_crater: float,
 
            std: float, 
            num_modules: int, 
            total_elapsed_time: float
            ):
        self.generations.append(generation)
        self.best_fitness.append(best)
        self.best_fitness_flat.append(best_flat)
        self.best_fitness_uneven.append(best_uneven)
        self.best_fitness_crater.append(best_crater)


        self.worst_fitness.append(worst)
        self.worst_fitness_flat.append(worst_flat)
        self.worst_fitness_uneven.append(worst_uneven)
        self.worst_fitness_crater.append(worst_crater)

        self.mean_fitness.append(mean)
        self.mean_fitness_flat.append(mean_flat)
        self.mean_fitness_uneven.append(mean_uneven)
        self.mean_fitness_crater.append(mean_crater)

        self.median_fitness.append(median)
        self.median_fitness_flat.append(median_flat)
        self.median_fitness_uneven.append(median_uneven)
        self.median_fitness_crater.append(median_crater)

        self.std.append(std)
        self.num_modules_in_best_individual.append(num_modules)
        self.total_elapsed_time.append(total_elapsed_time)

        if generation == 0:
            self.time_per_generation.append(total_elapsed_time)
        else:
            self.time_per_generation.append(total_elapsed_time - self.total_elapsed_time[generation - 1])

    def save_to_csv(self, filename: str):
        # Use newline='' to avoid extra blank lines on some platforms
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["generation", "best_fitness", "worst_fitness", "mean_fitness", "median_fitness", "std_fitness", "num_modules", "time_per_generation"])
            for i in range(len(self.generations)):
                writer.writerow([
                    self.generations[i],
                    self.best_fitness[i],
                    self.worst_fitness[i],
                    self.mean_fitness[i],
                    self.median_fitness[i],
                    self.std[i],
                    self.num_modules_in_best_individual[i],
                    self.time_per_generation[i]
                ])

    def append_last_n_to_csv(self, filename: str, n: int = 10):
        """Append the last `n` logged generations to a CSV file.

        If the file does not exist, a header will be written first.
        """
        if n <= 0:
            return

        start = max(0, len(self.generations) - n)
        if start >= len(self.generations):
            # Nothing to append
            return

        write_header = not os.path.exists(filename)

        # Use newline='' to avoid extra blank lines on some platforms
        with open(filename, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["generation", "best_fitness", "best_fitness_flat", "best_fitness_uneven", "best_fitness_crater", "worst_fitness","worst_fitness_flat","worst_fitness_uneven","worst_fitness_crater", "mean_fitness","mean_fitness_flat","mean_fitness_uneven","mean_fitness_crater", "median_fitness","median_fitness_flat","median_fitness_uneven","median_fitness_crater", "std_fitness", "num_modules", "time_per_generation", "total_elapsed_time"])

            for i in range(start, len(self.generations)):
                writer.writerow([
                    self.generations[i],
                    self.best_fitness[i],
                    self.best_fitness_flat[i],
                    self.best_fitness_uneven[i],
                    self.best_fitness_crater[i],

                    self.worst_fitness[i],
                    self.worst_fitness_flat[i],
                    self.worst_fitness_uneven[i],
                    self.worst_fitness_crater[i],

                    self.mean_fitness[i],
                    self.mean_fitness_flat[i],
                    self.mean_fitness_uneven[i],
                    self.mean_fitness_crater[i],

                    self.median_fitness[i],
                    self.median_fitness_flat[i],
                    self.median_fitness_uneven[i],
                    self.median_fitness_crater[i],

                    self.std[i],
                    self.num_modules_in_best_individual[i],
                    self.time_per_generation[i],
                    self.total_elapsed_time[i]
                ])

    def load_from_csv(self, filename: str):
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                self.generations.append(int(row[0]))
                self.best_fitness.append(float(row[1]))
                self.best_fitness_flat.append(float(row[2]))
                self.best_fitness_uneven.append(float(row[3]))
                self.best_fitness_crater.append(float(row[4]))

                self.worst_fitness.append(float(row[5]))
                self.worst_fitness_flat.append(float(row[6]))
                self.worst_fitness_uneven.append(float(row[7]))
                self.worst_fitness_crater.append(float(row[8]))

                self.mean_fitness.append(float(row[9]))
                self.mean_fitness_flat.append(float(row[10]))
                self.mean_fitness_uneven.append(float(row[11]))
                self.mean_fitness_crater.append(float(row[12]))

                self.median_fitness.append(float(row[13]))
                self.median_fitness_flat.append(float(row[14]))
                self.median_fitness_uneven.append(float(row[15]))
                self.median_fitness_crater.append(float(row[16]))

                self.std.append(float(row[17]))
                self.num_modules_in_best_individual.append(int(row[18]))

    def plot_best_mean_worst(self):

        plt.figure(figsize=(10, 6))
        plt.plot(self.generations, self.best_fitness, label='Best Fitness')
        plt.plot(self.generations, self.worst_fitness, label='Worst Fitness')
        plt.plot(self.generations, self.mean_fitness, label='Mean Fitness')
        
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('EA progression')
        plt.legend()
        plt.grid()
        plt.savefig(config.LOG_FOLDER + f"{self.runID}/fitness_progression.png")
        if show_plots: plt.show()


    def plot_best_mean_worst_flat(self):

        plt.figure(figsize=(10, 6))
        plt.plot(self.generations  ,  self.best_fitness_flat  , label='Best Fitness')
        plt.plot(self.generations, self.worst_fitness_flat, label='Worst Fitness')
        plt.plot(self.generations,  self.mean_fitness_flat, label='Mean Fitness')
        
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('EA progression, flat environment')
        plt.legend()
        plt.grid()
        plt.savefig(config.LOG_FOLDER + f"{self.runID}/fitness_progression_flat.png")
        if show_plots: plt.show()



    def plot_best_mean_worst_uneven(self):

        plt.figure(figsize=(10, 6))
        plt.plot(self.generations,  self.best_fitness_uneven  , label='Best Fitness')
        plt.plot(self.generations, self.worst_fitness_uneven, label='Worst Fitness')
        plt.plot(self.generations,  self.mean_fitness_uneven, label='Mean Fitness')
        
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('EA progression, uneven environment')
        plt.legend()
        plt.grid()
        plt.savefig(config.LOG_FOLDER + f"{self.runID}/fitness_progression_uneven.png")
        if show_plots: plt.show()



    def plot_best_mean_worst_crater(self):

        plt.figure(figsize=(10, 6))
        plt.plot(self.generations  ,self.best_fitness_crater  , label='Best Fitness')
        plt.plot(self.generations, self.worst_fitness_crater, label='Worst Fitness')
        plt.plot(self.generations,  self.mean_fitness_crater, label='Mean Fitness')
        
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('EA progression, crater environment')
        plt.legend()
        plt.grid()
        plt.savefig(config.LOG_FOLDER + f"{self.runID}/fitness_progression_crater.png")
        if show_plots: plt.show()




    def boxplot_fitness(self):

        data = [self.best_fitness, self.worst_fitness, self.mean_fitness]

        plt.figure(figsize=(10, 6))
        plt.boxplot(data, labels=['Best Fitness', 'Worst Fitness', 'Mean Fitness'])

        plt.ylabel('Fitness')
        plt.title('Fitness Distribution')
        plt.grid()
        plt.savefig(config.LOG_FOLDER + f"{self.runID}/fitness_boxplot.png")
        if show_plots: plt.show()
    
    def plot_std(self):

        plt.figure(figsize=(10, 6))
        plt.plot(self.generations, self.std, label='Standard Deviation')

        plt.xlabel('Generation')
        plt.ylabel('Standard Deviation')
        plt.title('EA progression')
        plt.legend()
        plt.grid()
        plt.savefig(config.LOG_FOLDER + f"{self.runID}/std_progression.png")
        if show_plots: plt.show()

    def plot_num_modules(self):

        plt.figure(figsize=(10, 6))
        plt.plot(self.generations, self.num_modules_in_best_individual, label='Num Modules in Best Individual')

        plt.xlabel('Generation')
        plt.ylabel('Number of modules')
        plt.title('EA progression')
        plt.legend()
        plt.grid()
        plt.savefig(config.LOG_FOLDER + f"{self.runID}/num_modules_progression.png")
        if show_plots: plt.show()

    def plot_time_per_generation(self):

        plt.figure(figsize=(10, 6))
        plt.plot(self.generations, self.time_per_generation, label='Time per Generation')

        plt.xlabel('Generation')
        plt.ylabel('Time (seconds)')
        plt.title('Time per Generation')
        plt.legend()
        plt.grid()
        plt.savefig(config.LOG_FOLDER + f"{self.runID}/time_per_generation.png")
        if show_plots: plt.show()

if __name__ == "__main__":
    """ Cannot run over ssh bc of plotting """
    run_id = input("> run ID: ")
    show_plots = input("> Show plots? [y/n]: ") == "y"
    plotter = Plotter(runID=run_id)
    plotter.load_from_csv(config.LOG_FOLDER + f"{run_id}/progress.csv")
    plotter.plot_best_mean_worst()
    plotter.plot_best_mean_worst_flat()
    plotter.plot_best_mean_worst_uneven()
    plotter.plot_best_mean_worst_crater()
    plotter.plot_num_modules()
    #plotter.boxplot_fitness()
    #plotter.plot_std()
    #plotter.plot_time_per_generation()