import csv

class Plotter:
    """ gathers data for plotting after ea """
    def __init__(self):
        self.generations = []
        self.best_fitness = []
        self.worst_fitness = []
        self.mean_fitness = []
        self.median_fitness = []
        self.std = []
        self.num_modules_in_best_individual = []

    def log_generation(self, generation: int, best: float, worst: float, mean: float, median: float, std: float, num_modules: int):
        self.generations.append(generation)
        self.best_fitness.append(best)
        self.worst_fitness.append(worst)
        self.mean_fitness.append(mean)
        self.median_fitness.append(median)
        self.std.append(std)
        self.num_modules_in_best_individual.append(num_modules)

    def save_to_csv(self, filename: str):
        with open(filename, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["generation", "best_fitness", "worst_fitness", "mean_fitness", "median_fitness", "std_fitness", "num_modules"])
            for i in range(len(self.generations)):
                writer.writerow([
                    self.generations[i],
                    self.best_fitness[i],
                    self.worst_fitness[i],
                    self.mean_fitness[i],
                    self.median_fitness[i],
                    self.std[i],
                    self.num_modules_in_best_individual[i]
                ])

    def load_from_csv(self, filename: str):
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                self.generations.append(int(row[0]))
                self.best_fitness.append(float(row[1]))
                self.worst_fitness.append(float(row[2]))
                self.mean_fitness.append(float(row[3]))
                self.median_fitness.append(float(row[4]))
                self.std.append(float(row[5]))
                self.num_modules_in_best_individual.append(int(row[6]))

    def plot(self):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.plot(self.generations, self.best_fitness, label='Best Fitness')
        plt.plot(self.generations, self.mean_fitness, label='Mean Fitness')
        plt.plot(self.generations, self.worst_fitness, label='Worst Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('EA progression')
        plt.legend()
        plt.grid()
        plt.show()