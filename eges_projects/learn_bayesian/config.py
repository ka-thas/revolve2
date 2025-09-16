"""Configuration parameters for this example."""

DATABASE_FILE = "test.sqlite"
DATABASE_FILE_OLD = "old_file.sqlite"
ENVIRONMENT = "noisy"
NUM_SIMULATORS = 1
NUM_PARALLEL_PROCESSES = 200
NUM_GENERATIONS = 1000
NUM_RANDOM_SAMPLES = 0
FREQUENCY = 4
MAX_NUMBER_OF_MODULES = 20
INIT_MIN_MODULES = 15
INIT_MAX_MODULES = 20
RUNS = 10

KAPPA = 3  # Variation for Acquisition function (Low is exploitation, high is exploration)
ALPHA = 0  # Sampling noise
NU = 5/2  # Smoothness parameter for Matern kernel (Low is rigid, high is smooth)
LENGTH_SCALE = 0.2  # Also affects smoothness, but I'm not sure how (low is rigid, high is smooth (is it though??))
NEIGHBOUR_SCALE = 0.001
ENERGY = 100000
