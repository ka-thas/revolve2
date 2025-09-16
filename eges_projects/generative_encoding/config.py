DATABASE_FILE = "test.sqlite"
NUM_REPETITIONS = 1
NUM_SIMULATORS = 1
NUM_PARALLEL_PROCESSES = 10

FREQUENCY = 4
ENERGY = 5000000

MODULE_IN_NEW_SPOT = 0.3
RULE_IN_NEW_SPOT = 0.5
NEW_MODULE_PLACEMENT = 0.3

LEARN_NUM_GENERATIONS = 4
NUM_RANDOM_SAMPLES = 1
KAPPA = 3  # Variation for Acquisition function (Low is exploitation, high is exploration)
ALPHA = 0  # Sampling noise
NU = 5/2  # Smoothness parameter for Matern kernel (Low is rigid, high is smooth)
LENGTH_SCALE = 0.2  # Also affects smoothness, but I'm not sure how (low is rigid, high is smooth (is it though??))
NEIGHBOUR_SCALE = 0.001

POPULATION_SIZE = 50
OFFSPRING_SIZE = 50
NUM_GENERATIONS = 500

MAX_NUMBER_OF_MODULES = 30
MUTATION_STD = 0.1
