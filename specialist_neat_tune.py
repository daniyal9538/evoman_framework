"""
    Filename:    specialist_neat_tune.py
    Author(s):   Thomas Bellucci
    Assignment:  Task 1 - Evolutionary Computing
    Description: Contains the NEAT tuning for the Evoman assignment (Task I).
"""
# Imports

import os, sys
sys.path.insert(0, 'evoman')
from environment import Environment
from controller import Controller
import numpy
import neat

from specialist_neat_training import Individual, EvomanEnvironment

text = """
[NEAT]
fitness_criterion     = max
fitness_threshold     = 100.0
pop_size              = 100
reset_on_extinction   = False

[DefaultGenome]
# node activation options
activation_default      = sigmoid
activation_mutate_rate  = 0.0
activation_options      = sigmoid

# node aggregation options
aggregation_default     = sum
aggregation_mutate_rate = 0.0
aggregation_options     = sum

# node bias options
bias_init_mean          = 0.0
bias_init_stdev         = 1.0
bias_max_value          = 30.0
bias_min_value          = -30.0
bias_mutate_power       = {}
bias_mutate_rate        = {}
bias_replace_rate       = {}

# genome compatibility options
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5

# connection add/remove rates
conn_add_prob           = {}
conn_delete_prob        = {}

# connection enable options
enabled_default         = True
enabled_mutate_rate     = 0.01

feed_forward            = True
initial_connection      = full_nodirect

# node add/remove rates
node_add_prob           = {}
node_delete_prob        = {}

# network parameters
num_hidden              = 0
num_inputs              = 20
num_outputs             = 5

# node response options
response_init_mean      = 1.0
response_init_stdev     = 0.0
response_max_value      = 30.0
response_min_value      = -30.0
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0

# connection weight options
weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_max_value        = 30
weight_min_value        = -30
weight_mutate_power     = {}
weight_mutate_rate      = {}
weight_replace_rate     = {}

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 20
species_elitism      = 1

[DefaultReproduction]
elitism            = 1
survival_threshold = {}
"""


if __name__ == '__main__':

    # SETTINGS!
    SETS = 1
    GENERATIONS = 1
    ENEMIES = [2, 3, 7]

    best_params = None
    best_score = -numpy.inf

    for i in range(SETS):

        # Sample random parameters
        parameters = numpy.round(numpy.random.uniform(0, 1, 6) * 10) / 10
        
        # Write out random config file.
        mutate_power, mutate_rate, replace_rate, conn_prob, node_prob, survival_thres = parameters
        text2 = text.format(mutate_power, mutate_rate, replace_rate,
                            conn_prob, conn_prob, node_prob, node_prob,
                            mutate_power, mutate_rate, replace_rate,
                            survival_thres)
        with open("test_neat.config", "w") as f:
            f.write(text2)

        # Evaluate on enemies
        score = 0
        for enemy in ENEMIES:

            config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                 neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                 "test_neat.config")
                
            env = EvomanEnvironment(enemy, 1, Individual=Individual)

            # Set up population and run EA for several generations.
            pop = neat.Population(config)
            winner = pop.run(env.evaluate_population, GENERATIONS)
            score += winner.fitness / len(ENEMIES)

        # Update if better param set is found
        if best_score < score:
            best_score = score
            best_params = parameters

    print("Best found: ", best_params)

        

    
