"""
    Filename:   specialist_neat_test.py
    Author(s):  Thomas Bellucci
    Assignment: Task 1 - Evolutionary Computing
"""

import pickle
import neat

from specialist_neat_training import Individual, EvomanEnvironment


if __name__ == "__main__":
    ENEMY = 4
    RUN = 1
    
    # Load configuration file.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         "neat.config")

    # Load (best) genome.
    filename = "neat_best_run-{}_enemy-{}.pkl".format(RUN, ENEMY)
    with open(filename, "rb") as f:
        genome = pickle.load(f)

    # Run game
    env = EvomanEnvironment(ENEMY, RUN)
    while True:
        fitness, gain = env.evaluate_individual(genome, config, show=True)
        print(fitness, gain)

    

    
