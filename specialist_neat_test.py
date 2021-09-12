"""
    Filename:   test_specialist_agent_training.py
    Author(s):  Thomas Bellucci
    Assignment: Task 1 - Evolutionary Computing
"""

import pickle
import neat

from specialist_neat_training import Individual, evaluate_individual


if __name__ == "__main__":
    # Load configuration file.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         "neat.config")

    # Load (best) genome.
    with open("best_neat_solution", "rb") as f:
        genome = pickle.load(f)

    while True:
        evaluate_individual(genome, config, headless=False)

    

    
