"""
    Filename:    specialist_neat_test.py
    Author(s):   Thomas Bellucci
    Assignment:  Task 1 - Evolutionary Computing
    Description: Executes the best found solutions in the environment.
"""
import pickle
import neat

from specialist_neat_training import Individual, EvomanEnvironment


if __name__ == "__main__":
    ENEMY = 3
    RUN = 1
    INDIVIDUAL_TYPE = 1
    
    # Load configuration file.
    if INDIVIDUAL_TYPE == 1:
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             "neat.config")
    else:
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         "neat_fixed_topology.config")

    # Load (best) genome.
    filename = "solutions/neat_best_run-{}_enemy-{}_ind-{}.pkl".format(RUN, ENEMY, INDIVIDUAL_TYPE)
    with open(filename, "rb") as f:
        genome = pickle.load(f)

    # Run game.
    env = EvomanEnvironment(ENEMY, RUN)
    while True:
        env.evaluate_individual(genome, config, show=True)


    

    
