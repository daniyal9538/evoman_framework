"""
    Filename:   specialist_neat_test.py
    Author(s):  Thomas Bellucci
    Assignment: Task 1 - Evolutionary Computing
"""

from evoman_framework.specialist_neat_training import ENEMIES, INDIVIDUAL_TYPE
import pickle
import argparse
import neat

from specialist_neat_training import Individual, EvomanEnvironment, Individual_RNN


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--best', help='full path to best player', type = int)
    # parser.add_argument('--generations', help = 'number of generations EA will run', default=15,type=int)
    # parser.add_argument('--enemies', help = 'comma seperated types of enemies', default=4)
    # parser.add_argument('--individual_type', help='type of individual (nn) (1: ff nn, 2: rnn)', default=1,type=int)

    args = parser.parse_args()
 

    # SETTINGS!
    FILE = args.best
    segmented_file_name =  FILE.split('-')[-3:]
    config_vars = [int(i[0]) for i in segmented_file_name]
    RUNS = config_vars[0]
    ENEMIES = config_vars[1]
    # ENEMIES = [int(i) for i in str(args.enemies).split(',')]
    INDIVIDUAL_TYPE = config_vars[2]


    


    # Load configuration file.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         "neat.config")

    # Load (best) genome.
    # filename = "neat_best_run-{}_enemy-{}-{}.pkl".format(RUN, ENEMY, INDIVIDUAL_TYPE)
    with open(FILE, "rb") as f:
        genome = pickle.load(f)
    print(f'{FILE} succesfully loaded with config:\nEnemy type {ENEMIES}\nNumber of runs: {RUNS}\nIndividual type: {INDIVIDUAL_TYPE}')
    # Run game
    if INDIVIDUAL_TYPE == 1:
        env = EvomanEnvironment(ENEMIES, RUNS, Individual=Individual)
    if INDIVIDUAL_TYPE == 2:
        env = EvomanEnvironment(ENEMIES, RUNS, Individual=Individual_RNN)
    while True:
        fitness, gain = env.evaluate_individual(genome, config, show=True)
        print(fitness, gain)

    

    
