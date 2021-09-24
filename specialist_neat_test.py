# from evoman_framework.specialist_neat_training import ENEMIES, INDIVIDUAL_TYPE
import pickle
import argparse
import neat

from specialist_neat_training import Individual, EvomanEnvironment


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--best', help='full path to best player', type = str)
    args = parser.parse_args()
 

    # SETTINGS!
    FILE = args.best
    config_vars = [int(i[0]) for i in FILE.replace(".pkl", "").split('-')[-3:]]
    RUNS = config_vars[0]
    ENEMIES = config_vars[1]
    INDIVIDUAL_TYPE = config_vars[2]


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
    with open(FILE, "rb") as f:
        genome = pickle.load(f)
    print(f'{FILE} succesfully loaded with config:\nEnemy type {ENEMIES}\nNumber of runs: {RUNS}\nIndividual type: {INDIVIDUAL_TYPE}')

    # Run game
    env = EvomanEnvironment(ENEMIES, RUNS, Individual=Individual)
    while True:
        fitness, gain = env.evaluate_individual(genome, config, show=True)
        print(fitness, gain)

    

    
