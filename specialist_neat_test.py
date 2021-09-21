import pickle
import argparse
import neat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


    # ENEMY = 4
    # RUN = 1
    # INDIVIDUAL_TYPE = 2
    
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
    # filename = "neat_best_run-{}_enemy-{}_ind-{}.pkl".format(RUN, ENEMY, str(INDIVIDUAL_TYPE))
    # with open(filename, "rb") as f:
    #     genome = pickle.load(f)
    # print(f'{FILE} succesfully loaded with config:\nEnemy type {ENEMIES}\nNumber of runs: {RUNS}\nIndividual type: {INDIVIDUAL_TYPE}')

    # # Run game
    # env = EvomanEnvironment(ENEMIES, RUNS, Individual=Individual)
    # while True:
    #     fitness, gain = env.evaluate_individual(genome, config, show=True)
    #     print(fitness, gain)
arrs = {}
for x in [2, 3, 7]:

    enemy_arr = []

    for y in range(1, 11):

        filename = "neat_best_run-{}_enemy-{}_ind-{}.pkl".format(y, x, str(INDIVIDUAL_TYPE))
        with open(filename, "rb") as f:
            genome = pickle.load(f)

        fitness_array = [0 for i in range(5)]

        # Run game
        env = EvomanEnvironment(x, y)
        for i in range(5):
            fitness, gain = env.evaluate_individual(genome, config, show=False)
            # print(fitness, gain)
            fitness_array[i] += fitness

        print(f'run {y}, enemy {x}, mean fitness {np.mean(fitness_array)}')

        enemy_arr.append(np.mean(fitness_array))
    arrs[f'Enemey {x}'] = enemy_arr

df = pd.DataFrame(arrs)
df.plot.box(title=f'Fitness Boxplot', ylabel = 'fitness scores', xlabel=' ')
plt.show()
    





    

    
