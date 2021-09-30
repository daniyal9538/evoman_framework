import pickle
import neat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from specialist_neat_training import Individual, EvomanEnvironment


if __name__ == "__main__":
    ENEMY = 7
    RUN = 1
    INDIVIDUAL_TYPE = 1
    
    # Load configuration file.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         "neat.config")

    # Load (best) genome.
arrs = {}
for x in [2, 3, 7]:

    enemy_arr = []

    for y in range(1, 11):

        if INDIVIDUAL_TYPE == 1:
            filename = "neat_best_not_fixed_topology/neat_best_run-{}_enemy-{}_ind-{}.pkl".format(y, x, str(INDIVIDUAL_TYPE))
        else:
            filename = "neat_best_fixed_topology/neat_fixed_best_run-{}_enemy-{}_ind-{}.pkl".format(y, x, str(INDIVIDUAL_TYPE))
        with open(filename, "rb") as f:
            genome = pickle.load(f)

        fitness_array = [0 for i in range(5)]

        # Run game
        env = EvomanEnvironment(x, y)
        for i in range(5):
            fitness, gain = env.evaluate_individual(genome, config, show=True)
            # print(fitness, gain)
            fitness_array[i] += fitness

        print(f'run {y}, enemy {x}, mean fitness {np.mean(fitness_array)}')

        enemy_arr.append(np.mean(fitness_array))
    arrs[f'Enemy {x}'] = enemy_arr

df = pd.DataFrame(arrs)
df.plot.box(title=f'Fitness Boxplot', ylabel = 'fitness scores', xlabel=' ')
plt.show()




    

    
