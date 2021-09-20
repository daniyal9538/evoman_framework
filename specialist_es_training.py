"""
    Filename:    specialist_es_training.py
    Author(s):   Thomas Bellucci
    Assignment:  Task 1 - Evolutionary Computing
    Description: Contains the Separable CMA-ES implementation
                 (EA 2) for the Evoman assignment (Task I).
"""
# Imports
import os, sys
sys.path.insert(0, 'evoman')
from environment import Environment
from controller import Controller
import pickle
import numpy
import tqdm
import glob
from cmaes import SepCMA as CMA


# Settings
SHAPE = (20, 32, 5)
RUNS = 1
GENERATIONS = 15
ENEMIES = [4]
POP_SIZE = 50


""" Implements the player controller using NEAT.
"""
class Individual(Controller):
    def __init__(self, genome):
        # Build network from genome (the genotype).
        self.w0 = genome[:SHAPE[0]*SHAPE[1]].reshape(SHAPE[1], SHAPE[0])
        self.w1 = genome[SHAPE[0]*SHAPE[1]:].reshape(SHAPE[2], SHAPE[1])

    def control(self, state, _):
        def sig(x):
            return 1 / (1 + numpy.exp(-x))

        h1 = sig(self.w0.dot(numpy.array(state)))
        y = sig(self.w1.dot(h1))
        return y > .5


""" Wrapper around the Evoman game environment. It contains the
    fitness function (evaluate_individual) and writes out per-
    generation stats for a run of the algorithm.
"""
class EvomanEnvironment:
    def __init__(self, enemy, run, outfile=None):
        self.enemy = enemy
        self.run = run
        self.outfile = outfile

        # Create outfile for stats when an outfile is given.
        if outfile is not None: 
            with open(outfile, "w") as f: # Create new file with header
                f.write("mean,max,enemy,run\n")
            
        
    def evaluate_individual(self, genome, show=False):
        """ Evaluates the phenotype-converted genome (genotype) by simulating
            a single round of Evoman.
        """
        # Build individual (controller/phenotype) from genome.
        controller = Individual(genome)

        # Set show = False to hide visuals and speed up learning.
        if not show:
            os.environ["SDL_VIDEODRIVER"] = 'dummy'
            env = Environment(experiment_name=None,
                              player_controller=controller,
                              speed='fastest',
                              logs='off')

        # Set show = True for real time gameplay (for testing).
        else:
            env = Environment(experiment_name=None,
                              player_controller=controller,
                              speed='normal',
                              logs='off')

        # Select enemy (1 to 8)
        env.update_parameter('enemies', [self.enemy])

        # Simulate game.
        fitness, player_life, enemy_life, game_duration = env.play()
        ind_gain = player_life - enemy_life
        return fitness, ind_gain


    def evaluate_population(self, population):
        """ Wrapper function to evaluate all individuals in the population.
        """
        # Run fitness function on all individuals in population (and store fitnesses).
        fitness_scores = []
        for ind in tqdm.tqdm(population):
            fitness, _ = self.evaluate_individual(ind)
            fitness_scores.append(fitness)

        # Print stats of generation
        avg_fitness = numpy.mean(fitness_scores)
        max_fitness = numpy.max(fitness_scores)
        print("Stats:", avg_fitness, max_fitness)

        # Write mean and max fitness stats of population to file (when enabled).
        if self.outfile is not None:
            with open(self.outfile, "a") as f:
                f.write("{},{},{},{}\n".format(avg_fitness, max_fitness, self.enemy, self.run))

        return list(zip(population, fitness_scores))



if __name__ == '__main__':

    # Initialize network
    num_weights = sum(SHAPE[i] * SHAPE[i+1] for i in range(len(SHAPE) - 1))
    init_weights = np.random.uniform(-30, 30, num_weights)

    # Run EA for 3 enemies and 10 runs.
    for enemy in ENEMIES:
        for run in range(1, RUNS + 1):

            # Setup Evoman environment
            outfile = "es_stats_run-{}_enemy-{}.csv".format(run, enemy)
            env = EvomanEnvironment(enemy, run, outfile)
            
            # Set up population and run EA for several generations.
            optimizer = CMA(mean=init_weights, sigma=1, population_size=POP_SIZE)
            for gen in range(GENERATIONS):
                
                genomes = []
                for _ in range(optimizer.population_size):
                    genomes.append(optimizer.ask())

                fitnesses = env.evaluate_population(genomes)
                optimizer.tell([(x,-f) for x,f in fitnesses]) # -1 as EA minimizes fitness func.

            # Extract winner
            winner = sorted(fitnesses, key=lambda x: x[1])[-1][0]

            # Store winner genome using pickle (for later use).
            winner_file = "es_best_run-{}_enemy-{}.pkl".format(run, enemy)
            with open(winner_file, "wb") as f:
                pickle.dump(winner, f)

    

    
