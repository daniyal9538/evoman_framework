"""
    Filename:    specialist_neat_training.py
    Author(s):   Thomas Bellucci
    Assignment:  Task 1 - Evolutionary Computing
    Description: Contains the NEAT implementation (AE 1) for
                 the Evoman assignment (Task I).
"""
# Imports
from ast import parse
from msilib.schema import Control
import os, sys
sys.path.insert(0, 'evoman')
from environment import Environment
from controller import Controller
import pickle
import numpy
import neat
import tqdm
import argparse
import glob


""" Implements the player controller using NEAT.
"""
class Individual(Controller):
    def __init__(self, genome, config):
        # Build network from NEAT genome (the genotype).
        self.net = neat.nn.FeedForwardNetwork.create(genome, config)

    def control(self, state, _):
        # Feed state vector (20 inputs) through network.
        out = self.net.activate(state)
        
        # Return binary vector of actions (allow multiple at once).
        return numpy.array(out) > .5


class Individual_RNN(Controller):
    def __init__(self, genome, config):
        self.net = neat.nn.RecurrentNetwork.create(genome, config)

    def control(self, state, _):
        out=self.net.activate(state)
        return numpy.array(out) > .5

""" Wrapper around the Evoman game environment. It contains the
    fitness function (evaluate_individual) and writes out per-
    generation stats for a run of the algorithm.
"""
class EvomanEnvironment:
    def __init__(self, enemy, run, outfile=None,Individual = Individual):
        self.enemy = enemy
        self.run = run
        self.outfile = outfile
        self.Individual = Individual

        # Create outfile for stats when an outfile is given.
        if outfile is not None: 
            with open(outfile, "w") as f: # Create new file with header
                f.write("mean,max,enemy,run\n")
            
        
    def evaluate_individual(self, genome, config, show=False):
        """ Evaluates the phenotype-converted genome (genotype) by simulating
            a single round of Evoman.
        """
        # Build individual (controller/phenotype) from genome.
        controller = self.Individual(genome, config)

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


    def evaluate_population(self, population, config):
        """ Wrapper function to evaluate all individuals in the population.
        """
        # Run fitness function on all individuals in population (and store fitnesses).
        fitness_scores = []
        for _, ind in tqdm.tqdm(population):
            ind.fitness, _ = self.evaluate_individual(ind, config)
            fitness_scores.append(ind.fitness)

        # Print stats of generation
        avg_fitness = numpy.mean(fitness_scores)
        max_fitness = numpy.max(fitness_scores)
        print("Stats:", avg_fitness, max_fitness)

        # Write mean and max fitness stats of population to file (when enabled).
        if self.outfile is not None:
            with open(self.outfile, "a") as f:
                f.write("{},{},{},{}\n".format(avg_fitness, max_fitness, self.enemy, self.run))



if __name__ == '__main__':
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', help='number of runs per enemy', default=10, type = int)
    parser.add_argument('--generations', help = 'number of generations EA will run', default=30,type=int)
    parser.add_argument('--enemies', help = 'comma seperated types of enemies', default='3, 7, 9')
    parser.add_argument('--individual_type', help='type of individual (nn) (1: ff nn, 2: rnn)', default=1,type=int)

    args = parser.parse_args()
 

    # SETTINGS!
    RUNS = args.runs
    GENERATIONS = args.generations
    ENEMIES = [int(i) for i in str(args.enemies).split(',')]
    INDIVIDUAL_TYPE = args.individual_type
    
    # Load configuration file.
    config = neat.Config(neat.DefaultGenome,
                         neat.DefaultReproduction,
                         neat.DefaultSpeciesSet,
                         neat.DefaultStagnation,
                         "neat.config")

    # Run EA for 3 enemies and 10 runs.
    for enemy in ENEMIES:
        for run in range(1, RUNS + 1):
            file_name = "neat_stats_run-{}_enemy-{}_ind-{}".format(run, enemy, str(INDIVIDUAL_TYPE))
            # Setup Evoman environment
            outfile = file_name+'.csv'
            if INDIVIDUAL_TYPE == 1:

                env = EvomanEnvironment(enemy, run, outfile, Individual=Individual)
                print("Training with FFN")
            elif INDIVIDUAL_TYPE == 2:
                env = EvomanEnvironment(enemy, run, outfile, Individual=Individual_RNN)
                print("Training with RNN")
            
            # Set up population and run EA for several generations.
            pop = neat.Population(config)
            winner = pop.run(env.evaluate_population, GENERATIONS)

            # Store winner genome using pickle (for later use).
            winner_file = "neat_best_run-{}_enemy-{}_ind-{}.pkl".format(run, enemy, str(INDIVIDUAL_TYPE))
            with open(winner_file, "wb") as f:
                pickle.dump(winner, f)

    

    
