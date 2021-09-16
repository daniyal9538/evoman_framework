"""
    Filename:    specialist_neat_training.py
    Author(s):   Thomas Bellucci
    Assignment:  Task 1 - Evolutionary Computing
    Description: Contains the NEAT implementation (AE 1) for
                 the Evoman assignment (Task I).
"""
# Imports
import os, sys
sys.path.insert(0, 'evoman')
from environment import Environment
from controller import Controller
import pickle
import numpy
import neat
import tqdm
import glob


""" Player controller using NEAT
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


""" Evoman Environment
"""
class EvomanEnvironment:
    def __init__(self, enemy, run, outfile=None):
        self.enemy = enemy
        self.run = run
        self.outfile = outfile

        # Create outfile for stats when desired.
        if outfile is not None:
            if os.path.exists(outfile):
                os.remove(outfile)
                   
            with open(outfile, "a") as f:
                f.write("mean,max,enemy,run\n")
            
        
    def evaluate_individual(self, genome, config, show=False):
        """ Evaluates the phenotype-converted genome (genotype)
            by simulating a single round of Evoman.
        """
        # Build individual (controller/phenotype) from genome.
        controller = Individual(genome, config)

        # Set show = False to hide visuals and speed up learning.
        if not show:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            env = Environment(experiment_name=None,
                              player_controller=controller,
                              speed='fastest',
                              logs='off')

        # Set show = True for real time game on screen (for testing).
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
        """ Wrapper function to evaluate all individuals
            in the population one by one.
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

    # SETTINGS!
    RUNS = 4
    GENERATIONS = 10
    ENEMIES = [1]
    
    # Load configuration file.
    config = neat.Config(neat.DefaultGenome,
                         neat.DefaultReproduction,
                         neat.DefaultSpeciesSet,
                         neat.DefaultStagnation,
                         "neat.config")

    # Run EA for 3 enemies and 10 runs.
    for enemy in ENEMIES:
        for run in range(1, RUNS + 1):

            # Setup Evoman environment
            outfile = "neat_stats_run-{}_enemy-{}.csv".format(run, enemy)
            env = EvomanEnvironment(enemy, run, outfile)
            
            # Set up population and run EA for several generations.
            pop = neat.Population(config)
            winner = pop.run(env.evaluate_population, GENERATIONS)

            # Store winner genome using pickle (for later use).
            winner_file = "neat_best_run-{}_enemy-{}.pkl".format(run, enemy)
            with open(winner_file, "wb") as f:
                pickle.dump(winner, f)

    

    
