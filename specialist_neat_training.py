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
        # Feed state vector through network.
        out = self.net.activate(state)
        
        # Allow multiple actions at once
        return numpy.array(out) > .5


def evaluate_individual(genome, config, enemy, show=False):
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
    env.update_parameter('enemies', [enemy])

    # Simulate game.
    fitness, player_life, enemy_life, game_duration = env.play()
    indv_gain = player_life - enemy_life
    return fitness, indv_gain


def evaluate_population(population, config):
    """ Wrapper function to evaluate all individuals
        in the population one by one.
    """
    global current_enemy, current_run, current_outfile
    
    # Run fitness function on all individuals in population (and store fitnesses).
    fitness_scores = []
    for _, ind in tqdm.tqdm(population):
        ind.fitness, _ = evaluate_individual(ind, config, current_enemy)
        fitness_scores.append(ind.fitness)

    # Print progress
    avg_fitness = numpy.mean(fitness_scores)
    max_fitness = numpy.max(fitness_scores)
    print("Avg:", avg_fitness, max_fitness)

    # Write mean and max fitness stats of population to file.
    with open(current_outfile, "a") as f:
        f.write("{},{},{},{}\n".format(avg_fitness, max_fitness, current_enemy, current_run))



if __name__ == '__main__':

    # SETTINGS!
    RUNS = 2
    GENERATIONS = 10
    ENEMIES = [1]
    
    # Load configuration file.
    config = neat.Config(neat.DefaultGenome,
                         neat.DefaultReproduction,
                         neat.DefaultSpeciesSet,
                         neat.DefaultStagnation,
                         "neat.config")

    # Run EA for 3 enemies and 10 runs.
    for current_enemy in ENEMIES:
        for current_run in range(1, RUNS + 1):

            # Write header of output file (Wipe when it already exists).
            current_outfile = "neat_stats_run-{}_enemy-{}.csv".format(current_run, current_enemy)
            if os.path.exists(current_outfile):
               os.remove(current_outfile)
               
            with open(current_outfile, "a") as f:
                f.write("mean,max,enemy,run\n")
            
            # Set up population and run EA for .
            pop = neat.Population(config)
            winner = pop.run(evaluate_population, GENERATIONS)

            # Store winner genome using pickle.
            current_winner_file = "neat_best_run-{}_enemy-{}.pkl".format(current_run, current_enemy)
            with open(current_winner_file, "wb") as f:
                pickle.dump(winner, f)

    

    
