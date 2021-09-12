"""
    Filename:   test_specialist_agent_training.py
    Author(s):  Thomas Bellucci
    Assignment: Task 1 - Evolutionary Computing
"""
import os, sys
sys.path.insert(0, 'evoman') 
from environment import Environment
from controller import Controller
import numpy as np
import pickle
import neat



class Individual(Controller):
    def __init__(self, genome, config):
        self.net = neat.nn.FeedForwardNetwork.create(genome, config)

    def control(self, state, controller=None):
        out = self.net.activate(state)
        action = [0, 0, 0, 0, 0]
        action[np.argmax(out)] = 1
        return action



def evaluate_individual(genome, config, enemy=1, headless=True):
    """ Evaluates a genome by simulating a single game with the
        current individual (represented by the genome).
    """
    # Set up individual (controller) from genome.
    controller = Individual(genome, config)
    
    # Set headless = True to speed up learning.
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    # Set directory
    experiment_name = 'dummy_demo'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    # Init environment (set to realtime if blitted)
    if headless:
        env = Environment(experiment_name=experiment_name,
                          player_controller=controller,
                          speed='fastest',
                          logs='off')
    else:
        env = Environment(experiment_name=experiment_name,
                          player_controller=controller,
                          speed='normal',
                          logs='off')

    # Select Enemy (1 to 8)
    env.update_parameter('enemies', [enemy])

    # Simulate game.
    fitness, player_life, enemy_life, game_duration = env.play("unused")
    return fitness


def evaluate_population(population, config):
    """ Wrapper function to evaluate all individuals in the population one by one. """
    # Run individual evaluation function on individs in population (and store fitnesses).
    fitnesses = []
    for ind_id, ind in population:
        ind.fitness = evaluate_individual(ind, config)
        fitnesses += [ind.fitness]

    # Print some stats
    print("Fitness: Avg:", np.mean(fitnesses), "\tStd: fitness =", np.std(fitnesses))




def run(config_file):
    # Load configuration file.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Init population and optimize its individuals.
    p = neat.Population(config)
    winner = p.run(evaluate_population, 20)

    # Store winner as Individual
    with open("best_neat_solution", "wb") as f:
        pickle.dump(winner, f)

if __name__ == '__main__':
    run("neat.config")

    

    
