import pickle

from specialist_es_training import Individual, EvomanEnvironment


if __name__ == "__main__":
    ENEMY = 4
    RUN = 1
    SHAPE = (20, 32, 5)

    # Load (best) genome.
    filename = "es_best_run-{}_enemy-{}.pkl".format(RUN, ENEMY)
    with open(filename, "rb") as f:
        genome = pickle.load(f)

    # Run game
    env = EvomanEnvironment(ENEMY, RUN)
    while True:
        fitness, gain = env.evaluate_individual(genome, show=True)
        print(fitness, gain)

    

    
