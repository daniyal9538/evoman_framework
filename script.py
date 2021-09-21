import numpy as np
import matplotlib.pyplot as plt


for j in [2, 3, 7]:
    runs_gens_mean = []
    runs_gens_max = []

    for i in range(1, 11):
        data = np.genfromtxt("neat_stats_run-"+str(i)+"_enemy-"+str(j)+"_ind-1.csv", delimiter=",", names=["mean", "max"], skip_header=1)
        runs_gens_mean.append(data['mean'])
        runs_gens_max.append(data['max'])

    mean_mean_array = np.mean(runs_gens_mean, axis=0)
    mean_max_array = np.mean(runs_gens_max, axis=0)

    std_array_mean = np.std(runs_gens_mean, axis=0)
    std_array_max = np.std(runs_gens_max, axis=0)


    plt.plot(mean_mean_array)
    plt.fill_between(range(30), mean_mean_array-std_array_mean, mean_mean_array+std_array_mean, alpha = 0.5)
    plt.title("Mean of fitness means over runs, enemy "+str(j))
    plt.xlabel("No. of generations")
    plt.ylabel("Fitness")
    plt.show()

    plt.plot(mean_max_array)
    plt.fill_between(range(30), mean_max_array-std_array_max, mean_max_array+std_array_max, alpha = 0.5)
    plt.title("Mean of fitness max over runs, enemy "+str(j))
    plt.xlabel("No. of generations")
    plt.ylabel("Fitness")
    plt.show()