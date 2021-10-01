This folder contains all code used to generate the results of Assignment 1 (Task I: specialist agent) for the 
Evolutionary Computing course at the VU (2021).

For this assignment we implemented NEuroEvolution of Augmenting Topologies (NEAT) and a slight variant of this
algorithm with bypasses the augmentation of the topology and only optimizes a fixed topology's weights. This 
README will hopefully provide some insight regarding our implementation of the algorithms and will explain how 
to run our files.

# Overview of files
The following files and directories are included in this repository:

## Directories:
* \evoman: 				Implementation of the Evoman environment (not modified in any way).
* \neat_best_not_fixed_topology: 	All final solutions found by the standard NEAT variant (ind-1).
* \neat_best_fixed_topology: 		All final solutions found by the fixed-topology NEAT variant (ind-3).
* \neat_stats_not_fixed_topology: 	CSV files containing the mean and max statistics recorded during each 
 					run of the standard NEAT variant (ind-1).
* \neat_stats_fixed_topology: 		CSV files containing the mean and max statistics recorded during each 
 					run of the fixed-topology NEAT variant (ind-3).
* \solutions: 				Folder containing all final solutions in .pkl format (includes results 
					of both algorithms). 
* \stats: 				Folder containing all csv files with mean and max statistics recorded 
 					during each run (includes results of both algorithms). .

Note: \solutions and \stats are technically duplicates of the remaining folders (excluding \evoman). This was done 
to simplify the plotting of the solutions and ensure no files were accidentally discarded or overwritten.

## Configuration files:
* neat.config				Configuration file used for the standard topology-optimizing NEAT.
* neat_fixed_topology.config		Configuration file used for the fixed-topology variant of NEAT.

## Code files:
* specialist_boxplot.py			Runs the solutions in the \solutions folder 5 times and creates a box-
					plot for each algorithm and enemy showing the individual gain scores.		
* specialist_fitness_plots.py		Plots the statistics (mean and stdev of max/mean fitness during a run) 
					stored in the csv files in the \stats folder.
* specialist_neat_test.py		Runs a player controller in the environment (used for debugging purposes 
					and to visually verify the found solutions).
* specialist_neat_training.py		Runs one of the evolutionary algorithm (NEAT or Fixed NEAT) for x amount 
					of runs and y generations with user specified enemies and algorithm.
* specialist_neat_tune.py		Runs NEAT with different types of configuration files (different parameters)
					to find appropriate hyperparameters that work for the chosen enemies.


# Usage

## Running the EAs
To run the standard NEAT algorithm with the settings used by us in the report, use the following command:

Windows:

$ py -3 specialist_neat_training.py --runs=10 --generations=30 --enemies=2,3,7 --individual_type=1

Ubuntu:

$ python3 specialist_neat_training.py --runs=10 --generations=30 --enemies=2,3,7 --individual_type=1

This runs the NEAT algorithm for 30 generations, 10 times on enemies 2, 3 and 7. To select the second NEAT 
variant which fixes the topology of all individuals in the population, use --individual_type=3 instead.


## Visualizing the EAs
To visualize the progress plots and boxplots, replace the solutions in \solutions and statistics files in \stats 
with the newly generated files. Then simply run the following command:

Windows:

$ py -3 specialist_boxplot.py
$ py -3 specialist_fitness_plots.py

Ubuntu:

$ python3 specialist_boxplot.py
$ python3 specialist_fitness_plots.py


## Visualizing the solutions
To run a found solution in the environment, simply execute the following command (specifying the desired algorithm 
(IND), enemy and run in the file itself):

Windows:

$ py -3 specialist_neat_test.py

Ubuntu:

$ python3 specialist_neat_test.py

