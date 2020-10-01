# imports framework
import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller

# imports other libs
import time
import numpy as np
from math import fabs,sqrt
import glob, os
import random


experiment_name = 'individual_demo'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10

# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                  enemies=[6],
                  playermode="ai",
                  #multiplemode="yes",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest")

# default environment fitness is assumed for experiment

env.state_to_log() # checks environment state

####   Optimization for controller solution (best genotype-weights for phenotype-network): Ganetic Algorihm    ###

ini = time.time()  # sets time marker

#variables
n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

# runs simulation
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f

############################### DEAP
import random

from deap import base
from deap import creator
from deap import tools
import numpy as np

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


toolbox = base.Toolbox()
# Attribute generator , float
toolbox.register("attr_float", random.gauss, 0, 1)
# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual, 
    toolbox.attr_float, n_vars)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluateoffspring(individual):
    individual = np.array(individual)
    #individual = np.array(individual[:-1]) # remove last element as that is the mutationrate
    #print(individual)
    fitness = simulation(env, individual)
    fitness = float(fitness)
    #print("fitness", fitness)
    #if fitness < 0:
        #fitness = 0.1
    #print(fitness)
    return (fitness,)

toolbox.register("evaluate", evaluateoffspring)
toolbox.register("mate", tools.cxUniform, indpb=0.8) # you can take a independent variable or place it further in the code
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2) # later on, coevolve this value
toolbox.register("select", tools.selTournament, k=75, tournsize=15)  # k number of individuals too select change these values for premature convergence
toolbox.register("Elite", tools.selBest, k=3)  # k is number of elite
toolbox.register("Worst", tools.selWorst, k=3)
def main():
    pop = toolbox.population(n=75) # change population size here

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # CXPB  is the probability with which two individuals
    #       are crossed
    #
    # MUTPB is the probability for mutating an individual
    CXPB, MUTPB = 0.5, 0.2

    # Extracting all the fitnesses of 
    fits = [ind.fitness.values[0] for ind in pop]

    # Variable keeping track of the number of generations
    g = 0
    
    # Begin the evolution
    while max(fits) < 100 and g < 1000:
        # A new generation
        g = g + 1
        print("-- Generation %i --" % g)

        # Select the parents
        offspring = toolbox.select(pop)
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))


        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            #if random.random() < CXPB:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values
        
        # mutation
        for mutant in offspring:
            toolbox.mutate(mutant)
            del mutant.fitness.values
            
        # select elite
        elite = toolbox.Elite(pop)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit


        # replace the old population by the offspring
        pop[:] = offspring
        # insert the elite in the first indices
        pop[0:len(elite)] = elite

        print("popsize", len(pop))
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
        


        #statistics
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        
        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)

if __name__ == '__main__':
    main()