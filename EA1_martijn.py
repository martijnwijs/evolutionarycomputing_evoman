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
pop_size = 80
mutationrate = .15
parent_selectionrate = .3
n_parents = int(parent_selectionrate*pop_size)
elitism = True



# runs simulation
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f

# evaluation
def evaluate(x):
    return np.array(list(map(lambda y: simulation(env,y), x)))

# create initial population
def create(n_vars, pop_size):
    # create population with random weights between 0 and 1
    population = np.random.randn(pop_size, n_vars)
    #print("initialpop", population)
    return population

# mutation
def mutate(population):
    for individual in range(1, pop_size): # never mutate the first because the first is reserved for elite
        for gene in range(n_vars):
            #print("gene:", gene)
            random_value = random.uniform(0,1)
            if random_value < mutationrate:
                
                #mutationvalue -> change this value to get other results
                population[individual][gene] = population[individual][gene]+np.random.normal(0, 1)
    return population

#   normalizing  -> change to numpy mutations
def normalize(evaluation):
    normalized = evaluation # create a new array for normalizing
    max = np.amax(evaluation)
    min = np.amin(evaluation)
    for i in range(pop_size):
        if max - min > 0:
            normalized[i] = (evaluation[i]-min)/(max-min)
        else:
            normalized[i] = 0
        if normalized[i] < 0:
            normalized[i] = 0
    return normalized



# 1 point crossover
def crossover(population, parents):
    for individual in range(pop_size):
        crossover_point = random.uniform(0, n_vars)
        for gene in range(n_vars):
            if gene < crossover_point:
                population[individual][gene] = parents[0][gene]
            else:
                population[individual][gene] = parents[1][gene]
    return population

# uniform crossover (seems to work better)
def uniform_crossover(population, parents, fittest_individual, elitism):
    for individual in range(pop_size):
        #randomly take 2 different parents

        first_parent = int(random.uniform(0, n_parents))
        second_parent = first_parent
        while second_parent == first_parent:
            second_parent = int(random.uniform(0, n_parents))

        for gene in range(n_vars):
            #randomly take a gene from one of the parents
            if random.choice([True, False]) == True:
                population[individual][gene] = parents[first_parent][gene]
            else:
                population[individual][gene] = parents[second_parent][gene]
    # if elitism, take the fittest individual and replace one offspring
    if elitism == True:
       # print("population0", population[0])
        population[0] = fittest_individual
        print("after should not be same")
        #print(population[0])
        #print("populationwithfittest", population)
    return population       

# returns the best individual
def elitism(population, evaluation):
    print("fittest: ")
    print(np.argmax(evaluation))
    fittest_individual = population[np.argmax(evaluation)]
    print("fittest individual ", fittest_individual)
    return fittest_individual

# returns an array of the selected parents
# proportionate selection method
def parent_selection(normalized_evaluation, population):
    # make copies to delete one parent after chosen
    population_copy = population
    normalized_evaluation_copy = normalized_evaluation
    parents = np.zeros((n_parents, n_vars)) 
    
    for i in range(n_parents): # select parents with selection rate
       # print("parent!!")
        fitness_total = np.sum(normalized_evaluation_copy) # sum all the fitnesses
        random_value = random.uniform(0, fitness_total)
        fitness_count = 0
        for j in range(normalized_evaluation_copy.shape[0]):
            fitness_count += normalized_evaluation_copy[j]  # count fitness
            
            if fitness_count > random_value: # select this parent
                parents[i] = population_copy[j]
                #print("parents index:", j)

                # delete parent
                #print("populationcopy: ")
                print(population_copy)
                #print("normalizedevaluationcopy")
               # print(normalized_evaluation_copy)
                population_copy = np.delete(population_copy, j, axis=0)
                normalized_evaluation_copy = np.delete(normalized_evaluation_copy, j, axis=0)
                #print("populationcopy after delete:")
                #print(population_copy)
                break

    return parents

# init
population = create(n_vars, pop_size)
print("population")
print(population)
while True:
    print("jhoi")
    evaluation = evaluate(population) # evaluate all offsprings
    print("evaluation", evaluation)
    normalized_evaluation = normalize(evaluation)
    parents = parent_selection(normalized_evaluation, population) # find parents
    fittest_individual = elitism(population, normalized_evaluation)
    #fittest_individual = 0
    
    population = uniform_crossover(population, parents, fittest_individual, elitism=True)  # create new population
    #population = mutate(population)