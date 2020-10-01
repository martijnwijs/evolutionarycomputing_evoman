import random

from deap import base
from deap import creator
from deap import tools

# create first individual ---------------------------------------------------------
creator.create("FitnessMin", base.Fitness, weights=(1.0,))  # make the weights positive to maximize fitness, negative to minimize fitness
creator.create("Individual", list, fitness=creator.FitnessMin)

n_var=10 # size of the individual (number of variables)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.random)      #  list of floats
toolbox.register("individual", tools.initRepeat, creator.Individual,   
                 toolbox.attr_float, n=n_var)

ind1 = toolbox.individual() #create a individual
print ind1          # a list containing weights
print ind1.fitness.valid     # no fitness values yet -> returns false

# ---------------------------------------------------------------------------------------


# the only function you have to write yourself
# returns the fitness value
def evaluate(individual): 
    return

# mutation
mutant = toolbox.clone(ind1) # clone individual
ind2, = tools.mutGaussian(mutant, mu=0.0, sigma=0.2, indpb=0.2) # mutate with gaussian distribution
del mutant.fitness.values


# crossover
child1, child2 = [toolbox.clone(ind) for ind in (ind1, ind2)]  # clone children beforehand
tools.cxBlend(child1, child2, 0.5)  # crossover
del child1.fitness.values
del child2.fitness.values

# selection
selected = toolbox.select(population, LAMBDA)
offspring = [toolbox.clone(ind) for ind in selected]

