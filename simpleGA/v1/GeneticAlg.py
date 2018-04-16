from copy import copy

from simpleGA.v1.population import Population
from simpleGA.v1.individual import Individual
import random
class _ConstGA(object):
    @staticmethod
    def uniformRate():
        return 0.25
    @staticmethod
    def mutationRate():
        return 0.01

    @staticmethod
    def tournamentSize():
        return 3

    @staticmethod
    def elitism():
        return False

class GeneticAlg:

    @staticmethod
    def evolvePopulation(pop):
        newPopulation = Population(pop.getSize(), False)
        #keep best individual
        if _ConstGA.elitism():
            newPopulation.saveIndividual(pop.getFittest())

        # crossover population
        if _ConstGA.elitism():
            elitismOffset = 1
        else:
            elitismOffset = 0

        for index in range(elitismOffset, pop.getSize()):
            indiv1 = GeneticAlg.__tournamentSelection(pop)
            indiv2 = GeneticAlg.__tournamentSelection(pop)
            newIndiv = GeneticAlg.__crossover(indiv1, indiv2)
            newPopulation.saveIndividual(newIndiv)

        # mutation population
        for index in range(elitismOffset, newPopulation.getSize()):
            newPopulation.changeIndividual(GeneticAlg.__mutate(newPopulation.getIndividual(index)), index)

        return newPopulation


    @staticmethod
    def __mutate(indiv):
        newIndiv = Individual()
        oldparam = copy(indiv.getParam())
        if (random.uniform(0, 1) <= _ConstGA.mutationRate()):
            oldparam["max_depth"] = indiv.mutMax_depth()

        if (random.uniform(0, 1) <= _ConstGA.mutationRate()):
            oldparam["min_samples_split"] = indiv.mutMin_samples_split()

        if (random.uniform(0, 1) <= _ConstGA.mutationRate()):
            oldparam["min_impurity_decrease"] = indiv.mutMin_impurity_decrease()

        if (random.uniform(0, 1) <= _ConstGA.mutationRate()):
            oldparam["criterion"] = indiv.mutCriterion()

        newIndiv.setParam(oldparam)
        return newIndiv


    @staticmethod
    def __crossover(indiv1, indiv2):
        newIndiv = Individual()
        oldparam = copy(indiv1.getParam())
        if (random.uniform(0, 1) <= _ConstGA.uniformRate()):
            oldparam["max_depth"] = indiv2.getParam()["max_depth"]

        if (random.uniform(0, 1) <= _ConstGA.uniformRate()):
            oldparam["min_samples_split"] = indiv2.getParam()["min_samples_split"]

        if (random.uniform(0, 1) <= _ConstGA.uniformRate()):
            oldparam["min_impurity_decrease"] = indiv2.getParam()["min_impurity_decrease"]

        if (random.uniform(0, 1) <= _ConstGA.uniformRate()):
            oldparam["criterion"] = indiv2.getParam()["criterion"]

        newIndiv.setParam(oldparam)
        return newIndiv

    @staticmethod
    def __tournamentSelection(pop):
        tournament = Population(pop.getSize(), False)
        for index in range(0, _ConstGA.tournamentSize()):
            randomIt = random.randint(1, pop.getSize() - 1)
            tournament.saveIndividual(pop.getIndividual(randomIt))
        fittest = tournament.getFittest()

        return fittest


