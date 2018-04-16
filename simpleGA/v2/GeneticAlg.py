from __future__ import division
from copy import copy

from simpleGA.v2.fitnessCalc import FitnessCalc
from simpleGA.v2.population import Population
from simpleGA.v2.individual import Individual
import random
import numpy as np


class _ConstGA(object):
    @staticmethod
    def uniformRate():
        return 0.7

    @staticmethod
    def mutationRate():
        return 0.01

    @staticmethod
    def tournamentSize():
        return 3

    @staticmethod
    def elitism():
        return True


class GeneticAlg:

    @staticmethod
    def evolvePopulation_v1(pop):
        newPopulation = Population(pop.getSize())
        # keep best individual
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
    def evolvePopulation_v2(pop):
        newPopulation = Population(pop.getSize())
        #selection
        selectedPopulation = GeneticAlg.__ER_Selection(pop, 0.99)

        #cross section
        selectedPopulation.updateSize()
        for index in range(0, newPopulation.getSize()):
            randomIts = np.random.randint(selectedPopulation.getSize(), size = 2)
            indiv1 = selectedPopulation.getIndividual(randomIts[0])
            indiv2 = selectedPopulation.getIndividual(randomIts[1])
            newIndiv = GeneticAlg.__crossover(indiv1, indiv2)
            newPopulation.saveIndividual(newIndiv)

        # mutation population
        for index in range(0, newPopulation.getSize()):
            newPopulation.changeIndividual(GeneticAlg.__mutate(newPopulation.getIndividual(index)), index)

        return newPopulation

    @staticmethod
    def __mutate(indiv):
        newIndiv = Individual()
        if (random.uniform(0, 1) <= _ConstGA.mutationRate()):
            indiv.mutMax_depth(FitnessCalc.getInstance().getData().getNumFeature())

        if (random.uniform(0, 1) <= _ConstGA.mutationRate()):
            indiv.mutMin_samples_split(FitnessCalc.getInstance().getData().getNumIntances())

        if (random.uniform(0, 1) <= _ConstGA.mutationRate()):
            indiv.mutMin_impurity_decrease()

        if (random.uniform(0, 1) <= _ConstGA.mutationRate()):
            indiv.mutCriterion()
        if (random.uniform(0, 1) <= _ConstGA.mutationRate()):
            indiv.mutN_estimators(FitnessCalc.getInstance().getData().getNumFeature())

        newIndiv.setParam(indiv.getParam())
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

        if (random.uniform(0, 1) <= _ConstGA.uniformRate()):
            oldparam["n_estimators"] = indiv2.getParam()["n_estimators"]

        newIndiv.setParam(oldparam)
        return newIndiv

    @staticmethod
    def __tournamentSelection(pop):
        tournament = Population(pop.getSize())
        for index in range(0, _ConstGA.tournamentSize()):
            randomIt = np.random.randint(1, pop.getSize() - 1)
            tournament.saveIndividual(pop.getIndividual(randomIt))
        fittest = tournament.getFittest()

        return fittest

    @staticmethod
    def __SUS_Selection(pop):
        n =  pop.getSize()
        listFitness = pop.getListFitness()
        sumFitness = sum(listFitness)
        mean = sumFitness / n
        alpha = random.uniform(0, 1)
        delta = alpha * mean
        j = 0
        fittest = Population(pop.getSize())
        count = 0
        Sum = pop.getIndividual(0).getFitness()
        while True:
            if delta < Sum:
                fittest.saveIndividual(pop.getIndividual(j))
                delta = delta + Sum
            else:
                j = j + 1
                if (j >= pop.getSize()):
                    break
                Sum = Sum + pop.getIndividual(j).getFitness()

        return fittest
    def __ER_Selection(pop, c):
        sort = pop.sortPopulation(False)
        fittest = Population(pop.getSize())
        s = []
        s.append(0)
        for index in range(1, pop.getSize()):
            s.append(s[index - 1] + ((c - 1)/(pow(c, index) - 1))*pow(c, index-1))

        for i in range(0, pop.getSize()):
            r = np.random.uniform(0, s[len(s)-1])
            for j in range (1, pop.getSize()):
                if (s[j-1] <= r ) and (r < s[j]):
                    fittest.saveIndividual(sort[j])
                    break
        fittest.updateSize()
        return fittest
