from copy import copy

from simpleGA.v2.individual import Individual
from simpleGA.v2.fitnessCalc import FitnessCalc

class Population:
    _individuals = None
    _Size = None
    def __init__(self, populationSize):
        self._Size = populationSize
        self._individuals = []

    def initPopulation(self):
        for i in range(0, self._Size, 1):
            indiv = Individual()
            indiv.randomDNA(FitnessCalc.getInstance().getData().getNumFeature(), FitnessCalc.getInstance().getData().getNumIntances())
            self.saveIndividual(indiv)
    def updateSize(self):
        self._Size = len(self._individuals)
    def getSize(self):
        return self._Size

    def saveIndividual(self, individual):
        self._individuals.append(individual)

    def getIndividual(self, index):
        return self._individuals[index]

    def changeIndividual(self, newindividual, index):
        self._individuals[index] = newindividual

    def getFittest(self):
        fittest = self._individuals[0]
        for individual in self._individuals:
            if fittest.getFitness() <= individual.getFitness():
                fittest = individual
        return fittest

    def getListFitness(self):
        list = []
        for individual in self._individuals:
            list.append(individual.getFitness())
        return copy(list)

    def sortPopulation(self, flag):
        return sorted(self._individuals, key=lambda x: x.getFitness(), reverse=flag)