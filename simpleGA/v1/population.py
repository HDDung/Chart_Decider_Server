from simpleGA.v1.individual import Individual

class Population:
    __individuals = None
    __Size = None
    def __init__(self, populationSize, initialise):
        self.__Size = populationSize
        self.__individuals = []
        if initialise:
            for i in range(0, self.__Size, 1):
                self.saveIndividual(Individual(5, 312))

    def getSize(self):
        return self.__Size

    def saveIndividual(self, individual):
        self.__individuals.append(individual)

    def getIndividual(self, index):
        return self.__individuals[index]

    def changeIndividual(self, newindividual, index):
        self.__individuals[index] = newindividual

    def getFittest(self):
        fittest = self.__individuals[0]
        for individual in self.__individuals:
            if fittest.getFitness() <= individual.getFitness():
                fittest = individual
        return fittest