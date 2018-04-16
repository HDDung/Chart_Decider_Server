import numpy as np
import random

from simpleGA.v1.fitnessCalc import FitnessCalc


class Individual:
    __fitness = 0
    __param_dist = None
    __num_features = None
    __num_instances = None
    def __init__(self, num_features = 5, num_instances = 299):
        rand = np.random.randint(2, size=10)
        self.__num_features = num_features
        self.__num_instances = num_instances
        self.__param_dist = {"max_depth":  np.random.randint(1, num_features),
                      "min_samples_split": np.random.randint(2, round(num_instances/20)),
                      "min_impurity_decrease": np.random.randint(0, 10 )/10,
                      "criterion": random.choice (["gini", "entropy"])}

    def mutMax_depth(self):
        return np.random.randint(1, self.__num_features)
    def mutMin_samples_split(self):
        return np.random.randint(2, round(self.__num_instances/30))
    def mutMin_impurity_decrease(self):
        return np.random.randint(0, 10 )/10
    def mutCriterion(self):
        return random.choice (["gini", "entropy"])

    def getParam(self):
        return self.__param_dist

    def setParam(self, param):
        self.__param_dist = param

    def getFitness(self):
        if self.__fitness == 0:
            self.__fitness = FitnessCalc.getFitness(self)
        return self.__fitness