from __future__ import division
from copy import copy

import numpy as np
import random
from simpleGA.v2.fitnessCalc import FitnessCalc


class Individual:
    _fitness = None
    _DNA_param_dist = None
    _model = None
    def __init__(self):
        self._fitness = 0;
        self._DNA_param_dist = {"max_depth":  '',
                      "min_samples_split": '',
                      "min_impurity_decrease": '',
                      "criterion": '',
                      "n_estimators": '',
                    "max_features": 'auto'}

    def randomDNA(self, num_features, num_instances):
        self.mutMax_depth(num_features)
        self.mutMin_samples_split(num_instances)
        self.mutMin_impurity_decrease()
        self.mutCriterion()
        self.mutN_estimators(num_features)

    def mutMax_depth(self, num_features):
        self._DNA_param_dist['max_depth'] = np.random.randint(8, num_features*20)
        # if (n == 0):
        #     self._DNA_param_dist['max_depth'] = None
        # else :
        #      = n
    def mutMin_samples_split(self, num_instances):
        self._DNA_param_dist['min_samples_split'] = np.random.randint(2, num_instances)
    def mutN_estimators(self, num_features):
        self._DNA_param_dist['n_estimators'] = np.random.randint(10, num_features*40)
    def mutMin_impurity_decrease(self):
        self._DNA_param_dist['min_impurity_decrease'] = np.random.randint(0, 5 )/10
    def mutCriterion(self):
        self._DNA_param_dist['criterion'] =  np.random.choice (["gini", "entropy"])

    def getParam(self):
        return copy(self._DNA_param_dist)

    def setParam(self, param):
        self._DNA_param_dist = param

    def getModel(self):
        return self._model

    def getFitness(self):
        if (self._fitness == 0):
            self._model = FitnessCalc.getInstance().getModel(self)
            self._fitness = FitnessCalc.getInstance().getFitness(self._model)
            #self._fitness = FitnessCalc.getInstance().getFitnessFold(self)
        return copy(self._fitness)