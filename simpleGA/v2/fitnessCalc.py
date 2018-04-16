from numpy import mean, math
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from copy import copy
import numpy as np
from simpleGA.v2.SingleTon import Singleton
from sklearn.model_selection import KFold

class FitnessCalc():
    # Here will be the instance stored.
    __instance = None
    _default_sol = 0.8
    _dataSet = None
    _estimator = None

    @staticmethod
    def getInstance():
        """ Static access method. """
        if FitnessCalc.__instance == None:
            FitnessCalc()
        return FitnessCalc.__instance

    def __init__(self):
        """ Virtually private constructor. """
        if FitnessCalc.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            FitnessCalc.__instance = self

    def setData(self, dataset):
        self._dataSet = dataset
    def getData(self):
        return copy(self._dataSet)

    def setEstimator(self, estimator):
        self._estimator = estimator

    def getEstimator(self):
        return copy(self._estimator)

    def setSol(self, num):
        self._default_sol = num

    def getSol(self):
        return copy(self._default_sol)

    def getModel(self, individual):
        parameter = individual.getParam()
        model = self._estimator.set_params(**parameter)
        model.fit(self._dataSet.getXTrain(), self._dataSet.getYTrain())
        return copy(model)
    def getFitness(self, model):
        scores = []
        for i in range(0, 1):
            scores.append(model.score(self._dataSet.getXTest(), self._dataSet.getYTest()))
        return copy(math.pow(100,mean(scores)))

    def getFitnessFold(self, individual):
        kf = KFold(n_splits=10)
        X = np.array(self._dataSet.getX())
        y = np.array(self._dataSet.getY())
        parameter = individual.getParam()
        model = self._estimator.set_params(**parameter)
        scores = []
        for train_index, test_index in kf.split(X):
            # print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model.fit(X_train, y_train)
            scores.append(model.score(X_test, y_test))
        return copy(math.pow(100,mean(scores)))

