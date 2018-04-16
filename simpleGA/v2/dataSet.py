from copy import copy
from sklearn.model_selection import KFold

class DataSet:
    _X_train = None
    _y_train = None
    _X_test = None
    _y_test = None
    _X = None
    _y = None

    def __init__(self, X_train, y_train, X_test, y_test, X, y):
        self._X_train = X_train
        self._X_test = X_test
        self._y_train = y_train
        self._y_test = y_test
        self._X = X
        self._y = y

    def getNumIntances(self):
        return len(self._X_train)

    def getNumFeature(self):
        return len(self._X_train[0])

    def getXTrain(self):
        return self._X_train

    def getYTrain(self):
        return self._y_train

    def getXTest(self):
        return self._X_test

    def getYTest(self):
        return self._y_test

    def getX(self):
        return self._X

    def getY(self):
        return self._y
