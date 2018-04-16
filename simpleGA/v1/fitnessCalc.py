from numpy import mean
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


class _ConstFC(object):
    default_sol = 0.8
    X_train = None
    y_train = None
    X_test = None
    y_test = None


class FitnessCalc:
    @staticmethod
    def setSol(num):
        _ConstFC.default_sol = num

    @staticmethod
    def getSol():
        return _ConstFC.default_sol

    @staticmethod
    def setTrainTest(X_train, y_train, X_test, y_test):
        _ConstFC.y_train = y_train
        _ConstFC.y_test = y_test
        _ConstFC.X_test = X_test
        _ConstFC.X_train = X_train
        # print(type(_ConstFC.y_train[0]), " y_train ", _ConstFC.y_train)
        # print(type(_ConstFC.y_test[0]), " y_test ", _ConstFC.y_test)
        # print(type(_ConstFC.X_test[0]), " X_test ", _ConstFC.X_test)
        # print(type(_ConstFC.X_train[0]), " X_train ", _ConstFC.X_train)

    @staticmethod
    def getFitness(individual):
        parameter = individual.getParam()
        model = DecisionTreeClassifier(**parameter)
        model.fit(_ConstFC.X_train, _ConstFC.y_train)
        scores = []
        for i in range(0, 100):
            scores.append(model.score(_ConstFC.X_test, _ConstFC.y_test))
        return mean(scores)
