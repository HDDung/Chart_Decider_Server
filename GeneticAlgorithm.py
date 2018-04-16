from __future__ import print_function
import matplotlib.pyplot as plt

import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from Utilities.utilities import Utilities
from simpleGA.v2.GeneticAlg import GeneticAlg
from simpleGA.v2.dataSet import DataSet
from simpleGA.v2.fitnessCalc import FitnessCalc
from simpleGA.v2.population import Population


class GeneticOptimization:
    _filename = 'Cooked Dataset - Copy.csv'
    _column_name = ['time', 'items', 'Numeric', 'negative values', 'String']
    _model = None

    def classify(self, features):
        result = "Training first"
        if self._model is not None:
            result = self._model.predict(features)
        return result

    def isHaveModel(self):
        if self._model is None:
            return True
        return False

    def training(self):
        # dataset = Utilities.load_csv(self._filename)
        # np.random.shuffle(dataset)
        # X, y = Utilities.extracXY(dataset)
        X, y = Utilities.load_csv_Json("Json.csv")
        OverallScore = []
        OverallScoreChart = {}
        for chart in np.unique(y):
            OverallScoreChart[chart] = 0
        It = 100
        for Dung in range(It):
            print(Dung)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=None, stratify=y, shuffle=True)

            # names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
            #          "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
            #          "Naive Bayes", "QDA"]
            #
            # classifiers = [
            #     KNeighborsClassifier(6),
            #     SVC(kernel="linear", C=0.025),
            #     SVC(gamma=4, C=1),
            #     GaussianProcessClassifier(1.0 * RBF(1.0)),
            #     DecisionTreeClassifier(max_depth=5),
            #     RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
            #     MLPClassifier(alpha=1),
            #     AdaBoostClassifier(),
            #     GaussianNB(),
            #     QuadraticDiscriminantAnalysis()]
            # for model in classifiers:
            #     print(model.get_params())
            # for name, clf in zip(names, classifiers):
            #     kf = KFold(n_splits=5)
            #     X = np.array(X)
            #     y = np.array(y)
            #     scores = []
            #     for train_index, test_index in kf.split(X):
            #         # print("TRAIN:", train_index, "TEST:", test_index)
            #         X_train, X_test = X[train_index], X[test_index]
            #         y_train, y_test = y[train_index], y[test_index]
            #         clf.fit(X_train, y_train)
            #         scores.append(clf.score(X_test, y_test))
            #     print(name," ", np.mean(scores))


            # a = np.array(X)
            # b = np.array(y)
            # print(a)

            # Test = dict()
            # k = np.unique(y)
            # for key in k:
            #     Test[key] = None
            # for index in range(len(X)):
            #     Test[y[index]]
            FitnessCalc.getInstance().setData(DataSet(X_train, y_train, X_test, y_test, X, y))
            FitnessCalc.getInstance().setSol(0.85)
            generation = 0

            myPop = Population(100)
            myPop.initPopulation()
            FitnessCalc.getInstance().setEstimator(RandomForestClassifier())
            bestFit = myPop.getFittest().getFitness()
            Flag = True
            score = []
            for index in range(0, 20):
                generation += 1
                score.append(np.math.log(myPop.getFittest().getFitness(), 100))
                # print("Generation: ", str(generation), " Fittest: ", str(np.math.log(myPop.getFittest().getFitness(), 100)))
                myPop = GeneticAlg.evolvePopulation_v1(myPop)
                if bestFit < myPop.getFittest().getFitness():
                    bestFit = myPop.getFittest().getFitness()
            # print("Best solution: ", str(np.math.log(myPop.getFittest().getFitness(), 100)))
            # print("Best solution: ", str(np.math.log(myPop.getFittest().getFitness(), 100)))
            # print("Best solution: ", str(np.math.log(myPop.getFittest().getFitness(), 100)))
            # print("Best solution: ", str(np.math.log(myPop.getFittest().getFitness(), 100)))

            OverallScore.append(np.math.log(myPop.getFittest().getFitness(), 100))
            plt.plot(range(len(score)), score, '--', linewidth=2)


            self._model = myPop.getFittest().getModel()

            scores = []
            for i in range(0, 100):
                scores.append(self._model.score(FitnessCalc.getInstance().getData().getXTest(), FitnessCalc.getInstance().getData().getYTest()))
            # print("Finial scores: ", np.mean(scores))
            # print("Parameter of estimator: ", myPop.getFittest().getParam())

            Tmp = self._model.predict(FitnessCalc.getInstance().getData().getXTest())
            charts = np.unique(y_test)

            for chart in charts:
                count = 0
                num_chart = 0
                t = {}
                for index in range(len(y_test)):
                    if y_test[index] == chart:
                        t[Tmp[index]] = 0
                for index in range(len(y_test)):
                    if y_test[index] == chart:
                        num_chart = num_chart + 1
                        t[Tmp[index]] = t[Tmp[index]] + 1
                        if y_test[index] == Tmp[index]:
                            count = count + 1
                # print(chart, " ", count/num_chart)
                OverallScoreChart[chart] = OverallScoreChart[chart] + count/num_chart
                # print(t)
            # bestIndi = myPop.getFittest()
            # kf = KFold(n_splits=10)
            # X = np.array(FitnessCalc.getInstance().getData().getX())
            # y = np.array(FitnessCalc.getInstance().getData().getY())
            # parameter = bestIndi.getParam()
            # model = RandomForestClassifier()
            # model = model.set_params(**parameter)
            # scores = []
            # for train_index, test_index in kf.split(X):
            #     # print("TRAIN:", train_index, "TEST:", test_index)
            #     X_train, X_test = X[train_index], X[test_index]
            #     y_train, y_test = y[train_index], y[test_index]
            #     model.fit(X_train, y_train)
            #     scores.append(model.score(X_test, y_test))
            # print("Finial scores: ", np.mean(scores))
        plt.show()
        print("Overall score: ", np.mean(OverallScore), " Overall score list: ", OverallScore)
        print("Sum score: ", np.sum(OverallScore))
        for element in OverallScoreChart:
            print(element, " ", OverallScoreChart[element]/It)
        print("Overall score of chart", OverallScoreChart)


