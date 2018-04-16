import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn import metrics

from Utilities.utilities import Utilities


class SugCofig():
    __instance = None
    _pipeline = None
    _corpus = None
    _uids = None
    @staticmethod
    def getInstance():
        """ Static access method. """
        if SugCofig.__instance == None:
            SugCofig()
        return SugCofig.__instance

    def __init__(self):
        """ Virtually private constructor. """
        if SugCofig.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            SugCofig.__instance = self
            self._corpus, self._uids = Utilities.Json_Parse()

    def validation(self):
        self._pipeline = Pipeline([('vectorizer', TfidfVectorizer()), ('classifier', LinearSVC())])
        kf = KFold(n_splits=10)
        scores = []
        X = np.array(self._corpus)
        y = np.array(self._uids)
        for train_index, test_index in kf.split(X):
            # print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            self._pipeline.fit(X_train, y_train)
            scores.append(self._pipeline.score(X_test, y_test))
        print("Finial scores: ", np.mean(scores))

    def train(self):
        self._pipeline = Pipeline([('vectorizer', TfidfVectorizer()), ('classifier', LinearSVC())])
        X = np.array(self._corpus)
        y = np.array(self._uids)
        predict = self._pipeline.fit(X, y).predict(X)

        print("Homogeneity: %0.3f" % metrics.homogeneity_score(self._uids, predict))
        print("Completeness: %0.3f" % metrics.completeness_score(self._uids, predict))
        print("V-measure: %0.3f" % metrics.v_measure_score(self._uids, predict))
        print("Adjusted Rand-Index: %.3f"
              % metrics.adjusted_rand_score(self._uids, predict))

    def predict(self, input):
        return self._pipeline.predict(input)

    def getCorpus(self):
        return self._corpus

    def getUid(self):
        return self._uids




