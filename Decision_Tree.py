import numpy as np
from sklearn import tree

class Decision_Tree:
    """A simple example class"""
    i = 12345
    __name = "helo"
    __model_tree = None
    def __init__(self):
        self.__model_tree = tree.DecisionTreeClassifier(criterion='entropy')

    def training(self, dataset):
        X = list()
        y = list()
        for row in dataset:
            x = list()
            for index in range(len(row) - 1):
                x.append(row[index])
            X.append(x)
            y.append(row[len(row) - 1])

        # for classification, here you can change the algorithm as gini or entropy (information gain) by default it is gini
        # model = tree.DecisionTreeRegressor() for regression
        # Train the model using the training sets and check score
        self.__model_tree.fit(X, y)
        self.__model_tree.score(X, y)

    def predict(self, input):
        x = np.fromstring(input, dtype=np.float, sep=',')
        x = [x]
        return self.__model_tree.predict(x)
