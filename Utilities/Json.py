import json

import os
from _csv import reader
from re import finditer

import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from settings import APP_STATIC
from Utilities.utilities import Utilities
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
# vectorizer = TfidfVectorizer()

from SugConfig.sugconfig import SugCofig

corpus, uids = Utilities.Json_Parse(True)
print(np.unique(uids))
# print(uids)
# print(type)
# print(len(type))
# print(corpus)
# print(uid)
# X = vectorizer.fit_transform(corpus)
# print(X[2])
# input = vectorizer.transform(['Avg number of query']).toarray()
#
# analyze = vectorizer.build_analyzer()
# print(vectorizer.get_feature_names())
# print(vectorizer.vocabulary_)
# km = KMeans(n_clusters=np.unique(uids).shape[0])
# km.fit(X, uids)
#
# print("Homogeneity: %0.3f" % metrics.homogeneity_score(uids, km.labels_))
# print("Completeness: %0.3f" % metrics.completeness_score(uids, km.labels_))
# print("V-measure: %0.3f" % metrics.v_measure_score(uids, km.labels_))
# print("Adjusted Rand-Index: %.3f"
#       % metrics.adjusted_rand_score(uids, km.labels_))
# print("Silhouette Coefficient: %0.3f"
#       % metrics.silhouette_score(X, km.labels_, sample_size=1000))
#
# print()
#
# result = km.predict(input)
# print(result)

# X_train, X_test, y_train, y_test = train_test_split(
#     corpus, uids, test_size=0.2, random_state=None, stratify=uids, shuffle=True)
# pipeline = Pipeline([('vectorizer', TfidfVectorizer()), ('classifier', LinearSVC())])
#



# pipeline.fit(X_train, y_train)
# score = pipeline.score(X_test, y_test)
# result = pipeline.predict(['Avg number of query'])
# # print(X.toarray())
# K = X.toarray()
# # print(vectorizer.transform(['Avg number of query']).toarray())
# transformer = TfidfTransformer()
# tfidf = transformer.fit_transform(X.toarray())
# # print(tfidf.toarray())
# tmp = tfidf.toarray()
#
#
# clf = svm.SVC(decision_function_shape='ovo')

SugCofig.getInstance().validation()
SugCofig.getInstance().train()
print(SugCofig.getInstance().predict(SugCofig.getInstance().getCorpus()))
