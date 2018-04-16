import os
import json

import os
from re import finditer

import nltk
import numpy as np

from settings import APP_STATIC
from csv import reader


class Utilities:
    @staticmethod
    def load_csv_Json(filename_input):
        file = open(os.path.join(APP_STATIC, filename_input), 'r', encoding='utf-8-sig')
        lines = reader(file)
        features = list()
        charts = list()
        for line in lines:
            oJson = json.loads(line[2])
            items = int(line[0])
            chart = line[1]
            time, numeric, string, date = Utilities.json_parse(oJson)
            features.append([time, items, numeric, string, date])
            charts.append(chart)
        return features, charts

    @staticmethod
    def json_parse(oJson):
        TIME = {"TIMESTAMP", "DATE", "SECONDDATE"}
        STRING = {"NVARCHAR", "VARCHAR"}
        DATE = {"DAY", "HOUR", "MONTH", "WEEK", "MINUTE", "YEAR"}
        time = numeric = string = date = 0
        for element in oJson:
            if Utilities.kmpFirstMatch(DATE, element.upper()) is not None:
                date = date + 1
            elif oJson[element] in TIME:
                time = time + 1
            elif oJson[element] in STRING:
                string = string + 1
            else:
                numeric = numeric + 1
            # print(element, ": ", oJson[element])
        # print(oJson, " ", [time, items, numeric, string, date], " ", chart)
        return time, numeric, string, date

    @staticmethod
    def load_csv(filename_input):
        print("Reading file")
        file = open(os.path.join(APP_STATIC, filename_input), 'r', encoding='utf-8-sig')
        lines = reader(file)
        dataset_load = list(lines)
        print("Return reading file")
        return dataset_load
    @staticmethod
    def extracXY(data_input):
        X_out = list()
        y_out = list()
        for row in data_input:
            x = list()
            for index in range(len(row) - 1):
                x.append(row[index])
            X_out.append(x)
            y_out.append(row[len(row) - 1])
        return X_out, y_out
    @staticmethod
    def train_test(array_train, array_test, data_input):
        data_train = []
        for index in array_train:
            data_train.append(data_input[index])
        X_train_output, y_train_out = Utilities.extracXY(data_train)
        # print("X_train_output: ", X_train_output)
        # print("y_train_out: ", y_train_out)

        data_test = []
        for index in array_test:
            data_test.append(data_input[index])
        X_test_output, y_test_out = Utilities.extracXY(data_test)
        # print("X_test_output: ", X_test_output)
        # print("y_test_out", y_test_out)
        return X_train_output, y_train_out, X_test_output, y_test_out

    # Naive algorithm to find and return starting position of first match
    # takes O(p*t) time e.g. for pattern='a'*(p-1)+'b', text='a'*t
    @staticmethod
    def naiveMatch(pattern, text):
        for startPos in range(len(text) - len(pattern) + 1):
            matchLen = 0
            while pattern[matchLen] == text[startPos + matchLen]:
                matchLen += 1
                if matchLen == len(pattern):
                    return startPos

    # Find and return starting position of first match, or None if no match exists
    #
    # Time analysis:
    # each iteration of the inner or outer loops increases 2*startPos + matchLen
    # this quantity starts at 0 and ends at most at 2*t+p
    # so the total number of iterations of both loops is O(t+p)
    #
    @staticmethod
    def kmpFirstMatch(patterns, text):
        for pattern in patterns:
            shift = Utilities.computeShifts(pattern)
            startPos = 0
            matchLen = 0
            for c in text:
                while matchLen >= 0 and pattern[matchLen] != c:
                    startPos += shift[matchLen]
                    matchLen -= shift[matchLen]
                matchLen += 1
                if matchLen == len(pattern):
                    return startPos

    # Slightly more complicated version to return sequence of all matches
    # using Python 2.2 generators (yield keyword in place of return).
    # Same time analysis as kmpFirstMatch.
    #
    @staticmethod
    def kmpAllMatches(pattern, text):
        shift = Utilities.computeShifts(pattern)
        startPos = 0
        matchLen = 0
        for c in text:
            while matchLen >= 0 and pattern[matchLen] != c:
                startPos += shift[matchLen]
                matchLen -= shift[matchLen]
            matchLen += 1
            if matchLen == len(pattern):
                yield startPos
                startPos += shift[matchLen]
                matchLen -= shift[matchLen]

    # Construct shift table used in KMP matching
    #
    # Time analysis: each iteration of either loop increases shift+pos
    # This quantity starts at 0 and ends at most at 2*p
    # So total time is O(p).
    #
    @staticmethod
    def computeShifts(pattern):
        shifts = [None] * (len(pattern) + 1)
        shift = 1
        for pos in range(len(pattern) + 1):
            while shift < pos and pattern[pos - 1] != pattern[pos - shift - 1]:
                shift += shifts[pos - shift - 1]
            shifts[pos] = shift
        return shifts

    @staticmethod
    def camel_case_split(identifier):
        matches = finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
        return [m.group(0) for m in matches]

    @staticmethod
    def split_multStr(split_char, string):
        strs = [string]
        for char in split_char:
            tmp = []
            for str in strs:
                str_splited = str.split(char)
                for each in str_splited:
                    tmp.append(each)
            strs = tmp
        strs = [x for x in strs if x != '']
        tmp = []
        for each in strs:
            t = Utilities.camel_case_split(each)
            for str in t:
                tmp.append(str)
        return tmp

    @staticmethod
    def Json_Parse(full=True):
        nltk.download('wordnet')
        wordnet_lemmatizer = nltk.WordNetLemmatizer()
        file = open(os.path.join(APP_STATIC, "feeds.csv"), 'r', encoding='utf-8-sig')
        lines = reader(file)
        type = set()
        corpus = []
        document = []
        uids = []
        uid = set()
        for line in lines:
            oJson = json.loads(line[1])
            for object in oJson:
                pair = (object['type'], object['uid'])
                uid.add(pair)
                for value in object['values']:
                    splited_value = Utilities.split_multStr(["[", "]", " ", "(", ")", ".", "_", "-"], value)
                    # print(value)
                    document.append(value)
                    splited_value.append(line[0])
                    corpus.append(" ".join(splited_value))
                    if full:
                        uids.append(object['uid'])
                    else:
                        if 'size' == object['uid'] or 'valueAxis2' == object['uid'] or 'bubbleWidth' == object['uid']:
                            uids.append('valueAxis')
                        elif 'color' == object['uid'] or 'categoryAxis2' == object['uid']:
                            uids.append('categoryAxis')
                        else:
                            uids.append(object['uid'])


                    for each in splited_value:
                        # print("     ", wordnet_lemmatizer.lemmatize(each.lower()))
                        type.add(wordnet_lemmatizer.lemmatize(each.lower()))
        print(uid)
        return corpus, uids
