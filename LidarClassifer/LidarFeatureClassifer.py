import sys
from sklearn.tree import DecisionTreeClassifier

class lidarFeatureClassifer:
    def __init__(self):
        None

    def readDataFile(self, filename):
        x = []
        with open(filename) as f:
            for line in f.readlines():
                featureVector = []
                for feature in line.split():
                    featureVector.append(float(feature))
                x.append(featureVector)
        return x

    def readClassificationFile(self, filename):
        y = []
        with open(filename) as f:
            for word in f.read().split():
                y.append(int(word))
        return y

    def analyze(self, yPredict, yTest):
        correctLabel = 0
        for p, l in map(None, yPredict, yTest):
            if p == l:
                correctLabel += 1
        print(str(correctLabel/float(len(yPredict))*100) + '% accuracy')
