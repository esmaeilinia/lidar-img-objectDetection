import sys
from sklearn.tree import DecisionTreeClassifier

# constants
#trainDataFilename = 'D-Laser_train_LIPD.txt'
trainDataFilename = 'trainFeatures.txt'
trainClassesFilename = 'Laser_train_class.txt'
#testDataFilename = 'D-Laser_test_LIPD.txt'
testDataFilename = 'testFeatures.txt'
testClassesFilename = 'Laser_test_class.txt'

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

lfc = lidarFeatureClassifer()
xTrain = lfc.readDataFile(trainDataFilename)
yTrain = lfc.readClassificationFile(trainClassesFilename)
xTest = lfc.readDataFile(testDataFilename)
yTest = lfc.readClassificationFile(testClassesFilename)

# train
dt = DecisionTreeClassifier()
dt.fit(xTrain, yTrain)
yPredict = dt.predict(xTest)
print(yPredict)

# check accuracy
#lfc.analyze(yPredict, yTest)
