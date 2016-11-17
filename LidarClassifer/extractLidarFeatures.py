## Samuel Rohrer (rohrer) and Ian Lin
#  November 6, 2016
#  read in LIDAR segment data and return feature vector

import math
import numpy as np
from scipy import optimize
import glob

# constants - update as needed
writeTrainFilename = 'trainFeatures.txt'
writeTestFilename = 'testFeatures.txt'
trainPosGlobPath = '../../../../../Desktop/442Data/Train_pos_segments/*.txt'
trainNegGlobPath = '../../../../../Desktop/442Data/Train_neg_segments/*.txt'
testPosGlobPath = '../../../../../Desktop/442Data/Test_pos_segments/*.txt'
testNegGlobPath = '../../../../../Desktop/442Data/Test_neg_segments/*.txt'

actualDataGlobPath = '../segmentData/*.txt'

# helper function to calc r


def findR(x, y):
	r = [0] * len(x)
	for i in range(0, len(r)):
		r[i] = math.sqrt(x[i]**2 + y[i]**2)
	return r

# product of num points and minimum range measurement


def feature1(x, y):
	r = findR(x, y)
	return len(r) * min(r)

# number of points in segment data


def feature2(x, y):
	return len(x)

# normalized cartesian dimension (CHECK THIS)


def feature3(x, y):
	deltaX = x[0] - x[len(x) - 1]
	deltaY = y[0] - y[len(y) - 1]
	return math.sqrt(deltaX**2 + deltaY**2)

# internal standard deviation


def feature4(x, y):
	r = findR(x, y)
	sum = 0
	for i in range(0, len(r)):
		sum += abs(r[i] - r[len(r) / 2])
	sum = sum / len(r)
	return math.sqrt(sum)

# skip feature 5 for now (can't find method)

# mean average deviation from the median


def feature6(x, y):
	r = findR(x, y)
	medR = np.median(r)
	sum = 0
	for i in range(0, len(r)):
		sum += abs(r[i] - medR)
	return sum / len(r)

# skip feature 7,8 for now (can't find method)

# feature 9: linearity


def line(x, A, B):
	return A * x + B


def feature9(x, y):
	# find the least squares line
	A, B = optimize.curve_fit(line, x, y)[0]
	# compute a vector of distances squared
	d = [0] * len(x)
	for i in range(0, len(x)):
		d[i] = (abs(-A * x[i] + 1 * y[i] - B) / math.sqrt(A * A + B * B))
		d[i] = d[i] * d[i]
	return sum(d) / len(d)

# feature 10: circularity
#def calcR(xcoord, ycoord, x, y):
#	return math.sqrt((x - xcoord)**2 + (y - ycoord)**2)

#def circDist(est, x, y):
#	R = [0] * len(x)
#	for i in range(0, len(x)):
#		R[i] = calcR(est[0], est[1], x[i], y[i])
#	return sum(R)/len(R)

#def feature10(x, y):
#	center_est = np.array([sum(x)/len(x), sum(y)/len(y)])
#	center_2, ier = optimize.leastsq(circDist, center_est, args=(x, y))
#	print center_2, ier

# features 11,12,13 all are same
# momentFeature calculates second,third,fourth central moment


def momentFeature(x, y, ko):
	r = findR(x, y)
	meanR = sum(r) / len(r)
	term = [0] * len(x)
	for i in range(0, len(x)):
		term[i] = ((r[i] - meanR)**ko) / len(x)
	return sum(term)

# feature 14


def calcDist(x, y):
	return math.sqrt(x**2 + y**2)


def feature14help(x, y):
	diffDist = [0] * (len(x) - 1)
	for i in range(0, len(x) - 1):
		diffDist[i] = abs(calcDist(x[i], y[i]) - calcDist(x[i + 1], y[i + 1]))
	return diffDist


def feature14(x, y):
	diffDist = feature14help(x, y)
	return sum(diffDist)

# feature 15


def feature15(x, y):
	diffDist = feature14help(x, y)
	return np.std(diffDist)

# called to call all of the other feature extraction functions


def extractFeatures(filename):
	# read in data
	file = open(filename, "r")
	x_data = file.readline()
	y_data = file.readline()
	x = x_data.split()
	y = y_data.split()
	# if segment is too short, just return empty array
	if len(x) < 3:
		return []
	# go elementwise and make into floats
	for i in range(0, len(x)):
		x[i] = float(x[i])
		y[i] = float(y[i])

	features = [feature1(x, y), feature2(x, y), feature3(x, y), feature4(x, y),
             feature6(x, y), feature9(x, y), momentFeature(x, y, 2),
             momentFeature(x, y, 3), momentFeature(x, y, 4), feature14(x, y),
             feature15(x, y)]
	return features

# write features extracted from files in a folder path


def writeFeaturesToFile(posGlobPath, negGlobPath, writeFilename):
	writefile = open(writeFilename, 'w')
	posfilename = glob.glob(posGlobPath)
	for i in range(0, len(posfilename)):
		features = extractFeatures(posfilename[i])
		if len(features) > 0:
			# write these to file
			val = " ".join([str(x) for x in features])
			writefile.write(val)
			writefile.write("\n")

	if posGlobPath != negGlobPath:
		negfilename = glob.glob(negGlobPath)
		for i in range(0, len(negfilename)):
			features = extractFeatures(negfilename[i])
			# write these to file
			val = " ".join([str(x) for x in features])
			writefile.write(val)
			writefile.write("\n")

	writefile.close()

# write both train and test features from segmented data
writeFeaturesToFile(trainPosGlobPath, trainNegGlobPath, writeTrainFilename)
#writeFeaturesToFile(testPosGlobPath, testNegGlobPath, writeTestFilename)
# TODO: this and the function is hacky, we should connect these various methods better
writeFeaturesToFile(actualDataGlobPath, actualDataGlobPath, writeTestFilename)
