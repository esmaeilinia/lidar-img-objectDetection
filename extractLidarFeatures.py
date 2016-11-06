## Samuel Rohrer (rohrer) and Ian Lin
#  November 6, 2016
#  read in LIDAR segment data and return feature vector

import math

# product of num points and minimum range measurement
def feature1(x, y):
	r = [0] * len(x)
	for i in range(0, len(r)):
		r[i] = math.sqrt(x[i]*x[i] + y[i]*y[i])
	return len(r)*min(r)

# number of points in segment data
def feature2(x, y):
	return len(x)

# normalized cartesian dimension (CHECK THIS)
def feature3(x, y):
	deltaX = x[0] - x[len(x)-1]
	deltaY = y[0] - y[len(y)-1]
	return math.sqrt(deltaX*deltaX + deltaY*deltaY)

# internal standard deviation
def feature4(x, y):
	r = [0] * len(x)
	for i in range(0, len(r)):
		r[i] = math.sqrt(x[i]*x[i] + y[i]*y[i])

	sum = 0;
	for i in range(0, len(r)):
		sum += abs(r[i] - r[len(r)/2])
	sum = sum / len(r)
	return math.sqrt(sum)	

# called to call all of the other feature extraction functions
def extractFeatures(x, y):
	features = [feature1(x,y), feature2(x,y), feature3(x,y), feature4(x,y) ]
	return features



# read in data
file = open("../../Downloads/Laser_train/Train_pos_segments/15_55_14_0630_1.txt","r")
x_data = file.readline()
y_data = file.readline()
x_data = x_data.split()
y_data = y_data.split()
# go elementwise and make into floats
for i in range(0, len(x_data)):
	x_data[i] = float(x_data[i])
	y_data[i] = float(y_data[i])

# call feature extraction
features = extractFeatures(x_data, y_data)
print features 