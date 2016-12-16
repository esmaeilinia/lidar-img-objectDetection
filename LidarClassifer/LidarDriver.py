import time
import os
import glob
import numpy as np
import math
import matplotlib
matplotlib.rcParams['backend'] = "MacOSX"
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.tree import DecisionTreeClassifier
from segmentData import extractSegments
from extractLidarFeatures import extractFeatures
from utility import Scan, Segment

# for people detect.py
import sys
from cv import *
from peopleDetectSample import peopleDetect

# constants
TrainPos = 'Train_pos_segments/*.txt'
TrainNeg = 'Train_neg_segments/*.txt'
TestPos = 'Test_pos_segments/*.txt'
TestNeg = 'Test_neg_segments/*.txt'
TrainClasses = 'Laser_train_class.txt'
TestClasses = 'Laser_test_class.txt'
TestImages = 'SmallTestFrames/*.jpg'
TestScans = 'SmallTestLidar/*.txt'

# steps:
# 1. train
# 2. test
class LidarDriver():
    def __init__(self):
        self.trainFeatures = []
        self.testFeatures = []
        self.trainClasses = []
        self.testClasses = []
        self.lfc = DecisionTreeClassifier(max_depth=4)
        # array of 4 arrays of Scan objects
        self.lasers = [[], [], [], []]
        # array of 4 arrays of Segment objects
        self.segments = []
        self.pedestrianSegs = []
        self.colors = ['b','g','r','y']

    # 1. read training segments from Train_pos_segments and Train_neg_segments
    # 2. extract features into self.trainFeatures
    # 3. read Laser_train_class into self.trainClasses
    # 4. fit self.lfc with self.trainFeatures and self.trainClasses
    def train(self):
        self.readTrainingSegments()
        with open(TrainClasses) as f:
            for line in f.readlines():
                self.trainClasses.append(int(line))
        self.lfc.fit(self.trainFeatures, self.trainClasses)

    def readTrainingSegments(self):
        self.readGlob(TrainPos, 1)
        self.readGlob(TrainNeg, 1)

    def readGlob(self, globPath, train):
        for filename in glob.glob(globPath):
            with open(filename) as f:
                X = map(float, f.readline().split())
                Y = map(float, f.readline().split())
            scans = []
            for x,y in zip(X, Y):
                # r and z don't matter for feature extractions
                scans.append(Scan(x, y, 0, 0))
            if train:
                self.trainFeatures.append(extractFeatures(scans))
            else:
                self.testFeatures.append(extractFeatures(scans))

    def testExtractFeatures(self):
        self.readGlob(TestPos, 0)
        self.readGlob(TestNeg, 0)
        with open(TestClasses) as f:
            for line in f.readlines():
                self.testClasses.append(int(line))
        print(self.lfc.score(self.testFeatures, self.testClasses))

    def filterSegments(self):
        thd = 10;
        deleteList = [[], [], [], []]
        # iterate over all the segments
        for laserNum in range(4):
            for segment in range(len(self.pedestrianSegs[laserNum])):
                l0_start = self.pedestrianSegs[laserNum][segment].startIdx
                l0_end = self.pedestrianSegs[laserNum][segment].endIdx
                numSimilar = 0
                # iterate over the other lasers
                for otherLaser in range(4):
                    if laserNum == otherLaser:
                        continue
                    for segment2 in range(len(self.pedestrianSegs[otherLaser])):
                        start = self.pedestrianSegs[otherLaser][segment2].startIdx
                        end = self.pedestrianSegs[otherLaser][segment2].endIdx
                        if abs(start - l0_start) < thd and abs(end - l0_end) < thd:
                            numSimilar += 1

                # check if enough similar to identify pedestrian
                if numSimilar < 1 :
                    deleteList[laserNum].append(self.pedestrianSegs[laserNum][segment])
        for laserNum in range(4):
            for segment in deleteList[laserNum]:
                self.pedestrianSegs[laserNum].remove(segment)

    # for each image in ISRtest_frames with its corresponding lidar scan file from ISRtest_LIDARlog
    # 1. open and display the image
    # 2. read the lidar file into self.laser
    # 3. segment each laser (4 total) into self.segments
    # 4. for each segment, extract features and classify with self.lfc
    # 5. overlay lidar scans with different colors if segment contains pedestrian or not
    # 6. close image
    def test(self):
        imgNames = glob.glob(TestImages)
        lidarNames = glob.glob(TestScans)
        for img, lidar in zip(imgNames, lidarNames):
            found = False
            self.segments = []
            self.lasers = [[], [], [], []]
            self.pedestrianSegs = [[], [], [], []]
            self.readLidarData(lidar)
            # convert lasers into segments
            for laser in self.lasers:
                self.segments.append(extractSegments(laser))
            # extract features and classify
            for segmentNum in range(len(self.segments)):
                for segment in self.segments[segmentNum]:
                    # ignore segments outside image
                    if segment.startIdx < 60 or segment.endIdx > 250:
                        continue
                    featureVec = extractFeatures(
                        self.lasers[segmentNum][segment.startIdx:segment.endIdx],
                        0
                    )
                    # if value is too high, then predict will throw error
                    if featureVec and featureVec[1] > 7 and featureVec[1] < 20 and all(i < 1e10 for i in featureVec):
                        p = self.lfc.predict(np.array(featureVec).reshape(1, -1))
                        if p:
                            self.pedestrianSegs[segmentNum].append(segment)
                            found = True
            if found:
                # filtering
                print('running with filtering may get false negatives')
                self.filterSegments()
                self.showImage(img)

    def readLidarData(self, lidarFile):
        Tetha = np.array([-1.6, -0.8, 0.8, 1.6])*math.pi/180
        with open(lidarFile) as f:
            # read header and blank line after
            header_line = next(f)
            next(f)
            for line in f:
                data = line.split()
                laserNum = int(data[0])
                x = float(data[3])
                y = float(data[4])
                r = math.sqrt(x*x+y*y)
                z = r*math.tan(Tetha[laserNum])
                scan = Scan(x, y, r, z)
                self.lasers[laserNum].append(scan)

    def showPoints(self):
        fc = np.array([ 622.06, 622.56])
        cc = np.array([ 366.08, 258.92])
        K =  np.array([ [fc[0], -0.05, cc[0]], [ 0, fc[1], cc[1]], [ 0,0,1]])
        Delta = np.array([[-18.592], [-259.84], [-8.2989]])
        Phi = np.array([[0.999, 0.05, 0.0001], [-0.005, 0.998, 0.001],
                 [0.008, -0.018, 0.984]])
        L = np.array([ [Phi[0][0], Phi[0][1], Phi[0][2], Delta[0]],
            [ Phi[2][0], Phi[2][1], Phi[2][2], Delta[2]],
            [ Phi[1][0], Phi[1][1], Phi[1][2], Delta[1]],
            [ 0, 0, 0, 1]])
        # iterate over the lasers
        for laserNum in range(len(self.segments)):
            # iterate over the segments in each laser
            for pointNum in range(len(self.lasers[laserNum])):
                point = np.array([[-10*self.lasers[laserNum][pointNum].y],
                    [10*self.lasers[laserNum][pointNum].x],
                    [-10*self.lasers[laserNum][pointNum].z], [1]])
                x_s = np.dot(np.eye(3,4), (np.dot(np.linalg.pinv(L), point)))
                x_s = x_s / x_s[2]
                x_s = np.dot(K, x_s)
                # exclude points outside the image
                if x_s[0] > 0 and x_s[0] < 640 and x_s[1] > 0 and x_s[1] < 480:
                    plt.plot((x_s[0]), (x_s[1]), color=self.colors[laserNum], marker='+', markersize=2)

    def showSegments(self):
        # plt.plot((x1, x2), (y1, y2), 'k-')
        fc = np.array([ 622.06, 622.56])
        cc = np.array([ 366.08, 258.92])
        K =  np.array([ [fc[0], -0.05, cc[0]], [ 0, fc[1], cc[1]], [ 0,0,1]])
        Delta = np.array([[-18.592], [-259.84], [-8.2989]])
        Phi = np.array([[0.999, 0.05, 0.0001], [-0.005, 0.998, 0.001],
                 [0.008, -0.018, 0.984]])
        L = np.array([ [Phi[0][0], Phi[0][1], Phi[0][2], Delta[0]],
            [ Phi[2][0], Phi[2][1], Phi[2][2], Delta[2]],
            [ Phi[1][0], Phi[1][1], Phi[1][2], Delta[1]],
            [ 0, 0, 0, 1]])
        # iterate over the lasers
        for laserNum in range(len(self.segments)):
            # iterate over the segments in each laser
            for segmentNum in self.pedestrianSegs[laserNum]:
                # find start and end idx then plot
                start = segmentNum.startIdx
                end = segmentNum.endIdx
                start_point = np.array([[-10*self.lasers[laserNum][start].y],
                    [10*self.lasers[laserNum][start].x],
                    [-10*self.lasers[laserNum][start].z], [1]])
                end_point = np.array([[-10*self.lasers[laserNum][end].y],
                   [10*self.lasers[laserNum][end].x],
                   [-10*self.lasers[laserNum][end].z], [1]])
                x_s = np.dot(np.eye(3,4), (np.dot(np.linalg.pinv(L), start_point)))
                x_e = np.dot(np.eye(3,4), (np.dot(np.linalg.pinv(L), end_point)))
                x_s = x_s / x_s[2]
                x_e = x_e / x_e[2]
                x_s = np.dot(K, x_s)
                x_e = np.dot(K, x_e)
                #print(laserNum, segmentNum)
                # plot them
                # make sure the segment is in the range of the image
                if x_s[0] > 0 and x_e[0] < 640 and x_s[1] > 0 and x_e[1] < 480:
                    plt.plot((x_s[0], x_e[0]), (x_s[1], x_e[1]), 'go', markersize=4)

    def showImage(self, filename):
        plt.close()
        plt.figure()
        img = mpimg.imread(filename)
        plt.imshow(img)
        self.showPoints()
        self.showSegments()
        peopleDetect(filename)
        plt.show()

if __name__ == "__main__":
    ld = LidarDriver()
    ld.train()
    ld.testExtractFeatures()
    ld.test()
