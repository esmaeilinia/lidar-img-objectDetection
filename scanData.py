## Samuel Rohrer (rohrer) and Ian Lin
#  October 30, 2016
#  read in LIDAR data and convert to nice struct
import os
from collections import namedtuple

# kinda like a c++ struct, just a simple tuple
Scan = namedtuple("Scan", "x y r z")
Segment = namedtuple("Segment", "startIdx endIdx")

# constants
lidarDir = './ISRtest_frames'
frameDir = './ISRtest_LIDARlog'

class Scans():
    def __init__(self):
        # array of 4 arrays of Scan objects
        self.laser = [[], [], [], []]
        # array of 4 arrays of Segment objects
        self.segments = [[], [], [], []]

    def readLidarData(self, filename):
        for filename in os.listdir(lidarDir):
            with open(filename) as f:

        None

# example: the z value for the 10th scan from laser 2 is
scans = Scans()
# index from 0
scans.laser[1][9].z

# example: create a segment representing the first 6
# points of laser 4 and add into class
seg = Segment(0 5)
scans.segments[3].append(seg)
