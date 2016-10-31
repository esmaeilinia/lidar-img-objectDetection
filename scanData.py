## Samuel Rohrer (rohrer) and Ian Lin
#  October 30, 2016
#  read in LIDAR data and convert to nice struct
from collections import namedtuple

# kinda like a c++ struct, just a simple tuple
Scan = namedtuple("Scan", "x y r z")

class Scans():
    def __init__(self):
        # array of 4 arrays of Scan objects
        self.laser = [[], [], [], []]

    def readLidarData(self, filename):
        None

# example: the z value for the 10th scan from laser 2 is
scans = Scans()
# index from 0
scans[1][9].z
