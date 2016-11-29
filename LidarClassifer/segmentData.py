## Samuel Rohrer (rohrer) and Ian Lin
#  October 30, 2016
#  read in LIDAR data and convert to nice struct
import os
from collections import namedtuple
import math
import numpy as np

# kinda like a c++ struct, just a simple tuple
Scan = namedtuple("Scan", "x y r z")
Segment = namedtuple("Segment", "startIdx endIdx")

def findCos3dPoint(x, xp1):
	mag_x = math.sqrt(x.x**2 + x.y**2 + x.z**2)
	mag_xp1 = math.sqrt(xp1.x**2 + xp1.y**2 + xp1.z**2)
	dot = x.x * xp1.x + x.y * xp1.y + x.z * xp1.z
	return dot / (mag_x * mag_xp1)

# TODO: implement
# extractSegments(laser) where laser is a list of Scans
# outputs a list of Segments
# note that the input doesn't have lasernum anymore, the driver will handle it
def extractSegments(laser):
    print("extracting segments")
    return []
    #
    # # start making segmentation data
    # # iterate over the laser
    # for i in range(0, 4):
    # 	# iterate over length of laser scan
    # 	last_new_seg_idx = 0
    # 	for j in range(0, len(scans.laser[i])-1):
    # 		cos_alpha = findCos3dPoint(scans.laser[i][j], scans.laser[i][j + 1])
    # 		r = scans.laser[i][j].r
    # 		rp1 = scans.laser[i][j + 1].r
    # 		dist = math.sqrt(r**2 + rp1**2 - 2 * r * rp1 * cos_alpha)
    # 		c0 = 600
    # 		c1 = math.sqrt(2 * (1 - cos_alpha))
    # 		dist_thd = c0 + c1 * min(r, rp1)
    # 		# check if its a new segment
    # 		if (dist > dist_thd):
    # 			seg = Segment(last_new_seg_idx, j)
    # 			scans.segments[i].append(seg)
    # 			last_new_seg_idx = j
    #
    # # write each segment to a new file
    # # iterate over each laser
    # for l in range(0, 4):
    # 	for i in range(0, len(scans.segments[l])):
    # 		# TODO: figure out how to write to many different files
    # 		file = open('segmentData/laser#%i_segment#%i.txt' % (l, i), 'w')
    # 		# go through segments and print coordinates
    # 		lenSegment = scans.segments[l][i].endIdx - scans.segments[l][i].startIdx
    # 		# print x
    # 		for x in range(0, lenSegment):
    # 			val = scans.laser[l][x].x
    # 			file.write(str(val) + " ")
    # 		file.write("\n")
    # 		# print y
    # 		for y in range(0, lenSegment):
    # 			val = scans.laser[l][y].y
    # 			file.write(str(val) + " ")
    # 		file.write("\n")
    # 		# print z
    # 		for z in range(0, lenSegment):
    # 			val = scans.laser[l][z].z
    # 			file.write(str(val) + " ")
    # 		file.write("\n")
    # 		file.close()
