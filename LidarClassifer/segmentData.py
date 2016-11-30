## Samuel Rohrer (rohrer) and Ian Lin
#  October 30, 2016
#  read in LIDAR data and convert to nice struct
import math
from utility import Scan, Segment

def extractSegments(laser):
    # start making segmentation data
    # iterate over length of laser scan
    segments = []
    last_new_seg_idx = 0
    for j in range(0, len(laser)-1):
        cos_alpha = findCos3dPoint(laser[j], laser[j + 1])
        r = laser[j].r
        rp1 = laser[j + 1].r
        dist = math.sqrt(r**2 + rp1**2 - 2 * r * rp1 * cos_alpha)
        c0 = 600
        # TODO: CHECK WITH SAM, is this okay?
        # throws an error if cos_alpha is close to one but not one
        # probably underflow
        if cos_alpha > 0.999 and cos_alpha < 1.001:
            c1 = 0
        else:
            c1 = math.sqrt(2 * (1 - cos_alpha))
        dist_thd = c0 + c1 * min(r, rp1)
        # check if its a new segment
        if (dist > dist_thd):
            seg = Segment(last_new_seg_idx, j)
            segments.append(seg)
            last_new_seg_idx = j
    # return the list of segments
    return segments

# internal helper function
def findCos3dPoint(x, xp1):
	mag_x = math.sqrt(x.x**2 + x.y**2 + x.z**2)
	mag_xp1 = math.sqrt(xp1.x**2 + xp1.y**2 + xp1.z**2)
	dot = x.x * xp1.x + x.y * xp1.y + x.z * xp1.z
	return dot / (mag_x * mag_xp1)
