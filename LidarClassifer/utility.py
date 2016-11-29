## Samuel Rohrer (rohrer) and Ian Lin
#  November 29, 2016
#  Utility functions and classes
from collections import namedtuple

# kinda like a c++ struct, just a simple tuple
Scan = namedtuple("Scan", "x y r z")
Segment = namedtuple("Segment", "startIdx endIdx")
