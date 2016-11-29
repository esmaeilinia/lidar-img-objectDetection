from segmentData import extractSegments
from extractLidarFeatures import extractFeatures
from LidarFeatureClassifer import lidarFeatureClassifer
from utility import Scan, Segment

# steps:
# 1. train
# 2. test
class LidarDriver():
    def __init__(self):
        self.trainFeatures = []
        self.trainClasses = []
        self.lfc = lidarFeatureClassifer()
        # array of 4 arrays of Scan objects
        self.laser = [[], [], [], []]
        # array of 4 arrays of Segment objects
        self.segments = [[], [], [], []]

    # 1. read training segments from Train_pos_segments and Train_neg_segments
    # 2. extract features into self.trainFeatures
    # 3. read Laser_train_class into self.trainClasses
    # 4. fit self.lfc with self.trainFeatures and self.trainClasses
    def train(self):
        None

    # for each image in ISRtest_frames with its corresponding lidar scan file from ISRtest_LIDARlog
    # 1. open and display the image
    # 2. read the lidar file into self.laser
    # 3. segment each laser (4 total) into self.segments
    # 4. for each segment, classify with self.lfc
    # 5. overlay lidar scans with different colors if segment contains pedestrian or not
    # 6. close image
    def test(self):
        None
    # def readLidarData(self):
    #     #for filename in os.listdir(lidarDir):
    #     #    with open(filename) as f:
    #     # Lets do one file for now
    #     Tetha = np.array([-1.6, -0.8, 0.8, 1.6])*math.pi/180
    #     with open(lidarLogWithHuman) as f:
    #         # read header and blank line after
    #         header_line = next(f)
    #         next(f)
    #         for line in f:
    #             data = line.split()
    #             laserNum = int(data[0])
    #             x = float(data[3])
    #             y = float(data[4])
    #             r = math.sqrt(x*x+y*y)
    #             z = r*math.tan(Tetha[laserNum])
    #             scan = Scan(x, y, r, z)
    #             self.laser[laserNum].append(scan)

if __name__ == "__main__":
    ld = LidarDriver()
    ld.train()

# # constants - update as needed
# writeTrainFilename = 'trainFeatures.txt'
# writeTestFilename = 'testFeatures.txt'
# trainPosGlobPath = '../../../../../Desktop/442Data/Train_pos_segments/*.txt'
# trainNegGlobPath = '../../../../../Desktop/442Data/Train_neg_segments/*.txt'
# testPosGlobPath = '../../../../../Desktop/442Data/Test_pos_segments/*.txt'
# testNegGlobPath = '../../../../../Desktop/442Data/Test_neg_segments/*.txt'
#
# actualDataGlobPath = '../segmentData/*.txt'
# # write features extracted from files in a folder path
# def writeFeaturesToFile(posGlobPath, negGlobPath, writeFilename):
# 	writefile = open(writeFilename, 'w')
# 	posfilename = glob.glob(posGlobPath)
# 	for i in range(0, len(posfilename)):
# 		features = extractFeatures(posfilename[i])
# 		if len(features) > 0:
# 			# write these to file
# 			val = " ".join([str(x) for x in features])
# 			writefile.write(val)
# 			writefile.write("\n")
#
# 	if posGlobPath != negGlobPath:
# 		negfilename = glob.glob(negGlobPath)
# 		for i in range(0, len(negfilename)):
# 			features = extractFeatures(negfilename[i])
# 			# write these to file
# 			val = " ".join([str(x) for x in features])
# 			writefile.write(val)
# 			writefile.write("\n")
# 	writefile.close()
