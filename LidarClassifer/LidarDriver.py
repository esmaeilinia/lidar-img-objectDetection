from segmentData import extractSegments
from extractLidarFeatures import extractFeatures
from utility import Scan, Segment
import os
import glob
from sklearn.tree import DecisionTreeClassifier

# constants
ianTrainPos = '../../../../../Desktop/442Data/Train_pos_segments/*.txt'
ianTrainNeg = '../../../../../Desktop/442Data/Train_neg_segments/*.txt'
ianTestPos = '../../../../../Desktop/442Data/Test_pos_segments/*.txt'
ianTestNeg = '../../../../../Desktop/442Data/Test_neg_segments/*.txt'
ianTrainClasses = 'Laser_train_class.txt'
ianTestClasses = 'Laser_test_class.txt'

# steps:
# 1. train
# 2. test
class LidarDriver():
    def __init__(self):
        self.trainFeatures = []
        self.testFeatures = []
        self.trainClasses = []
        self.testClasses = []
        self.lfc = DecisionTreeClassifier()
        # array of 4 arrays of Scan objects
        self.laser = [[], [], [], []]
        # array of 4 arrays of Segment objects
        self.segments = [[], [], [], []]

    # 1. read training segments from Train_pos_segments and Train_neg_segments
    # 2. extract features into self.trainFeatures
    # 3. read Laser_train_class into self.trainClasses
    # 4. fit self.lfc with self.trainFeatures and self.trainClasses
    def train(self):
        self.readTrainingSegments()
        with open(ianTrainClasses) as f:
            for line in f.readlines():
                self.trainClasses.append(int(line))
        self.lfc.fit(self.trainFeatures, self.trainClasses)

    def readTrainingSegments(self):
        self.readGlob(ianTrainPos, 1)
        self.readGlob(ianTrainNeg, 1)

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
        self.readGlob(ianTestPos, 0)
        self.readGlob(ianTestNeg, 0)
        with open(ianTestClasses) as f:
            for line in f.readlines():
                self.testClasses.append(int(line))
        print(self.lfc.score(self.testFeatures, self.testClasses))

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
    ld.testExtractFeatures()
    ld.test()
