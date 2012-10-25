#include <iostream> // for standard I/O
#include <string>   // for strings
#include <iomanip>  // for controlling float print precision
#include <sstream>  // string to number conversion

#include <opencv2/imgproc/imgproc.hpp>  // Gaussian Blur
#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/legacy/legacy.hpp>
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv/cv.h"
//#include "projectStructs.h"

using namespace std;
using namespace cv;

#define DEBUG 0


class addingPoints {

private:
	Mat Points2d;
	Mat Points3d;

	Mat overlapping

public:
	void addingInitialPoints(vector<KeyPoint> keypoints, vector<SpacePoint> pointCloud, Mat calibrationMatrix);

	addingPoints();

	Mat get2dPoints();

	Mat get3dPoints();

	Mat lookup3D(KeyPoint keypoint);

	bool hasKeypoint(KeyPoint keypoint);

	Mat getCameraMatrix();

	void setUpNewImage(vector<KeyPoint> newKeyPoints1, Vector<KeyPoint> newKeyPoints2);

};