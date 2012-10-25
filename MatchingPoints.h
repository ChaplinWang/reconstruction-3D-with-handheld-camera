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

using namespace std;
using namespace cv;

struct Colours {
	int red;
	int blue;
	int green;
};

class MatchingPoints {
private:
	vector<KeyPoint> keyPoints1;
	vector<KeyPoint> keyPoints2;
	vector<Colours> pointColours;
	Mat fundamentalMatrics;
	// Match the two images
	vector<DMatch> matches;
	vector<KeyPoint> fullKeypoints1, fullKeypoints2;
	bool enoughMatches;
public:
	MatchingPoints(Mat image1, Mat image2);
	vector<KeyPoint> getKeyPoints1();
	vector<KeyPoint> getKeyPoints2();
	Mat getFundamentalMatrix();
	void displayFull(Mat image1, Mat image2);
	bool hasEnoughMatches();
	vector<Colours> getColours(Mat frame1);
};
