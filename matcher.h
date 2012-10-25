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

class Matcher {
private:	
	// pointer to the feature point detector object
	cv::Ptr<cv::FeatureDetector> detector;
	// pointer to the feature descriptor extractor object
	cv::Ptr<cv::DescriptorExtractor> extractor;
	float ratio; // max ratio between 1st and 2nd NN
	bool refineF; // if true will refine the F matrix
	double distance; // min distance to epipolar
	double confidence; // confidence level (probability)
	Mat fundamentalMatrics;
public:
	Matcher() ;

	Mat getFundamentalMatrix();

	void setFeatureDetector(Ptr<FeatureDetector> &detect);

	void setDescriptorExtractor(Ptr<DescriptorExtractor> & desExtract);

	void setMinDistanceToEpipolar(double distance);

	void setRatio(float ratio);

	bool match (Mat &image1, Mat &image2, vector<DMatch> &matches, vector<KeyPoint> &keypoints1,vector<KeyPoint> &keypoints2);

	int ratioTest(std::vector<std::vector<cv::DMatch>> &matches);

	void symmetryTest(const std::vector<std::vector<cv::DMatch>>& matches1,	const std::vector<std::vector<cv::DMatch>>& matches2, std::vector<cv::DMatch>& symMatches);

	void setConfidenceLevel(double confidence);

	cv::Mat ransacTest(
		const std::vector<cv::DMatch>& matches,
		const std::vector<cv::KeyPoint>& keypoints1,
		const std::vector<cv::KeyPoint>& keypoints2,
		std::vector<cv::DMatch>& outMatches);


};

