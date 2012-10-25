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
//#include "projectHeaders.h"
#include "matcher.h"

using namespace std;
using namespace cv;

Matcher::Matcher() : ratio(0.65f), refineF(true),
	confidence(0.99), distance(3.0) {
		// SURF is the default feature
		detector= new cv::SurfFeatureDetector();
		extractor= new cv::SurfDescriptorExtractor();
}

Mat Matcher::getFundamentalMatrix() {
	return fundamentalMatrics;
}

// Set the feature detector
void Matcher::setFeatureDetector(
	cv::Ptr<cv::FeatureDetector>& detect) {
		detector= detect;
}
// Set the descriptor extractor
void Matcher::setDescriptorExtractor(
	cv::Ptr<cv::DescriptorExtractor>& desc) {
		extractor= desc;
}

void Matcher::setConfidenceLevel(double confidenceIn) {
	confidence = confidenceIn;
}

void Matcher::setMinDistanceToEpipolar(double distanceIn) {
	distance = distanceIn;
}

void Matcher::setRatio(float ratioIn) {
	ratio = ratioIn;
}


// Match feature points using symmetry test and RANSAC
// returns fundemental matrix
bool Matcher::match(cv::Mat& image1,
	cv::Mat& image2, // input images
	// output matches and keypoints
	std::vector<cv::DMatch>& matches,
	std::vector<cv::KeyPoint>& keypoints1,
	std::vector<cv::KeyPoint>& keypoints2) {

		bool enoughMatches = false;
		// 1a. Detection of the SURF features
		detector->detect(image1,keypoints1);
		detector->detect(image2,keypoints2);
		// 1b. Extraction of the SURF descriptors
		cv::Mat descriptors1, descriptors2;
		extractor->compute(image1,keypoints1,descriptors1);
		extractor->compute(image2,keypoints2,descriptors2);
		// 2. Match the two image descriptors
		// Construction of the matcher
		cv::BruteForceMatcher<cv::L2<float>> matcher;
		// from image 1 to image 2
		// based on k nearest neighbours (with k=2)		

		std::vector<std::vector<cv::DMatch>> matches1;
		matcher.knnMatch(descriptors1,descriptors2,
			matches1, // vector of matches (up to 2 per entry)
			2); // return 2 nearest neighbours
		// from image 2 to image 1
		// based on k nearest neighbours (with k=2)
		std::vector<std::vector<cv::DMatch>> matches2;
		matcher.knnMatch(descriptors2,descriptors1,
			matches2, // vector of matches (up to 2 per entry)
			2); // return 2 nearest neighbours
		// 3. Remove matches for which NN ratio is
		// > than threshold
		// clean image 1 -> image 2 matches
		int removed= ratioTest(matches1);
		// clean image 2 -> image 1 matches
		removed= ratioTest(matches2);
		// 4. Remove non-symmetrical matches
		std::vector<cv::DMatch> symMatches;
		symmetryTest(matches1,matches2,symMatches);
		// 5. Validate matches using RANSAC

		//cout << "CHECK " << endl;
		//cout << symMatches.size();

		//check if there are enough matches in the first place
		if (symMatches.size() < 30) {
			enoughMatches = false;
		} else {
			enoughMatches = true;
			fundamentalMatrics= ransacTest(symMatches,
				keypoints1, keypoints2, matches);
			//cout << " " << matches.size() << " ";
			if (matches.size() < 25) {
				enoughMatches = false;
			}
		}
		//cout << "END CHECK" << endl;
		return enoughMatches;
}

int Matcher::ratioTest(std::vector<std::vector<cv::DMatch>> &matches) {
	int removed=0;
	// for all matches
	for (std::vector<std::vector<cv::DMatch>>::iterator
		matchIterator= matches.begin();
		matchIterator!= matches.end(); ++matchIterator) {
			// if 2 NN has been identified
			if (matchIterator->size() > 1) {
				// check distance ratio
				if ((*matchIterator)[0].distance/
					(*matchIterator)[1].distance > ratio) {
						matchIterator->clear(); // remove match
						removed++;
				}
			} else { // does not have 2 neighbours
				matchIterator->clear(); // remove match
				removed++;
			}
	}
	return removed;
}

// Insert symmetrical matches in symMatches vector
void Matcher::symmetryTest(
	const std::vector<std::vector<cv::DMatch>>& matches1,
	const std::vector<std::vector<cv::DMatch>>& matches2,
	std::vector<cv::DMatch>& symMatches) {
		// for all matches image 1 -> image 2
		for (std::vector<std::vector<cv::DMatch>>::
			const_iterator matchIterator1= matches1.begin();
			matchIterator1!= matches1.end(); ++matchIterator1) {
				// ignore deleted matches
				if (matchIterator1->size() < 2)
					continue;
				// for all matches image 2 -> image 1
				for (std::vector<std::vector<cv::DMatch>>::
					const_iterator matchIterator2= matches2.begin();
					matchIterator2!= matches2.end();
				++matchIterator2) {
					// ignore deleted matches
					if (matchIterator2->size() < 2)
						continue;
					// Match symmetry test
					if ((*matchIterator1)[0].queryIdx ==
						(*matchIterator2)[0].trainIdx &&
						(*matchIterator2)[0].queryIdx ==
						(*matchIterator1)[0].trainIdx) {
							// add symmetrical match
							symMatches.push_back(
								cv::DMatch((*matchIterator1)[0].queryIdx,
								(*matchIterator1)[0].trainIdx,
								(*matchIterator1)[0].distance));
							break; // next match in image 1 -> image 2
					}
				}
		}
}

// Identify good matches using RANSAC
// Return fundemental matrix
cv::Mat Matcher::ransacTest(
	const std::vector<cv::DMatch>& matches,
	const std::vector<cv::KeyPoint>& keypoints1,
	const std::vector<cv::KeyPoint>& keypoints2,
	std::vector<cv::DMatch>& outMatches) {

		// Convert keypoints into Point2f
		std::vector<cv::Point2f> points1, points2;
		for (std::vector<cv::DMatch>::
			const_iterator it= matches.begin();
			it!= matches.end(); ++it) {
				// Get the position of left keypoints
				float x= keypoints1[it->queryIdx].pt.x;
				float y= keypoints1[it->queryIdx].pt.y;
				points1.push_back(cv::Point2f(x,y));
				// Get the position of right keypoints
				x= keypoints2[it->trainIdx].pt.x;
				y= keypoints2[it->trainIdx].pt.y;
				points2.push_back(cv::Point2f(x,y));
		}
		// Compute F matrix using RANSAC
		std::vector<uchar> inliers(points1.size(),0);
		cv::Mat fundemental= cv::findFundamentalMat(
			cv::Mat(points1),cv::Mat(points2), // matching points
			inliers, // match status (inlier or outlier)
			CV_FM_RANSAC, // RANSAC method
			distance, // distance to epipolar line
			confidence); // confidence probability
		// extract the surviving (inliers) matches
		std::vector<uchar>::const_iterator
			itIn= inliers.begin();
		std::vector<cv::DMatch>::const_iterator
			itM= matches.begin();
		// for all matches
		for ( ;itIn!= inliers.end(); ++itIn, ++itM) {
			if (*itIn) { // it is a valid match
				outMatches.push_back(*itM);
			}
		}
		if (refineF) {
			// The F matrix will be recomputed with
			// all accepted matches
			// Convert keypoints into Point2f
			// for final F computation
			points1.clear();
			points2.clear();
			for (std::vector<cv::DMatch>::
				const_iterator it= outMatches.begin();
				it!= outMatches.end(); ++it) {
					// Get the position of left keypoints
					float x= keypoints1[it->queryIdx].pt.x;
					float y= keypoints1[it->queryIdx].pt.y;
					points1.push_back(cv::Point2f(x,y));
					// Get the position of right keypoints
					x= keypoints2[it->trainIdx].pt.x;
					y= keypoints2[it->trainIdx].pt.y;
					points2.push_back(cv::Point2f(x,y));
			}
			// Compute 8-point F from all accepted matches
			fundemental= cv::findFundamentalMat(
				cv::Mat(points1),cv::Mat(points2), // matches
				CV_FM_8POINT); // 8-point method
		}
		return fundemental;
}



