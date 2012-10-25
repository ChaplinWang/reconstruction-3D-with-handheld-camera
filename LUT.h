#include <iostream> // for standard I/O
#include <string>   // for strings
#include <iomanip>  // for controlling float print precision
#include <sstream>  // string to number conversion
#include <vector>

#include <opencv2/imgproc/imgproc.hpp>  // Gaussian Blur
#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/legacy/legacy.hpp>
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv/cv.h"
#include "projectStructs.h"

using namespace std;
using namespace cv;


typedef struct LUT_ENRTY{
	Point2d *fp;
	Point3d *cp;
}Entry;


class LUTable {
  public:
	LUTable();
	void init();
    void add_entry(Point3d *cp, Point2d *fp);  // add to the end of the list
	Point3d* find_3d(Point2d fp);				//GET 3D cord given 2D
	void cleanup();
	bool test();
	void addAllEntries(vector<KeyPoint> twoDee, vector<SpacePoint> threeDee);
	//Matx34d getNewCameraMatrix(vector<KeyPoint> newKeyPoints);

	int tableSize();

  private:
	vector <Entry> table;				//storage of entries
	int entry_num;
	
};