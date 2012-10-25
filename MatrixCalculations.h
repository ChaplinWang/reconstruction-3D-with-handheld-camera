
//#include "MatchingPoints.h"
#include "projectStructs.h";

using namespace std;
using namespace cv;

#define DEBUG 0

class MatrixCalculations {
private:
	Mat K; 
public:
	MatrixCalculations();

	Mat findMatrixK(Mat image);

	Mat_<double> CalculateEssentialMatrix(Mat F);

	Mat_<double> CalculateCameraCalibrationMatrix(Mat &K);

	bool MatrixCalculations::DecomposeEtoRandT(
		Mat_<double>& E,
		Mat_<double>& R1,
		Mat_<double>& R2,
		Mat_<double>& t1,
		Mat_<double>& t2);

	bool FindCameraMatrices(
		vector<KeyPoint> featurePoints1,
		vector<KeyPoint> featurePoints2,
		Mat F,
		Matx34d& P,
		Matx34d& P1,
		vector<SpacePoint>& outCloud
		);

	vector<SpacePoint> triangulation(vector<KeyPoint> keypoints1,
		vector<KeyPoint> keypoints2,
		Mat &K,
		Matx34d &P,
		Matx34d & P1,
		vector<SpacePoint> pointCloud);


	Mat_<double> IterativeTriangulation(Point3d u,	//homogenous image point (u,v,1)
		Matx34d P,			//camera 1 matrix
		Point3d u1,			//homogenous image point in 2nd camera
		Matx34d P1			//camera 2 matrix
		);

	Mat_<double> LinearLSTriangulation(Point3d u,		//homogenous image point (u,v,1)
		Matx34d P,		//camera 1 matrix
		Point3d u1,		//homogenous image point in 2nd camera
		Matx34d P1		//camera 2 matrix
		);

	
};