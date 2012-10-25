#include <sstream>  // string to number conversion
#include "createModel.h"

void insert_header(int pointCloudSize, int index){

	string number = static_cast<ostringstream*>(&(ostringstream() << index)) ->str();
	string name = "out" + number + ".ply";
	
	ofstream outfile (name);
	if (outfile.is_open())
	{
		outfile << "ply\n";
		outfile << "format ascii 1.0\n";
		outfile << "element face 0";
		outfile << "property list uchar int vertex_indices\n";
		outfile << "element vertex " << pointCloudSize + 1 << "\n";
		outfile << "property float x\n";
		outfile << "property float y\n";
		outfile << "property float z\n";
		outfile << "property uchar diffuse_red\n";
		outfile << "property uchar diffuse_green\n";
		outfile << "property uchar diffuse_blue\n";
		outfile << "end_header\n";

		outfile << "0 0 0 255 0 0\n";

		outfile.close();
	}
	else cout << "Unable to open file";



}


void insert_point
	(double x, double y, double z, int b,
	int g, int r, int index){

		string number = static_cast<ostringstream*>(&(ostringstream() << index)) ->str();
		string name = "out" + number + ".ply";

		ofstream outfile;
		outfile.open(name,ios::app);
		outfile << x << " ";
		outfile << y << " ";
		outfile << z << " ";
		outfile << b << " ";
		outfile << g << " ";
		outfile << r << endl;
		outfile.close();

		//cout << x << " " << y << " " << z << " ";

}

