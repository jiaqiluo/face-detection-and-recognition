// #include "opencv2/contrib/contrib.hpp"
// #include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include "lid.hpp"
// using namespace std;
// using namespace cv;

// prototypes
static void read_csv(const string &, vector<Mat> &, vector<int> &, char);
int eigenTraining(vector<Mat> &trainImages, vector<int> &trainLabels);
int eigenFaceRecognization(Mat &testImage);
int fisherTraining(vector<Mat> &trainImages, vector<int> &trainLabels);
int fisherFaceRecognization(Mat &testImage);
// Face Recognition based on Local Binary Patterns
int lbphTraining(vector<Mat> &trainImages, vector<int> &trainLabels);
int lbphFaceRecognization(Mat &testImage);
int lidTraining(vector<Mat> &trainImages, vector<int> &trainLabels);
int lidFaceRecognization(Mat &testImage);
