#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#ifndef EXAMPLE_H
#define EXAMPLE_H
#include "params.hpp"
#endif
#include <iostream>
#include <string>
#include <vector>
using namespace std;
using namespace cv;
// Function prototypes
static Mat norm0_255(InputArray _src);
static string parserImageName(string path);
Mat loadInImages(string);
Mat prepareImage(Mat frame);
void detectFace(vector<Rect> &faces, Mat frame_gray);
void displayFacesList(vector<Rect> faces, Mat frame);
void displayMarkedImage(vector<Rect> faces, Mat frame);
void saveFaces(String path, vector<Rect> faces, Mat frame);
