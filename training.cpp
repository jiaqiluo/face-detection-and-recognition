#include "opencv2/contrib/contrib.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
using namespace std;
using namespace cv;

// prototypes
static void read_csv(const string&, vector<Mat>&, vector<int>&, char);
void eigenFaceRecognization(vector<Mat>&, vector<int>&, Mat&, int&);
void fisherFaceRecognization(vector<Mat>&, vector<int>&, Mat&, int&);
void LBPHFaceRecognization(vector<Mat>&, vector<int>&, Mat&, int&);


static void read_csv(const string &filename, vector<Mat> &images,
                     vector<int> &labels, char separator = ';') {
  ifstream file(filename.c_str(), ifstream::in);
  if (!file) {
    string error_message =
        "No valid input file was given, please check the given filename.";
    CV_Error(CV_StsBadArg, error_message);
  }

  string line, path, classlabel;
  while (getline(file, line)) {
    stringstream liness(line);
    getline(liness, path, separator);
    getline(liness, classlabel);
    if (!path.empty() && !classlabel.empty()) {
      images.push_back(imread(path, 0));
      labels.push_back(atoi(classlabel.c_str()));
    }
  }
  return;
}

void eigenFaceRecognization(std::vector<Mat> &trainImages,
                            std::vector<int> &trainLabels, Mat &testImage,
                            int &testLabel) {
  Ptr<FaceRecognizer> model = createEigenFaceRecognizer();
  model->train(trainImages, trainLabels);
  int predictedLabel = model->predict(testImage);
  string result_message = format("Predicted class = %d / Actual class = %d.",
                                 predictedLabel, testLabel);
  cout << result_message << endl;
  return;
}

void fisherFaceRecognization(std::vector<Mat> &trainImages,
                            std::vector<int> &trainLabels, Mat &testImage,
                            int &testLabel) {
  Ptr<FaceRecognizer> model = createFisherFaceRecognizer();
  model->train(trainImages, trainLabels);
  int predictedLabel = model->predict(testImage);
  string result_message = format("Predicted class = %d / Actual class = %d.",
                                 predictedLabel, testLabel);
  cout << result_message << endl;
  return;
}

void LBPHFaceRecognization(std::vector<Mat> &trainImages,
                            std::vector<int> &trainLabels, Mat &testImage,
                            int &testLabel) {
  Ptr<FaceRecognizer> model = createLBPHFaceRecognizer();
  model->train(trainImages, trainLabels);
  int predictedLabel = model->predict(testImage);
  string result_message = format("Predicted class = %d / Actual class = %d.",
                                 predictedLabel, testLabel);
  cout << result_message << endl;
  return;
}

int main(int argc, char const *argv[]) {
  vector<Mat> images;
  vector<int> labels;
  // string filename = "trainSource.csv";
  string filename = "ob_train.csv";
  read_csv(filename, images, labels);
  // Quit if there are not enough images for this demo.
  if (images.size() <= 1) {
    string error_message = "This demo needs at least 2 images to work. Please "
                           "add more images to your data set!";
    CV_Error(CV_StsError, error_message);
  }
  for (int i = 0; i < labels.size(); i++)
    cout << labels[i] << endl;
  cout << "number of Labels " << labels.size() << endl;
  cout << "number of Images " << images.size() << endl;
  cout << "image size " << images[2].size() << endl;

  // The following lines simply get the last images from your dataset and
  // remove it from the vector.
  Mat testImage = images[images.size() - 1];
  int testLabel = labels[labels.size() - 1];
  images.pop_back();
  labels.pop_back();

  Mat temp = imread("0.jpg", 0);
  int templable = 9;
  cout << "-- eigenFaceRecognization --" << endl;
  eigenFaceRecognization(images, labels, temp, templable);
  // cout << "-- fisherFaceRecognization --" << endl;
  // fisherFaceRecognization(images, labels, testImage, testLabel);
  // cout << "-- LBPHFaceRecognization --" << endl;
  // LBPHFaceRecognization(images, labels, testImage, testLabel);
  return 0;
}
