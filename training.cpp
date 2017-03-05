#include "opencv2/contrib/contrib.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
using namespace std;
using namespace cv;

// prototypes
static void read_csv(const string &, vector<Mat> &, vector<int> &, char);
int eigenTraining(vector<Mat> &trainImages, vector<int> &trainLabels);
int eigenFaceRecognization(Mat &testImage);
int fisherTraining(vector<Mat> &trainImages, vector<int> &trainLabels);
int fisherFaceRecognization(Mat &testImage);
// Face Recognition based on Local Binary Patterns
int LBPHTraining(vector<Mat> &trainImages, vector<int> &trainLabels);
int LBPHFaceRecognization(Mat &testImage);


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
  if (images.size() <= 1) {
    string error_message = "This demo needs at least 2 images to work. Please "
                           "add more images to your data set!";
    CV_Error(CV_StsError, error_message);
  }
  cout << "---- Loading csv file ----" << endl;
  cout << "label vaules:" << endl;
  for (int i = 0; i < labels.size(); i++)
    cout << labels[i] << ", ";
  cout << "\nSummary:" << endl;
  cout << "  number of Labels " << labels.size() << endl;
  cout << "  number of Images " << images.size() << endl;
  cout << "  image size " << images[1].size() << endl;
  cout << "---- Loading Success ----" << endl;
  return;
}

// This function uses input images and labels to train eigenFaceRecognization
// and saves the tranning result into a xml file
int eigenTraining(vector<Mat> &trainImages, vector<int> &trainLabels) {
  if (trainImages.size() == 0 || trainLabels.size() == 0) {
    cout << "--(eigenTraining) Error: the training data is empty." << endl;
    return -1;
  }
  Ptr<FaceRecognizer> model = createEigenFaceRecognizer();
  model->train(trainImages, trainLabels);
  model->save("trainingResult/eigenTrained.xml");
  return 1;
}

// this function uses the training result to predict whethe the input
// testImage is the same person,
// outout: an integer for predicted lable
int eigenFaceRecognization(Mat &testImage) {
  Ptr<FaceRecognizer> model = createEigenFaceRecognizer();
  model->load("trainingResult/eigenTrained.xml");
  int predictedLabel = model->predict(testImage);
  // string result_message = format("Predicted class = %d / Actual class = %d.",
  //                                predictedLabel, testLabel);
  // cout << result_message << endl;
  return predictedLabel;
}

// This function uses input images and labels to train fisherFaceRecognization
// and saves the tranning result into a xml file
int fisherTraining(vector<Mat> &trainImages, vector<int> &trainLabels) {
  if (trainImages.size() == 0 || trainLabels.size() == 0) {
    cout << "--(fisherTraining) Error: the training data is empty." << endl;
    return -1;
  }
  Ptr<FaceRecognizer> model = createFisherFaceRecognizer();
  model->train(trainImages, trainLabels);
  model->save("trainingResult/fisherTrained.xml");
  return 1;
}

// this function uses the training result to predict whethe the input
// testImage is the same person,
// outout: an integer for predicted lable
int fisherFaceRecognization(Mat &testImage) {
  Ptr<FaceRecognizer> model = createFisherFaceRecognizer();
  model->load("trainingResult/fisherTrained.xml");
  int predictedLabel = model->predict(testImage);
  return predictedLabel;
}

// This function uses input images and labels to train LBPHFaceRecognization
// and saves the tranning result into a xml file
int LBPHTraining(vector<Mat> &trainImages, vector<int> &trainLabels) {
  if (trainImages.size() == 0 || trainLabels.size() == 0) {
    cout << "--(LBPHTraining) Error: the training data is empty." << endl;
    return -1;
  }
  Ptr<FaceRecognizer> model = createLBPHFaceRecognizer();
  model->train(trainImages, trainLabels);
  model->save("trainingResult/LBPHTrained.xml");
  cout << "LBPHTrained works" << endl;
  return 1;
}

// this function uses the training result to predict whethe the input
// testImage is the same person,
// outout: an integer for predicted lable
int LBPHFaceRecognization(Mat &testImage) {
  Ptr<FaceRecognizer> model = createLBPHFaceRecognizer();
  model->load("trainingResult/LBPHTrained.xml");
  int predictedLabel = model->predict(testImage);
  return predictedLabel;
}

int main(int argc, char const *argv[]) {
  vector<Mat> images;
  vector<int> labels;
  // string filename = "trainSource.csv";
  string filename = "ob_train.csv";
  read_csv(filename, images, labels);

  // The following lines simply get the last images from your dataset and
  // remove it from the vector.
  Mat testImage = images[images.size() - 1];
  int testLabel = labels[labels.size() - 1];
  images.pop_back();
  labels.pop_back();

  cout << "-- eigenFaceRecognization --" << endl;
  if (eigenTraining(images, labels) == 1) {
    cout << "testLabel: " << testLabel << endl;
    cout << eigenFaceRecognization(testImage) << endl;
  }
  cout << "-- fisherFaceRecognization --" << endl;
  if (fisherTraining(images, labels) == 1) {
    cout << "testLabel: " << testLabel << endl;
    cout << fisherFaceRecognization(testImage) << endl;
  }
  cout << "-- LBPHFaceRecognization --" << endl;
  if (LBPHTraining(images, labels) == 1) {
    cout << "testLabel: " << testLabel << endl;
    cout << LBPHFaceRecognization(testImage) << endl;
  }
  // LBPHFaceRecognization(images, labels, testImage, testLabel);
  return 0;
}
