#include "opencv2/contrib/contrib.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
using namespace std;
using namespace cv;

static void read_csv(const string &filename, vector<Mat> &images,
                     vector<int> &labels, char separator = ';') {
  std::ifstream file(filename.c_str(), ifstream::in);
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
}

void EigenFaceRecognization(std::vector<Mat> &trainImages,
                            std::vector<int> &trainLabels, Mat &testImage,
                            int &testLabel) {

  // The following lines create an Eigenfaces model for
  // face recognition and train it with the images and
  // labels read from the given CSV file.
  // This here is a full PCA, if you just want to keep
  // 10 principal components (read Eigenfaces), then call
  // the factory method like this:
  //
  //      cv::createEigenFaceRecognizer(10);
  //
  // If you want to create a FaceRecognizer with a
  // confidence threshold (e.g. 123.0), call it with:
  //
  //      cv::createEigenFaceRecognizer(10, 123.0);
  //
  // If you want to use _all_ Eigenfaces and have a threshold,
  // then call the method like this:
  //
  //      cv::createEigenFaceRecognizer(0, 123.0);
  //
  Ptr<FaceRecognizer> model = createEigenFaceRecognizer();
  model->train(trainImages, trainLabels);
  // The following line predicts the label of a given
  // test image:
  int predictedLabel = model->predict(testImage);
  //
  // To get the confidence of a prediction call the model with:
  //
  //      int predictedLabel = -1;
  //      double confidence = 0.0;
  //      model->predict(testSample, predictedLabel, confidence);
  //
  string result_message = format("Predicted class = %d / Actual class = %d.",
                                 predictedLabel, testLabel);
  cout << result_message << endl;
}

int main(int argc, char const *argv[]) {
  vector<Mat> images;
  vector<int> labels;
  string filename = "trainSource.csv";
  read_csv(filename, images, labels);
  // Quit if there are not enough images for this demo.
  if (images.size() <= 1) {
    string error_message = "This demo needs at least 2 images to work. Please "
                           "add more images to your data set!";
    CV_Error(CV_StsError, error_message);
  }
  for (int i = 0; i < labels.size(); i++)
    cout << labels[1] << endl;
  cout << "Labels Size " << labels.size() << endl;
  cout << "Images Size " << images.size() << endl;

  // The following lines simply get the last images from
  // your dataset and remove it from the vector. This is
  // done, so that the training data (which we learn the
  // cv::FaceRecognizer on) and the test data we test
  // the model with, do not overlap.
  Mat testImage = images[images.size() - 1];
  int testLabel = labels[labels.size() - 1];
  images.pop_back();
  labels.pop_back();

  EigenFaceRecognization(images, labels, testImage, testLabel);
  return 0;
}
