#include "training.hpp"

static void read_csv(const string &filename, vector<Mat> &images,
                     vector<int> &labels, char separator = ';') {
  ifstream file(filename.c_str(), ifstream::in);
  if (!file) {
    string error_message =
        "No valid input file was given, please check the given filename.";
    CV_Error(CV_StsBadArg, error_message);
  }
  string line;
  string path;
  string classlabel;
  while (getline(file, line)) {
    stringstream temp(line);
    getline(temp, path, separator);
    getline(temp, classlabel);
    if (!path.empty() && !classlabel.empty()) {
      images.push_back(imread(path, 0));
      labels.push_back(atoi(classlabel.c_str()));
    }
  }
  if (images.size() <= 1) {
    string error_message = "It needs at least 2 images to work. Please add "
                           "more images to your data set!";
    CV_Error(CV_StsError, error_message);
  }
  cout << "---- Loading csv file ----" << endl;
  cout << "label vaules:" << endl;
  for (int i = 0; i < labels.size(); i++)
    cout << labels[i] << ", ";
  cout << "\nSummary:" << endl;
  cout << "  number of Labels " << labels.size() << endl;
  cout << "  number of Images " << images.size() << endl;
  // cout << "  image size " << images[1].size() << endl;
  cout << "image size: " << endl;
  for (int i = 0; i < labels.size(); i++)
    cout << images[i].size() << ", ";
  cout << "Loading Success\n" << endl;
  return;
}

// This function uses input images and labels to train eigenFaceRecognization
// and saves the tranning result into a xml file
int eigenTraining(vector<Mat> &trainImages, vector<int> &trainLabels) {
  if (trainImages.size() == 0 || trainLabels.size() == 0) {
    cout << "--(eigenTraining) Error: the training data is empty." << endl;
    return -1;
  }
  Ptr<FaceRecognizer> model = createEigenFaceRecognizer(
      params::eigenFace::numComponents, params::eigenFace::threshold);
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
  Ptr<FaceRecognizer> model = createFisherFaceRecognizer(
      params::fisherFace::numComponents, params::fisherFace::threshold);
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
int lbphTraining(vector<Mat> &trainImages, vector<int> &trainLabels) {
  if (trainImages.size() == 0 || trainLabels.size() == 0) {
    cout << "--(LBPHTraining) Error: the training data is empty." << endl;
    return -1;
  }
  Ptr<FaceRecognizer> model = createLBPHFaceRecognizer(
      params::lbphFace::radius, params::lbphFace::neighbors,
      params::lbphFace::gridX, params::lbphFace::gridY,
      params::lbphFace::threshold);
  model->train(trainImages, trainLabels);
  model->save("trainingResult/LBPHTrained.xml");
  return 1;
}

// this function uses the training result to predict whethe the input
// testImage is the same person,
// outout: an integer for predicted lable
int lbphFaceRecognization(Mat &testImage) {
  Ptr<FaceRecognizer> model = createLBPHFaceRecognizer();
  model->load("trainingResult/LBPHTrained.xml");
  int predictedLabel = model->predict(testImage);
  return predictedLabel;
}

int lidTraining(vector<Mat> &trainImages, vector<int> &trainLabels) {
  if (trainImages.size() == 0 || trainLabels.size() == 0) {
    cout << "--(LBPHTraining) Error: the training data is empty." << endl;
    return -1;
  }
  cv::Ptr<cv::FaceRecognizer> model = createLidFaceRecognizer();
  model->train(trainImages, trainLabels);
  model->save("trainingResult/LIDTrained.xml");
  // cout << "LIDTraining works" << endl;
  return 1;
}

int lidFaceRecognization(Mat &testImage) {
  Ptr<FaceRecognizer> model = createLidFaceRecognizer();
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
  if (lbphTraining(images, labels) == 1) {
    cout << "testLabel: " << testLabel << endl;
    cout << lbphFaceRecognization(testImage) << endl;
  }

  cout << "-- LIDFaceRecognization --" << endl;
  cv::Ptr<cv::FaceRecognizer> model = createLidFaceRecognizer();
  model->train(images, labels);
  cout << "testLabel: " << testLabel << endl;
  cout << model->predict(testImage) << endl;
  return 0;
}
