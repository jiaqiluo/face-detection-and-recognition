// Include Standard C/C++ Header Files Here
#include <stdio.h>
#include <string.h>

// Include OpenCV Header Files Here
// #include <opencv/cv.h>
// #include <opencv2/cvaux.h>
// #include <opencv2/highgui.h>
using namespace std;
using namespace cv;

class pcamah : public cv::FaceRecognizer {
private:
  // the number of training images
  int nTrainFaces;
  // number of eigenvalues
  int nEigens;
  // array of face images
  IplImage **faceImgArr;
  // array of person numbers
  Mat *personNumTruthMat;
  // the average image
  IplImage *pAvgTrainImg;
  // Eigen vectors
  IplImage **eigenVectArr;
  // OpenCV's matrix datatype : matrix of the eigenvalues
  Mat *eigenValMat;
  // OpenCV's matrix datatype :Matrix of the projected training faces
  Mat *projectedTrainFaceMat;
  // result counters.
  int correctAcceptance;
  int falseAcceptance;

public:
  using cv::FaceRecognizer::save;
  using cv::FaceRecognizer::load;
  pcamah::pcamah(int nTrainFaces, int nEigens, IplImage **faceImgArr, Mat personNumTruthMat, IplImage **pAvgTrainImg, IplImage *eigenVectArr, Mat *eigenValMat, Mat *projectedTrainFaceMat, int correctAcceptance, int falseAcceptance): nTrainFaces(0), nEigens(0), faceImgArr(0), personNumTruthMat(0), pAvgTrainImg(0), eigenVectArr(0), eigenValMat(0), projectedTrainFaceMat(0), correctAcceptance(0), falseAcceptance(0){}
  void train(cv::InputArrayOfArrays src, cv::InputArray labels);

  // Predicts the label of a query image in src.
  int predict(cv::InputArray src) const;

  // Predicts the label and confidence for a given sample.
  void predict(cv::InputArray _src, int &label, double &dist) const;

  // see FaceRecognizer::load.
  void load(const cv::FileStorage &fs);

  // See FaceRecognizer::save.
  void save(cv::FileStorage &fs) const;

  cv::AlgorithmInfo *info() const;
};
