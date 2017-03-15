#include "pcamah.hpp"
void pcamah::train(cv::InputArrayOfArrays src, cv::InputArray labels){
  return;
}

// Predicts the label of a query image in src.
int pcamah::predict(cv::InputArray src) const {
  return 1;
}

// Predicts the label and confidence for a given sample.
void pcamah::predict(cv::InputArray _src, int &label, double &dist) const {
  return;
}

// see FaceRecognizer::load.
void pcamah::load(const cv::FileStorage &fs) {
  return;
}

// See FaceRecognizer::save.
void pcamah::save(cv::FileStorage &fs) const {
  return;
}

AlgorithmInfo *pcamah::info() const { return NULL; }


int main(int argc, char const *argv[]) {
  return 0;
}
