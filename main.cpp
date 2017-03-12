#include "faceDetection.hpp"


int main(int argc, char const *argv[]) {
  // string path = "sampleInput/obama/obama_1.jpg";
  if (argc < 2) {
    cout << "Use: ./a.out [ImagePath]" << endl;
    exit(1);
  }
  string path = argv[1];
  Mat image = loadInImages(path);
  Mat temp;
  vector<Rect> faces;
  if (!image.empty()) {
    temp = prepareImage(image);
    detectFace(faces, temp);
    // displayFacesList(faces, temp);
    displayMarkedImage(faces, image);
    saveFaces(path, faces, temp);
    waitKey(0);
  }
  return 0;
}
