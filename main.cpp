#include "faceDetection.hpp"


int main(int argc, char const *argv[]) {
  // string path = "sampleInput/obama/obama_1.jpg";
  if (argc < 2) {
    cout << "Use: ./a.out [ImagePath]" << endl;
    exit(1);
  }
  string path = argv[1];
  Mat image = loadInImages(path);
  Mat image_gray;
  vector<Rect> faces;
  if (!image.empty()) {
    image_gray = prepareImage(image);
    detectFace(faces, image_gray);
    displayFacesList(faces, image_gray);
    displayMarkedImage(faces, image);
    saveFaces(path, faces, image_gray);
  }
  if(faces.size() > 0) {
    cout << "---- Start recognizing faces -----" << endl;
  }
  cout << "Press any key to exit. " << endl;
  waitKey(0);
  return 0;
}
