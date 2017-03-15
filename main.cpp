#include "faceDetection.hpp"
#include "training.hpp"


int main(int argc, char const *argv[]) {
  // string path = "sampleInput/obama/obama_1.jpg";
  if (argc < 4) {
    cout << "Use: ./a.out [ImagePath] [label] [retrain:y/n]" << endl;
    exit(1);
  }
  string path = argv[1];
  int testLabel = atoi(argv[2]);
  Mat testImage = loadInImages(path);
  char temp = argv[3][0];
  bool retrain = false;
  if(temp == 'y' || temp == 'Y'){
      retrain = true;
  }
  else if (temp == 'n' || temp == 'N'){
      retrain = false;
  }
  else {
    cout << "please give y/n for the third argument" << endl;
    exit(1);
  }
  Mat image_gray;
  vector<Rect> faces;
  if (!testImage.empty()) {
    image_gray = prepareImage(testImage);
    detectFace(faces, image_gray);
    // displayFacesList(faces, image_gray);
    displayMarkedImage(faces, testImage);
    saveFaces(path, faces, image_gray);
  }
  if(faces.size() > 0) {
    cout << "---- Start recognizing faces -----" << endl;
    run(image_gray, testLabel, retrain);
  }
  cout << "Press any key to exit. " << endl;
  waitKey(0);
  return 0;
}
