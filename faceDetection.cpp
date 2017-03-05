#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <sys/stat.h>   // for creating directory
#include <sys/types.h>  // for creating directory
using namespace std;
using namespace cv;

// Function prototypes
Mat loadInImages(string);
Mat prepareImage(Mat frame);
void detectFace(vector<Rect> &faces, Mat frame_gray);
void displayFacesList(vector<Rect> faces, Mat frame);
void saveFacestoFiles(string path, string pattern, vector<Rect> faces,
                      Mat frame);
void displayMarkedImage(vector<Rect> faces, Mat frame);

// Global variables
string FACE_CASCADE_NAME = "trainingResult/haarcascade_frontalface_alt2.xml";

static Mat norm0_255(InputArray _src) {
  Mat src = _src.getMat();
  // Create and return normalized image:
  Mat dst;
  switch (src.channels()) {
  case 1:
    cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
    break;
  case 3:
    cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
    break;
  default:
    src.copyTo(dst);
    break;
  }
  return dst;
}

// this function loads in and normalizes the source image from the given path
// return type: Mat
//        note: the source image
Mat loadInImages(string path) {
  // Read the image file
  Mat frame = imread(path);
  Mat output;
  // Apply the classifier to the frame
  if (!frame.empty()) {
    cout << "Success: read in image\n" << endl;
    namedWindow("Source Image", WINDOW_AUTOSIZE);
    imshow("Source Image", frame);
    output = norm0_255(frame);
    namedWindow("normalized source", WINDOW_AUTOSIZE);
    imshow("normalized source", output);
  } else
    cout << " --(loadInImages) Error: No captured frame -- Break!\n" << endl;
  return output;
}

// this function converts the image to grayscale and normalize histogram
// return type: Mat
//        note: the image that is ready for detection
Mat prepareImage(Mat frame) {
  Mat frame_gray;
  if (frame.empty()) {
    cout << "--(prepareImage) Error: the Mat frame is empty." << endl;
    return frame_gray;
  }
  cvtColor(frame, frame_gray, CV_BGR2GRAY);
  equalizeHist(frame_gray, frame_gray);
  namedWindow("equalizeHist", WINDOW_AUTOSIZE);
  imshow("equalizeHist", frame_gray);
  return frame_gray;
}

// this function detects the faces in the input images
// return type: vector<Rect> (passed as argument)
//        note: the list of position of faces found in the input image
void detectFace(vector<Rect> &faces, Mat frame_gray) {
  if (frame_gray.empty()) {
    cout << "--(detectFace) Error: the Mat frame_gray is empty." << endl;
    return;
  }
  CascadeClassifier face_cascade;
  // Load the cascade
  if (!face_cascade.load(FACE_CASCADE_NAME)) {
    cout << "--(detectFace) Error: loading face cascade failes.\n" << endl;
    return;
  }
  // detect faces
  face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2,
                                0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
  cout << "the number of faces = " << faces.size() << endl;
  return;
}

void displayFacesList(vector<Rect> faces, Mat frame) {
  if (frame.empty()) {
    cout << "--(displayFacesList) Error: the Mat frame is empty." << endl;
    return;
  }
  int fsize = faces.size();
  if (fsize == 0) {
    cout << "--(displayFacesList) Error: the vector<Rect> faces is empty."
         << endl;
    return;
  }
  // show each face in the list
  for (size_t i = 0; i < fsize; i++) {
    Mat temp = Mat(frame, faces[i]);
    resize(temp, temp, Size(300, 300));
    string title = "face" + to_string(i) + ".jpg";
    namedWindow(title, WINDOW_AUTOSIZE);
    imshow(title, temp);
    // imwrite(title, temp);
  }
  return;
}

// TODO : finish this function
void saveFacestoFiles(string path, vector<Rect> faces, Mat frame) {
  if (frame.empty()) {
    cout << "--(saveFacestoFiles) Error: the Mat frame is empty." << endl;
    return;
  }
  int fsize = faces.size();
  if (fsize == 0) {
    cout << "--(saveFacestoFiles) Error: the vector<Rect> faces is empty."
         << endl;
    return;
  }
  for (size_t i = 0; i < fsize; i++) {
    Mat temp = Mat(frame, faces[i]);
    resize(temp, temp, Size(300, 300));
    string title = "test/face" + to_string(i) + ".jpg";
    imwrite(title, temp);
  }
  return;
}


void save(vector<Rect> faces,
                      Mat frame) {
  if (frame.empty()) {
    cout << "--(saveFacestoFiles) Error: the Mat frame is empty." << endl;
    return;
  }
  int fsize = faces.size();
  if (fsize == 0) {
    cout << "--(saveFacestoFiles) Error: the vector<Rect> faces is empty."
         << endl;
    return;
  }
  for (size_t i = 0; i < fsize; i++) {
    Mat temp = Mat(frame, faces[i]);
    resize(temp, temp, Size(300, 300));
    string title = "test/face" + to_string(i) + ".jpg";
    imwrite(title, temp);
  }
  return;
}

void displayMarkedImage(vector<Rect> faces, Mat frame) {
  int fsize = faces.size();
  if (frame.empty()) {
    cout << "--(displayMarkedImage) Error: the Mat frame is empty." << endl;
    return;
  }
  if (fsize == 0) {
    cout << "--(displayMarkedImage) Error: the vector<Rect> faces is empty."
         << endl;
    return;
  }
  Mat temp;
  frame.copyTo(temp);
  for (size_t i = 0; i < fsize; i++) {
    Point pt1(faces[i].x, faces[i].y);
    Point pt2(pt1.x + faces[i].width, pt1.y  + faces[i].height);
    // circle(frame, pt1, 5, Scalar(255), 2, 8, 0);
    rectangle(temp, pt1, pt2, Scalar(0, 255, 0), 2, 8, 0);
  }
  // show the result
  namedWindow("Capture - Face detection", WINDOW_AUTOSIZE);
  imshow("Capture - Face detection", temp);
}

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
    displayFacesList(faces, temp);
    displayFacesList(faces, temp);
    displayMarkedImage(faces, image);
    save(faces, temp);
    waitKey(0);
  }
  return 0;
}
