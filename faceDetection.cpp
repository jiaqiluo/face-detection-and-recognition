#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include <iostream>
#include <string>

using namespace std;
using namespace cv;

// Function prototypes
Mat loadInImages(string);
void detectAndDisplay(Mat &);

// Global variables
string face_cascade_name = "haarcascade_frontalface_alt.xml";

static Mat norm_0_255(InputArray _src) {
    Mat src = _src.getMat();
    // Create and return normalized image:
    Mat dst;
    switch(src.channels()) {
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

Mat loadInImages(string path) {

  // Read the image file
  Mat frame = imread(path);
  Mat output;
  // Apply the classifier to the frame
  if (!frame.empty()) {
    cout << "Success: read in image\n" << endl;
    namedWindow("Source Image", WINDOW_AUTOSIZE);
    imshow("Source Image", frame);
    output = norm_0_255(frame);
  } else
    cout << " --(!) No captured frame -- Break!\n" << endl;
  return output;
}

void detectAndDisplay(Mat &frame) {
  CascadeClassifier face_cascade;
  vector<Rect> faces;
  Mat frame_gray;

  // convert the image to grayscale and normalize histogram:
  cvtColor(frame, frame_gray, CV_BGR2GRAY);
  equalizeHist(frame_gray, frame_gray);
  namedWindow("equalizeHist", WINDOW_AUTOSIZE);
  imshow("equalizeHist", frame_gray);

  // Load the cascade
  if (!face_cascade.load(face_cascade_name)) {
    cout << "--(!)Error loading\n" << endl;
    return;
  }
  // detect faces
  face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2,
                                0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
  cout << "the number of faces = " << faces.size() << endl;
  // draw a circle around the face
  // for (size_t i = 0; i < faces.size(); i++) {
  //   Point center(faces[i].x + faces[i].width * 0.5,
  //                faces[i].y + faces[i].height * 0.5);
  //   ellipse(frame, center, Size(faces[i].width * 0.5, faces[i].height * 0.5), 0,
  //           0, 360, Scalar(255, 0, 255), 4, 8, 0);
  // }
  for (size_t i = 0; i < faces.size(); i++) {
    // show each face in the list
    Mat temp = Mat(frame_gray, faces[i]);
    resize(temp, temp, Size(300,300));
    string title = "face"+ to_string(i) + ".jpg";
    namedWindow(title, WINDOW_AUTOSIZE);
    imshow(title, temp);
    imwrite(title, temp);
  }
  for (size_t i = 0; i < faces.size(); i++) {
    Point pt1(faces[i].x , faces[i].y);
    Point pt2(pt1.x + 150, pt1.y + 200);
    circle(frame, pt1, 5, Scalar(255), 2, 8, 0);
    rectangle(frame, pt1, pt2, Scalar(0, 255, 0), 2, 8, 0);
  }
  // show the result
  namedWindow("Capture - Face detection", WINDOW_AUTOSIZE);
  imshow("Capture - Face detection", frame);
  waitKey(0);
}

int main(int argc, char const *argv[]) {
  // string path = "sampleInput/obama/obama_1.jpg";
  if(argc <2) {
    cout << "Use: ./a.out [ImagePath]" << endl;
    exit(1);
  }
  string path = argv[1];
  Mat image = loadInImages(path);
  if (!image.empty())
    detectAndDisplay(image);
  return 0;
}
