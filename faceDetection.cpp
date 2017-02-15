#include "faceDetection.h"

// Global variables
string face_cascade_name = "haarcascade_frontalface_alt.xml";
CascadeClassifier face_cascade;
string window_name = "Capture - Face detection";

// Function prototypes
int loadInImages(string, Mat &);
void detectAndDisplay(Mat &);

int loadInImages(string path, Mat & output) {
  // Load the cascade
  if (!face_cascade.load(face_cascade_name)) {
    printf("--(!)Error loading\n");
    return 0;
  }

  // Read the image file
  Mat frame = imread(path);
  // Apply the classifier to the frame
  if (!frame.empty()) {
    printf("Success: read in image\n");
    namedWindow( "Source Image", WINDOW_AUTOSIZE );
    imshow("Source Image", frame);
    frame.copyTo(output);  
    return 1;
  } else {
    printf(" --(!) No captured frame -- Break!");
    return 0;
  }
}

void detectAndDisplay(Mat &frame) {
  std::vector<Rect> faces;
  Mat frame_gray;

  // convert the image to grayscale and normalize histogram:
  cvtColor(frame, frame_gray, CV_BGR2GRAY);
  equalizeHist(frame_gray, frame_gray);

  // detect faces
  face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2,
                                0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
  // draw a circle around the face
  for (size_t i = 0; i < faces.size(); i++) {
    Point center(faces[i].x + faces[i].width * 0.5,
                 faces[i].y + faces[i].height * 0.5);
    ellipse(frame, center, Size(faces[i].width * 0.5, faces[i].height * 0.5), 0,
            0, 360, Scalar(255, 0, 255), 4, 8, 0);
  }
  // show the result
  namedWindow("Capture - Face detection", WINDOW_AUTOSIZE);
  imshow("Capture - Face detection", frame);
  waitKey(0);
}

int main(int argc, char const *argv[]) {
  string path = "sampleInput/obama_1.jpg";
  Mat image;
  loadInImages(path, image);
  if (!image.empty()) detectAndDisplay(image);
  return 0;
}
