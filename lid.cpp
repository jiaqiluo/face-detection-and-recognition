#include "lid.hpp"
// Generates the histograms for each label and puts them in the hists vector
void generateHistograms(vector<Mat> &hists, const vector<Mat> &separatedLabels,
                        int clusterCount) {
  // Histogram paramaters
  const int nimages = 1;      // Only 1 image (The labels)
  const int channels[] = {0}; // Use the 0 index channel (none)
  const int dims = 1;         // Only 1 channel
  const int histSize[] = {
      clusterCount}; // The number of bins is the number of clusters
  const float hranges[] = {
      0, clusterCount}; // Cluster group varies from 0 to the number of clusters
  const float *ranges[] = {hranges};
  // For each image, calculate its histogram
  for (unsigned int i = 0; i < separatedLabels.size(); ++i) {
    calcHist(&separatedLabels[i], nimages, channels,
             Mat(), // Do not use a mask
             hists[i], dims, histSize, ranges,
             true,   // The histogram is uniform
             false); // Do not accumulate
  }
}

void normalizeHistograms(vector<Mat> &hists) {
  Mat normalizedMat;
  for (vector<Mat>::const_iterator it = hists.begin(); it != hists.end();
       ++it) {
    normalize(*it, normalizedMat);
    normalizedMat.copyTo(*it);
  }
}

size_t getSize(InputArrayOfArrays src) {
  vector<Mat> images;
  src.getMatVector(images);
  return images.size();
}

// Populates allKeyPoints and descriptors
void Lidfaces::detectKeypointsAndDescriptors(
    InputArrayOfArrays src, vector<vector<KeyPoint> > &allKeyPoints,
    Mat &descriptors) const {

  SIFT detector(params::sift::nfeatures, params::sift::nOctaveLayers,
                params::sift::contrastThreshold, params::sift::edgeThreshold,
                params::sift::sigma);
  vector<Mat> images;
  src.getMatVector(images);
  for (unsigned int i = 0; i < images.size(); ++i) {
    // Determine the SIFT keypoints (but discard SIFT descriptors)
    vector<KeyPoint> keyPointsForCurrentImage;
    detector(images[i], // Get keypoints in the current image
             noArray(), // No mask
             keyPointsForCurrentImage,
             noArray()); // We don't care about the SIFT descriptors (We
                         // will use LID descriptors)
    allKeyPoints.push_back(keyPointsForCurrentImage);
    // For each of the keypoints, calculate the LID descriptor
    Mat singleImgDescriptors;
    for (size_t j = 0; j < keyPointsForCurrentImage.size(); ++j)
      descriptors.push_back(
          lid(images[i], keyPointsForCurrentImage[j].pt, mInradius));
  }
  // Ensure that each descriptors size is the number of neighbors
  assert(descriptors.cols = 8 * mInradius);
}

Ptr<FaceRecognizer> createLidFaceRecognizer(int inradius, double threshold) {
  return Ptr<Lidfaces>(
      new Lidfaces(inradius, threshold)); // This is equivalent to makePtr
}

// Returns the LID descriptor of mat about p
// For an image I : Z^2 -> R
// lid(I, p) = [d(p1, p), ..., d(pn, p)]
// where d(pi, p) = I(pi) - I(p)
Mat lid(const Mat &src, Point p, int inradius) {
  assert(src.type() == CV_8UC1);
  assert(inradius >= 1);
  // For illustration, if p is the point and X are the neighbors of inradius=2
  // (N=16) and o are ignored points
  // XXXXX
  // XoooX
  // XopoX
  // XoooX
  // XXXXX
  int totalNeighbors = 8 * inradius; // This is the formula for the perimeter of
                                     // a square given the inradius
  Mat lidDescriptor(1, totalNeighbors, CV_8UC1);
  // Calculate the real bounds (making sure not to go off the end of the image)
  // These are the bounds of the square with appropriate inradius centred about
  // p
  const int MIN_X = max(p.x - inradius, 0);
  const int MAX_X = min(p.x + inradius, src.cols);
  const int MIN_Y = max(p.y - inradius, 0);
  const int MAX_Y = min(p.y + inradius, src.cols);
  const unsigned char centerIntensity = src.at<unsigned char>(p.y, p.x);
  // neighborIndex is i where p_i is the ith neighbor
  // It goes from 0 to totalNeighbors-1
  int neighborIndex = 0;
  // For each pixel in the square perimeter (going clockwise from the top right)
  // Set the nth descriptor element
  // Top (left to rigth)
  for (int x = MIN_X; x < MAX_X; ++x) {
    lidDescriptor.at<unsigned char>(neighborIndex++) =
        max(src.at<unsigned char>(MIN_Y, x) - centerIntensity, 0);
  }
  // Right (top to bottom)
  for (int y = MIN_Y; y < MAX_Y; ++y) {
    lidDescriptor.at<unsigned char>(neighborIndex++) =
        max(src.at<unsigned char>(y, MAX_X) - centerIntensity, 0);
  }
  // Bottom (right to left)
  for (int x = MAX_X; x > MIN_X; --x) {
    // Set the nth descriptor element
    lidDescriptor.at<unsigned char>(neighborIndex++) = max(
        src.at<unsigned char>(MIN_Y, x) - src.at<unsigned char>(p.y, p.x), 0);
  }
  // Left (bottom to top)
  for (int y = MAX_Y; y > MIN_Y; --y) {
    // Set the nth descriptor element
    lidDescriptor.at<unsigned char>(neighborIndex++) = max(
        src.at<unsigned char>(y, MIN_X) - src.at<unsigned char>(p.y, p.x), 0);
  }
  return lidDescriptor;
}

// Computes an Lidfaces model with images in src and corresponding labels
// in labels.
void Lidfaces::train(InputArrayOfArrays src, InputArray labels) {
  vector<vector<KeyPoint> > allKeyPoints;
  Mat descriptors;
  // Get SIFT keypoints and LID descriptors
  detectKeypointsAndDescriptors(src, allKeyPoints, descriptors);
  // kmeans function requires points to be CV_32F
  descriptors.convertTo(descriptors, CV_32FC1);
  // Do k-means clustering
  const int CLUSTER_COUNT =
      params::lidFace::clustersAsPercentageOfKeypoints * descriptors.rows;
  Mat histogramLabels;
  // This function populates histogram bin labels
  // The nth element of histogramLabels is an integer which represents the
  // cluster that the
  // nth element of allKeyPoints is a member of.
  kmeans(descriptors,     // The points we are clustering are the descriptors
         CLUSTER_COUNT,   // The number of clusters (K)
         histogramLabels, // The label of the corresponding keypoint
         params::kmeans::termCriteria, params::kmeans::attempts,
         params::kmeans::flags, mCenters);
  // Convert to single channel 32 bit float as the matrix needs to be in a form
  // supported
  // by calcHist
  histogramLabels.convertTo(histogramLabels, CV_32FC1);
  // We end up with a histogram for each image
  const size_t NUM_IMAGES = getSize(src);
  vector<Mat> hists(NUM_IMAGES);
  // mCodebook.resize(NUM_IMAGES);
  // The histogramLabels vector contains ALL the points from EVERY image. We
  // need to split
  // it up into groups of points for each image.
  // Because there are the same number of points in each image, and the points
  // were put
  // into the labels vector in order, we can simply divide the labels vector
  // evenly to get
  // the individual image's points.
  vector<Mat> separatedLabels;
  for (unsigned int i = 0, startRow = 0; i < NUM_IMAGES; ++i) {
    separatedLabels.push_back(
        histogramLabels.rowRange(startRow, startRow + allKeyPoints[i].size()));
    startRow += allKeyPoints[i].size();
  }
  // Populate the hists vector
  generateHistograms(hists, separatedLabels, CLUSTER_COUNT);
  // Make the magnitude of each histogram equal to 1
  normalizeHistograms(hists);
  mCodebook = hists;
  mLabels = labels.getMat();
}

// Predicts the label of a query image in src by creating a histogram by
// clustering the sift
// descriptors using the centres we generated in training. The distances between
// the histogram and
// the histogram of every training image is calculated. The smallest average
// distance to a class of
// images is used to classify the image.
int Lidfaces::predict(InputArray src) const {
  int label;
  double dummy;
  predict(src, label, dummy);
  return label;
}

// Predicts the label and confidence for a given sample.
void Lidfaces::predict(InputArray src, int &label, double &dist) const {
  label = -1;
  dist = DBL_MAX;
  vector<vector<KeyPoint> > keyPoints;
  Mat descriptors;
  vector<Mat> imageVector; // A vector containing just one image (this
                           // is so we can use the same
                           // detectKeypointsAndDescriptors function
  imageVector.push_back(src.getMat());
  // Get SIFT keypoints and LID descriptors
  detectKeypointsAndDescriptors(imageVector, keyPoints, descriptors);
  // Cluster the image using the training centres
  int closestCentroidIndex = 0;
  mCenters.convertTo(mCenters, CV_32FC1);
  descriptors.convertTo(descriptors, CV_32FC1);
  Mat histogramLabels(descriptors.rows, 1, CV_32F);
  for (int descriptorIndex = 0; descriptorIndex < descriptors.rows;
       ++descriptorIndex) {
    // (Give it a classification)
    double smallestDist = DBL_MAX;
    // For each centroid
    for (int centroidIndex = 0; centroidIndex < mCenters.rows;
         ++centroidIndex) {
      // Calculate the distance from the descriptor to the centroid
      double currentDist =
          norm(descriptors.row(descriptorIndex) - mCenters.row(centroidIndex));
      // If it is the smallest distance, remember it and the centroid
      if (currentDist < smallestDist) {
        smallestDist = currentDist;
        closestCentroidIndex = centroidIndex;
      }
    }
    histogramLabels.at<float>(descriptorIndex) = closestCentroidIndex;
  }
  assert(histogramLabels.rows == descriptors.rows);
  vector<Mat> separatedLabels;
  vector<Mat> hists(1);
  separatedLabels.push_back(histogramLabels);
  generateHistograms(hists, separatedLabels, mCenters.rows);
  normalizeHistograms(hists);
  dist = DBL_MAX;
  multimap<int, double> distances; // Maps label to distance
  // Compare this histogram against all the other histograms
  for (size_t codebookIndex = 0; codebookIndex < mCodebook.size();
       ++codebookIndex) {
    // Get dist hist
    double currentDist =
        compareHist(hists[0], mCodebook[codebookIndex], CV_COMP_CHISQR);
    distances.insert(
        pair<int, double>(mLabels.at<int>(codebookIndex), currentDist));
  }
  // Calculate the smallest average distance
  double smallestAverageDist = DBL_MAX;
  int closestLabel = -1;
  double curDist = 0;
  for (int curLabel = 0; curLabel < mCenters.rows; ++curLabel) {
    if (distances.count(curLabel) ==
        0)      // If this histogram has none of this label
      continue; // Don't bother calculating it
    double totalDist = 0;
    pair<multimap<int, double>::const_iterator,
         multimap<int, double>::const_iterator>
        itRange = distances.equal_range(curLabel);
    for (multimap<int, double>::const_iterator it = itRange.first;
         it != itRange.second; ++it) {
      totalDist += it->second;
    }
    curDist = totalDist / distances.count(curLabel);
    if (curDist < smallestAverageDist) {
      smallestAverageDist = curDist;
      closestLabel = curLabel;
    }
  }
  label = closestLabel;
  dist = smallestAverageDist;
}

void Lidfaces::load(const FileStorage &fs) {
  // Read matrices
  fs["inradius"] >> mInradius;
  fs["numNeighbors"] >> mNumNeighbors;
  fs["threshold"] >> mThreshold;
  // read sequences
  // Read the codebook
  // readFileNodeList(fs["projections"], _projections);
  const FileNode &fn = fs["codebook"];
  if (fn.type() == FileNode::SEQ) {
    for (FileNodeIterator it = fn.begin(); it != fn.end();) {
      Mat item;
      it >> item;
      mCodebook.push_back(item);
    }
  }
  fs["labels"] >> mLabels;
  fs["centers"] >> mCenters;
}

void Lidfaces::save(FileStorage &fs) const {
  // Write matrices
  fs << "inradius" << mInradius;
  fs << "numNeighbors" << mNumNeighbors;
  fs << "threshold" << mThreshold;
  // Write sequences
  // Write the codebook
  // writeFileNodeList(fs, "codebook", mCodebook);
  fs << "codebook"
     << "[";
  for (vector<Mat>::const_iterator it = mCodebook.begin();
       it != mCodebook.end(); ++it) {
    fs << *it;
  }
  fs << "]";

  fs << "labels" << mLabels;
  fs << "centers" << mCenters;
}

AlgorithmInfo *Lidfaces::info() const { return NULL; }
