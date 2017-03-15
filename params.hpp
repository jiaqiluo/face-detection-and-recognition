namespace params {
  namespace detection {
    const std::string FACE_CASCADE_NAME = "trainingResult/haarcascade_frontalface_alt2.xml";
  }
  namespace cascadeClassifier {
  // Parameters:
  // scaleFactor - specifying how much the image size is reduced at each image scale.
  // minNeighbors - specifying how many neighbors each candidate rectangle should have to retain it.
  // flags - This is a legacy parameter and it will not do anything with our application
  // minSize, maxSize - Chose not to have a min/max size so that we could handle close/far faces. Note: In OpenCV 2.4.2, this parameter does not work for haar
    const double scaleFactor = 1.1;
    const int minNeighbors = 2;
    const int flags = 0;
    const cv::Size minSize;
    const cv::Size maxSize;
  }
  namespace eigenFace {
    // Parameters:
    // num_components – The number of components (read: Eigenfaces) kept for this
    // Prinicpal Component Analysis. As a hint: There’s no rule how many components
    // (read: Eigenfaces) should be kept for good reconstruction capabilities. It is
    // based on your input data, so experiment with the number. Keeping 80 components
    // should almost always be sufficient.
    // threshold – The threshold applied in the prediciton.
    const int numComponents = 80;
    const double threshold = 4710.0f;
  }
  namespace fisherFace {
    // Parameters:
    // num_components – The number of components (read: Fisherfaces) kept for
    // this Linear Discriminant Analysis with the Fisherfaces criterion. It’s useful to
    // keep all components, that means the number of your classes c (read: subjects,
    // persons you want to recognize). If you leave this at the default (0) or
    // set it to a value less-equal 0 or greater (c-1), it will be set to
    // the correct number (c-1) automatically.
    // threshold – The threshold applied in the prediction. If the distance to the nearest neighbor is larger than the threshold, this method returns -1.
    const int numComponents = 1; // OpenCV will set numComponents to {classes}-1
    const double threshold = DBL_MAX;
  }
  namespace lbphFace {
    // Parameters:
    // radius – The radius used for building the Circular Local Binary Pattern.
    // neighbors – The number of sample points to build a Circular Local Binary Pattern
    // from. An appropriate value is to use 8 sample points.
    // Keep in mind: the more sample points you include, the higher the computational cost.
    // grid_x – The number of cells in the horizontal direction, 8 is a common value
    // used in publications. The more cells, the finer the grid, the higher the
    // dimensionality of the resulting feature vector.
    // grid_y – The number of cells in the vertical direction, 8 is a common value
    // used in publications. The more cells, the finer the grid, the higher the
    // dimensionality of the resulting feature vector.
    // threshold – The threshold applied in the prediction. If the distance to
    // the nearest neighbor is larger than the threshold, this method returns -1.
    const int radius = 1;
    const int neighbors = 8;
    const int gridX = 8;
    const int gridY = 8;
    const double threshold = 52.0f;
  }
  namespace lidFace {
  const int inradius = 1;
  const double threshold = DBL_MAX;
  const double clustersAsPercentageOfKeypoints = 0.9f;
  }
  namespace sift {
    // There is a bug in 2.4.2 where this parameter does not work, so I have to
    // leave it at 0
    const int nfeatures = 0;               // Default 0
    const int nOctaveLayers = 5;           // Default 3
    const double contrastThreshold = 0.06; // Default 0.04
    const double edgeThreshold = 10;       // Default 10
    const double sigma = 3;                // Default 1.6
  }
  namespace kmeans {
    const cv::TermCriteria
        termCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 1000, 0.01);
    const int attempts = 5;
    const int flags = cv::KMEANS_PP_CENTERS;
    // K (the number of clusters) is not specified here as it is determined by the
    // number of keypoints (and not a compile-time constant)
  }
}
