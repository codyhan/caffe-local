#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "caffe/caffe_lstm.hpp"

#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0]
              << " deploy.prototxt network.caffemodel"
              << " mean.binaryproto labels.txt img.jpg" << std::endl;
    return 1;
  }

  ::google::InitGoogleLogging(argv[0]);

  string model_file   = "/media/media_share/linkfile/faster_rcnn/models/lstm/deploy.prototxt";
  string trained_file = "/media/media_share/linkfile/faster_rcnn/models/lstm/lstm.caffemodel";
  string label_file   = "/media/media_share/linkfile/faster_rcnn/models/lstm/dict.txt";
 
  Classifier classifier;
  
  classifier.loadModel(model_file,trained_file,label_file);
  	
  string file = argv[1];
  cv::Mat img = cv::imread(file, -1); 
  CHECK(!img.empty()) << "Unable to decode image " << file; 
  
  string predictions = classifier.Classify(img);
  
  std::cout << predictions <<std::endl;
}
#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV
