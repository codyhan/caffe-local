#ifndef CAFFELSTM_H
#define CAFFELSTM_H

#include <stdint.h>
#include <algorithm>
#include <utility>
#include <vector>

#include <string>
#include <caffe/caffe.hpp>


#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV


using caffe::shared_ptr;
using caffe::string;
using caffe::vector;
using caffe::map;


class Classifier {
 public:
 	
  Classifier();
  
  ~Classifier();

  string Classify(const cv::Mat& img, int N = 3);

public:

  vector<float> Predict(const cv::Mat& img);
  
  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  void Preprocess(const cv::Mat& img,vector<cv::Mat>* input_channels);

public:
	bool loadModel(const string& model_file,const string& trained_file,const string& label_file);
	
  	bool Initialised(){ return m_initialized; }
    bool BeFree()const{ return m_free;}
    void Occupy(){ m_free = false;}
    void SetFree(){ if( m_initialized == true ) m_free = true;}

private:
	
	shared_ptr<caffe::Net<float> > net_;
	
	cv::Size input_geometry_;
	
	int num_channels_;
	
	cv::Mat mean_;
	
	vector<string> labels_;
	
public:
	
	bool  m_initialized;
	
	bool  m_free;
	
};


#endif // CAFFELSTM_H
