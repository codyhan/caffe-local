#ifndef DETECTORPRIVATE_H
#define DETECTORPRIVATE_H

//#include "detectresult.h"
#define INPUT_SIZE_NARROW  600
#define INPUT_SIZE_LONG  1000

#include <stdint.h>
#include <algorithm>
#include <utility>
#include <vector>

#include <string>
#include <caffe/net.hpp>
#include <opencv2/core/core.hpp>

using caffe::shared_ptr;
using caffe::string;
using caffe::vector;
using caffe::map;

struct Result
{
    float arry[6];
};

class Caffe_Detector
{

public:
	float conf_thresh;
	float nms_thresh;
	bool  m_initialized;
	bool  m_free;
		
public:
    Caffe_Detector();
	~Caffe_Detector();
	
    bool loadModel(const std::string &model_file, const std::string &weights_file);
	
    map<int,vector<cv::Rect> > detect(cv::Mat &image,map<int,vector<float> >* objectScore);
	 
	bool Initialised(){ return m_initialized; }
	void SetThresh(float cf_thresh,float ns_thresh);
	void SetConfThresh(float cf_thresh);
    bool BeFree()const{ return m_free;}
    void Occupy(){ m_free = false;}
    void SetFree(){ if( m_initialized == true ) m_free = true;}

private:
    shared_ptr< caffe::Net<float> > net_;
	int class_num_;
};

#endif // DETECTORPRIVATE_H