#include "caffe/caffe_detector.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "caffe/layers/rpn_layer.hpp"
#include <opencv2/core/core.hpp>
#include <vector>

#include <fstream>

using std::string;
using std::vector;
using std::max;
using std::min;

using namespace caffe;


//Using for box sort

Caffe_Detector::Caffe_Detector()
{
    conf_thresh = 0.8;
	nms_thresh = 0.3;
    m_initialized = false;
    m_free        = false;
}

Caffe_Detector::~Caffe_Detector()
{
    m_initialized   = false;
    m_free          = false;
    conf_thresh = 0.0;
	nms_thresh = 0.0;
}

bool Caffe_Detector::loadModel(const std::string &model_file, const std::string &weights_file)
{
#ifdef CPU_ONLY
    Caffe::set_mode(Caffe::CPU);
#else
    Caffe::set_mode(Caffe::GPU);
#endif //#ifdef CPU_ONLY 
	class_num_ = 0 ;

    net_.reset(new Net<float>(model_file, TEST));
	
	if (net_ == NULL)
		return false ;
		
    net_->CopyTrainedLayersFrom(weights_file); 
	
    class_num_ = net_->blob_by_name("cls_prob")->channels();
	
	m_initialized = true;
	m_free        = true;
	
	if (class_num_ <= 1 )
		return false;
	
    return true;
}


map<int,vector<cv::Rect> > Caffe_Detector::detect(cv::Mat &image,map<int,vector<float> >* objectScore)
{
    if(objectScore!=NULL)   //如果需要保存置信度  
        objectScore->clear();  

    float CONF_THRESH = conf_thresh;
	
    float NMS_THRESH = nms_thresh;

	//cout << " conf_thresh is " << CONF_THRESH << " , nms thresh is " << NMS_THRESH << endl;

    int max_side = max(image.rows, image.cols);
    int min_side = min(image.rows, image.cols);

    float max_side_scale = float(max_side) / float(INPUT_SIZE_LONG);
    float min_side_scale = float(min_side) / float(INPUT_SIZE_NARROW);
    float max_scale = max(max_side_scale, min_side_scale);

    float img_scale = float(1) / max_scale;
    int height = int(image.rows * img_scale);
    int width = int(image.cols * img_scale);
    //printf("%d,%d", height, width);
    //int num_out;
    cv::Mat cv_resized;
    image.convertTo(cv_resized, CV_32FC3);
    cv::resize(cv_resized, cv_resized, cv::Size(width, height));
    cv::Mat mean(height, width, cv_resized.type(), cv::Scalar(102.9801, 115.9465, 122.7717));
    cv::Mat normalized;
    subtract(cv_resized, mean, normalized);

    float im_info[3];
    im_info[0] = height;
    im_info[1] = width;
    im_info[2] = img_scale;
    shared_ptr<Blob<float> > input_layer = net_->blob_by_name("data");
	std::cout << " ======" << input_layer->data_at(0,0,0 ,0) << std::endl;
	
	std::cout << " ======" << input_layer->data_at(0,0,0 ,1) << std::endl;
    input_layer->Reshape(1, normalized.channels(), height, width);
	
    net_->Reshape();
    float* input_data = input_layer->mutable_cpu_data();
    vector<cv::Mat> input_channels;
    for (int i = 0; i < input_layer->channels(); ++i) {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels.push_back(channel);
        input_data += height * width;
    }
    cv::split(normalized, input_channels);
    net_->blob_by_name("im_info")->set_cpu_data(im_info);
    net_->Forward();


    int num = net_->blob_by_name("rois")->num();
    const float *rois_data = net_->blob_by_name("rois")->cpu_data();
    //int num1 = net_->blob_by_name("bbox_pred")->num();
    cv::Mat rois_box(num, 4, CV_32FC1);
    for (int i = 0; i < num; ++i)
    {
        rois_box.at<float>(i, 0) = rois_data[i * 5 + 1] / img_scale;
        rois_box.at<float>(i, 1) = rois_data[i * 5 + 2] / img_scale;
        rois_box.at<float>(i, 2) = rois_data[i * 5 + 3] / img_scale;
        rois_box.at<float>(i, 3) = rois_data[i * 5 + 4] / img_scale;
    }

    shared_ptr<Blob<float> > bbox_delt_data = net_->blob_by_name("bbox_pred");

	//std::cout << "bbox_delt_data shape" << bbox_delt_data->shape_string()<< std::endl;

    shared_ptr<Blob<float> >score = net_->blob_by_name("cls_prob");

	//shared_ptr<Blob<float> >temp = net_->blob_by_name("fc7");

	//std::cout<<"fc num out is " << temp->count()/temp->num() <<std::endl;

    map<int,vector<cv::Rect> > label_objs;    //每个类别，对应的检测目标框  
	
    for (int i = 1; i < class_num_; ++i)
    {
        cv::Mat bbox_delt(num, 4, CV_32FC1);
        for (int j = 0; j < num; ++j)
        {
            bbox_delt.at<float>(j, 0) = bbox_delt_data->data_at(j, i-1 * 4, 0, 0);
            bbox_delt.at<float>(j, 1) = bbox_delt_data->data_at(j, i * 4+1, 0, 0);
            bbox_delt.at<float>(j, 2) = bbox_delt_data->data_at(j, i * 4+2, 0, 0);
            bbox_delt.at<float>(j, 3) = bbox_delt_data->data_at(j, i * 4+3, 0, 0);
        }

        cv::Mat box_class = RPN::bbox_tranform_inv(rois_box, bbox_delt);

        vector<RPN::abox>aboxes;
        for (int j = 0; j < box_class.rows; ++j)
        {
            if (box_class.at<float>(j, 0) < 0)  box_class.at<float>(j, 0) = 0;
            if (box_class.at<float>(j, 0) > (image.cols - 1))   box_class.at<float>(j, 0) = image.cols - 1;
            if (box_class.at<float>(j, 2) < 0)  box_class.at<float>(j, 2) = 0;
            if (box_class.at<float>(j, 2) > (image.cols - 1))   box_class.at<float>(j, 2) = image.cols - 1;

            if (box_class.at<float>(j, 1) < 0)  box_class.at<float>(j, 1) = 0;
            if (box_class.at<float>(j, 1) > (image.rows - 1))   box_class.at<float>(j, 1) = image.rows - 1;
            if (box_class.at<float>(j, 3) < 0)  box_class.at<float>(j, 3) = 0;          
            if (box_class.at<float>(j, 3) > (image.rows - 1))   box_class.at<float>(j, 3) = image.rows - 1;
            RPN::abox tmp;
            tmp.x1 = box_class.at<float>(j, 0);
            tmp.y1 = box_class.at<float>(j, 1);
            tmp.x2 = box_class.at<float>(j, 2);
            tmp.y2 = box_class.at<float>(j, 3);
            tmp.score = score->data_at(j,i,0,0);
            aboxes.push_back(tmp);
        }
        std::sort(aboxes.rbegin(), aboxes.rend());
        RPN::nms(aboxes, NMS_THRESH);
        for (int k = 0; k < aboxes.size();)
        {
            if (aboxes[k].score < CONF_THRESH)
            {
                aboxes.erase(aboxes.begin() + k);
            }
            else
            {
                k++;
            }
        }


		//################ 将类别i的所有检测框，保存  
        vector<cv::Rect> rect(aboxes.size());    //对于类别i，检测出的矩形框  
        for(int ii=0;ii<aboxes.size();++ii)  
            rect[ii]=cv::Rect(cv::Point(aboxes[ii].x1,aboxes[ii].y1),cv::Point(aboxes[ii].x2,aboxes[ii].y2));  
        label_objs[i]=rect;     
        //################ 将类别i的所有检测框的打分，保存  
        if(objectScore!=NULL){           //################ 将类别i的所有检测框的打分，保存  
            vector<float> tmp(aboxes.size());       //对于 类别i，检测出的矩形框的得分  
            for(int ii=0;ii<aboxes.size();++ii)  
                tmp[ii]=aboxes[ii].score;  
            objectScore->insert(pair<int,vector<float> >(i,tmp));  
        }  
		
    }
	
    return label_objs;
	
}


void Caffe_Detector::SetThresh(float cf_thresh,float ns_thresh)
{
        conf_thresh = cf_thresh;
		nms_thresh = ns_thresh;
}

void Caffe_Detector::SetConfThresh(float cf_thresh)
{
        conf_thresh = cf_thresh;
}










