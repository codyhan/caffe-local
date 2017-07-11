#include "caffe/caffe_detector_text.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "caffe/layers/rpn_text_layer.hpp"
#include <opencv2/core/core.hpp>
#include <vector>
#include<iterator>

#include <fstream>

using std::string;
using std::vector;
using std::max;
using std::min;

using namespace caffe;
using namespace std;


//Using for box sort

Caffe_Detector_Text::Caffe_Detector_Text()
{
    conf_thresh = 0.8;
	nms_thresh = 0.3;
    m_initialized = false;
    m_free        = false;
}

Caffe_Detector_Text::~Caffe_Detector_Text()
{
    m_initialized   = false;
    m_free          = false;
    conf_thresh = 0.0;
	nms_thresh = 0.0;
}

bool Caffe_Detector_Text::loadModel(const std::string &model_file, const std::string &weights_file)
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
	
    class_num_ = net_->blob_by_name("scores")->num();

	std::cout<< "class_num is " << class_num_ << std::endl;
	
	m_initialized = true;
	m_free        = true;
	
	if (class_num_ != 1 )
		return false;
	
    return true;
}


void normalize(std::vector<RPNTEXT::abox> &aboxes)
{
	
	float max_score = 0;
	
	float min_score = 100;

	for (int i = 0; i < aboxes.size() ; i++)
	{
		if (aboxes[i].score < min_score)
		{
			min_score = aboxes[i].score;
		} 

		if (aboxes[i].score > max_score)
		{
			max_score = aboxes[i].score;
		} 
	}

	for (int i = 0; i < aboxes.size() ; i++)
	{
		float distance = max_score - min_score;

		aboxes[i].score = (aboxes[i].score - min_score ) / distance;
	}
	
}

void show_set(set<int> a)
{
 
    for(set<int>::iterator i = a.begin(); i != a.end(); i++)
    {
        cout<<*i<<" ";
    }
    cout<<endl;
}
set<int> operator+(const set<int> &a, const set<int>& b)
{
    set<int> c;
    c = b;
    for(set<int>::iterator i = a.begin(); i != a.end(); i++)
    {
         c.insert(*i);
    }
    return c;
}
set<int> operator-(const set<int> &a, const set<int>& b)
{
    set<int> c;
    c = a;
    for(set<int>::iterator i = b.begin(); i != b.end(); i++)
    {
         if(c.find(*i) != c.end())
               c.erase(*i);
    }
    return c;
}
set<int> operator*(const set<int> &a, const set<int>& b)
{
    set<int> c,d;
    c = a;
    for(set<int>::iterator i = b.begin(); i != b.end(); i++)
    {
         if(c.find(*i) != c.end())
               d.insert(*i);
    }
    return d;
}


void make_set_union(int a, int b , set<int> &set_union)
{
	int start = a;
	
	while( start <= b )
	{
		set_union.insert(start);
		start++;
	}
}



bool is_best_neighbor(std::vector<RPNTEXT::abox> vector_boxes, RPNTEXT::abox box)
{
	
	float center_x = (box.x1 + box.x2) / 2.0;

	for(int i = 0; i < vector_boxes.size(); i++)
	{
		RPNTEXT::abox tmp;
		
	    tmp.x1 = vector_boxes[i].x1;
		
	    tmp.y1 = vector_boxes[i].y1;
		
	    tmp.x2 = vector_boxes[i].x2;
		
	    tmp.y2 = vector_boxes[i].y2;

		float vec_center_x = (tmp.x1 + tmp.x2) / 2.0;

		if (abs( center_x - vec_center_x ) < (50 + 16) )
		{
			int box_a = box.y1;

			int box_b = box.y2;

			set<int> box_set;
			
			make_set_union(box_a , box_b,box_set);

			int vec_box_a = vector_boxes[i].y1;

			int vec_box_b = vector_boxes[i].y2;

			set<int> vec_box_set;
			
			make_set_union(vec_box_a , vec_box_b,vec_box_set);

			set<int> union_box = box_set + vec_box_set;

			int number_union_box = union_box.size();

			set<int> intersection_box = box_set * vec_box_set;

			int number_intersection_box = intersection_box.size();

			float overlap = float(number_intersection_box) / float (number_union_box);

			if (overlap > 0.7)
				return true;

			overlap = float(number_intersection_box) / float (box_set.size());

			if (overlap > 0.7)
				return true;

			overlap = float(number_intersection_box) / float (vec_box_set.size());

			if (overlap > 0.7)
				return true;
			
		}
		
		
	}

	return false;
}

void get_text_lines(std::vector<RPNTEXT::abox> &aboxes,std::vector<std::vector<RPNTEXT::abox> > &out_boxes)
{

	while (aboxes.size() > 0)
	{

		std::vector<RPNTEXT::abox> vector_boxes;

		RPNTEXT::abox tmp;
		
	    tmp.x1 = aboxes[0].x1;
		
	    tmp.y1 = aboxes[0].y1;
		
	    tmp.x2 = aboxes[0].x2;
		
	    tmp.y2 = aboxes[0].y2;
		
	    tmp.score = aboxes[0].score;

		vector_boxes.push_back(tmp);

		aboxes.erase(aboxes.begin() + 0);
#if 0
		for (int i = 0 ; i < aboxes.size();)
		{
			if (is_best_neighbor(vector_boxes,aboxes[i]))
			{
				RPNTEXT::abox tmp_box;
		
			    tmp_box.x1 = aboxes[i].x1;
				
			    tmp_box.y1 = aboxes[i].y1;
				
			    tmp_box.x2 = aboxes[i].x2;
				
			    tmp_box.y2 = aboxes[i].y2;
				
			    tmp_box.score = aboxes[i].score;

				vector_boxes.push_back(tmp_box);
				
				aboxes.erase(aboxes.begin() + i);
			}
			else
			{
				i++;
			}
			
		}
#endif

#if 1
		
		while (1)
		{
			int current_number = aboxes.size();
			
			for (int i = 0 ; i < aboxes.size();)
			{
				if (is_best_neighbor(vector_boxes,aboxes[i]))
				{
					RPNTEXT::abox tmp_box;
			
				    tmp_box.x1 = aboxes[i].x1;
					
				    tmp_box.y1 = aboxes[i].y1;
					
				    tmp_box.x2 = aboxes[i].x2;
					
				    tmp_box.y2 = aboxes[i].y2;
					
				    tmp_box.score = aboxes[i].score;

					vector_boxes.push_back(tmp_box);
					
					aboxes.erase(aboxes.begin() + i);
				}
				else
				{
					i++;
				}
			
			}

			if (current_number == aboxes.size())
				break;
			
		}
#endif

		out_boxes.push_back(vector_boxes);
		
	}



}

void get_text_location(std::vector<std::vector<RPNTEXT::abox> > out_boxes,std::vector<RPNTEXT::abox> &result_box)
{
	
	if (out_boxes.size() > 0)
	{

		for (int i = 0 ; i < out_boxes.size(); i++)
		{
			std::vector<RPNTEXT::abox> temp = out_boxes[i];

			float min_top_x = 100000;

			float min_top_y = 100000;

			float max_bottom_x = 0;

			float max_bottom_y = 0;

			float max_score = 0;

			for (int j = 0; j < temp.size(); j++)
			{
				if (min_top_x > temp[j].x1 )
					min_top_x = temp[j].x1;
				
				if (min_top_y > temp[j].y1 )
					min_top_y = temp[j].y1;

				if (max_bottom_x < temp[j].x2 )
					max_bottom_x = temp[j].x2;
				
				if (max_bottom_y < temp[j].y2 )
					max_bottom_y = temp[j].y2;

				if (max_score < temp[j].score )
					max_score = temp[j].score;
				
			}

			RPNTEXT::abox result ;

			result.x1 = min_top_x;
			result.y1 = min_top_y;

			result.x2 = max_bottom_x;
			result.y2 = max_bottom_y;

			result.score = max_score;
			
			result_box.push_back(result);
		
		}

		
	}
	
}


map<int,cv::Rect> Caffe_Detector_Text::detect(cv::Mat &image, map<int,float>* objectScore)
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
	
    int height =  round(image.rows * img_scale);
	
    int width =   round(image.cols * img_scale);

   // printf("%d,%d\n", height, width);
	
    //int num_out;
    
    cv::Mat cv_resized;
    image.convertTo(cv_resized, CV_32FC3);
    cv::resize(cv_resized, cv_resized, cv::Size(width, height));
    cv::Mat mean(height, width, cv_resized.type(), cv::Scalar(102.9801, 115.9465, 122.7717));
    cv::Mat normalized;
    subtract(cv_resized, mean, normalized);

	//cv::imwrite("/media/media_share/linkfile/CTPN/out_img/out.jpg"  , normalized);


    float im_info[3];
	
    im_info[0] = height;
	
    im_info[1] = width;
	
	im_info[2] = img_scale;
	
    boost::shared_ptr<Blob<float> > input_layer = net_->blob_by_name("data");
	
	input_layer->Reshape(1, normalized.channels() , height, width);

	net_->Reshape();

	cout << "input_layer Reshape "<< input_layer->shape_string() <<endl;
	
    float* input_data = input_layer->mutable_cpu_data();
	
    vector<cv::Mat> input_channels;
	
    for (int i = 0; i < input_layer->channels(); ++i) 
	{
        cv::Mat channel(height, width, CV_32FC1, input_data);
		
        input_channels.push_back(channel);
		
        input_data += height * width;
    }
	
    cv::split(normalized, input_channels);
/*
	for (int i=0; i<height ; i++)  
	{  
		float* pData1=normalized.ptr<float>(i); 
		for (int j=0; j< 10 ; j++)  
		{  
			std::cout << " normalized : " << pData1[j] << std::endl;
		}
		break;
	}
*/
    net_->blob_by_name("im_info")->set_cpu_data(im_info);

    net_->Forward();

	map<int,cv::Rect > label_objs;    //每个类别，对应的检测目标框  

    boost::shared_ptr<Blob<float> > bbox_delt_data = net_->blob_by_name("rois");

	int num = net_->blob_by_name("rois")->num();

    boost::shared_ptr<Blob<float> >score = net_->blob_by_name("scores");

    cv::Mat bbox_delt(num, 4, CV_32FC1);

	vector<RPNTEXT::abox>aboxes;
	
    for (int i = 0; i < num; ++i)
    {
        bbox_delt.at<float>(i, 0) = bbox_delt_data->data_at(i, 0, 0, 0) / img_scale;
        bbox_delt.at<float>(i, 1) = bbox_delt_data->data_at(i, 1, 0, 0) / img_scale;
        bbox_delt.at<float>(i, 2) = bbox_delt_data->data_at(i, 2, 0, 0) / img_scale;
        bbox_delt.at<float>(i, 3) = bbox_delt_data->data_at(i, 3, 0, 0) / img_scale;
    }

    for (int j = 0; j < num; ++j)
    {

        RPNTEXT::abox tmp;
		
        tmp.x1 = bbox_delt.at<float>(j, 0);
		
        tmp.y1 = bbox_delt.at<float>(j, 1);
		
        tmp.x2 = bbox_delt.at<float>(j, 2);
		
        tmp.y2 = bbox_delt.at<float>(j, 3);
		
        tmp.score = score->data_at(j,0,0,0);
		
        aboxes.push_back(tmp);
    }
	
	std::sort(aboxes.rbegin(), aboxes.rend());

	RPNTEXT::nms(aboxes, NMS_THRESH);

	normalize(aboxes);
	
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

	std::vector<std::vector<RPNTEXT::abox> > out_boxes;

	get_text_lines(aboxes,out_boxes);
	
	std::vector<RPNTEXT::abox>  box_location;
	
	get_text_location(out_boxes,box_location);


	
    for (int k = 0; k < box_location.size();)
    {
    	float width = box_location[k].x2 - box_location[k].x1;

		float height = box_location[k].y2 - box_location[k].y1;

		float min_ratio =  width / height;
		
        if (width <= 32 || min_ratio <= 1.2 || box_location[k].score <= 0.7)
        {
            box_location.erase(box_location.begin() + k);
        }
        else
        {
            k++;
        }
    }

	RPNTEXT::nms(box_location, NMS_THRESH);
		
    for(int i=0;i<box_location.size();++i) 
	{
		cv::Rect rect = cv::Rect(cv::Point(box_location[i].x1,box_location[i].y1),cv::Point(box_location[i].x2,box_location[i].y2)); 
		
		label_objs[i]=rect;
	}
	  

    if(objectScore!=NULL)
	{    
        for(int i = 0; i < box_location.size() ; ++i) 
    	{
            float tmp = box_location[i].score;  
			
       	    objectScore->insert(pair<int,float>(i,tmp));  
    	}	
    }  
	
	
    return label_objs;
	
}


void Caffe_Detector_Text::SetThresh(float cf_thresh,float ns_thresh)
{
        conf_thresh = cf_thresh;
		nms_thresh = ns_thresh;
}

void Caffe_Detector_Text::SetConfThresh(float cf_thresh)
{
        conf_thresh = cf_thresh;
}










