#include "caffe/caffe_detector.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "caffe/layers/rpn_layer.hpp"
#include <opencv2/core/core.hpp>
#include <iostream>
#include <pthread.h>
#include <sys/types.h>
#include <unistd.h>
#include <memory>
#include <sys/sysinfo.h>
#include<sstream>  
#include<opencv2/opencv.hpp> 
#include <caffe/json/json.h>

using namespace std;

using namespace cv; 


struct detectinfo{
	int x;
	int y;
	int width;
	int height;
	int conf;
	int label;
};

string ToJsonString(const vector<detectinfo> &info)
{   
	cout << "in function ToJsonString" << endl;
	cout << "info size  1 is " << info.size() << endl;
	Json::Value root;
	cout << "info size is " << info.size() << endl;
	
	for(unsigned k=0; k<info.size(); k++)    
	{       
		detectinfo detect_info = info[k];           
		Json::Value word;        
		word["rect"].append(detect_info.x);        
		word["rect"].append(detect_info.y);        
		word["rect"].append(detect_info.width);        
		word["rect"].append(detect_info.height);       
		word["conf"]  = Json::Value(detect_info.conf);      
		word["label"] = Json::Value(detect_info.label);       
		root["detectinfo"].append(word);   

	}//for k    
	cout << "in function ToJsonString finish" << endl;
	
	string json_str = root.toStyledString();  
	
	return json_str;
	
}


void DrawRects(cv::Mat &img,
               const vector<cv::Rect> &rects,
               const cv::Scalar &color,
               int thick)
{
    for(unsigned k=0; k<rects.size(); k++)
    {
        cv::rectangle(img, rects[k],color, thick);
    }
}

string num2str(float i){  
    stringstream ss;  
    ss<<i;  
    return ss.str();  
}  

void int2str(const int &int_temp,string &string_temp)  
{  
        stringstream stream;  
        stream<<int_temp;  
        string_temp=stream.str();   //此处也可以用 stream>>string_temp  
}  

int* repeat_detect(Caffe_Detector &my_detect , cv::Mat img)
{	
	cv::Mat bgr_img = img;
	
	map<int,vector<float> > score; 
	
	map<int,vector<cv::Rect> > label_objs = my_detect.detect(bgr_img,&score);

	cout << " repeat img width : " << bgr_img.cols << endl;

	int min_x = bgr_img.cols;

	int max_x = 0;

	int result [2] ={-1,-1};

	for(map<int,vector<cv::Rect> >::iterator it=label_objs.begin();it!=label_objs.end();it++)
	{
		vector<cv::Rect> rects=it->second;  //检测框  

		for(int j=0; j<rects.size();j++)
		{  

	 		if ( (rects[j].x + rects[j].width) > max_x )
				max_x = rects[j].x + rects[j].width;

	 		if ( rects[j].x  < min_x )
				min_x = rects[j].x ;
		}
		
	}

	if (max_x > 0)
	{
		result[0] = min_x;

		result[1] = max_x;
	}


	return result;
	
	
}

int main(int argc, char **argv)
{
    if(argc < 2 )
    {
        cout<<"usage: detector.bin  imagename"<<endl;
		return -1;
    }
	const std::string slover_file = "/media/media_share/linkfile/faster_rcnn/models/frcnn/deploy.prototxt";
	const std::string model_file = "/media/media_share/linkfile/faster_rcnn/models/frcnn/faster_rcnn.caffemodel";
	
    string str_img_name(argv[1]);
	
	Caffe_Detector my_detect ;

	vector<cv::Rect>    obj_rects;

	vector<detectinfo> info;
	
	bool flag = my_detect.loadModel(slover_file,model_file);

	if (flag)
	{

		cv::Mat bgr_img = cv::imread(str_img_name,1);

		cv::Mat src_img = bgr_img;
		
		int w = bgr_img.cols;
		
		int h = bgr_img.rows;
		
		my_detect.SetThresh(0.8,0.3);
		
		map<int,vector<float> > score; 

		map<int,vector<cv::Rect> > label_objs = my_detect.detect(bgr_img,&score);
		
		for(map<int,vector<cv::Rect> >::iterator it=label_objs.begin();it!=label_objs.end();it++)
		{  
			int label=it->first;  //标签  
			
			vector<cv::Rect> rects=it->second;  //检测框  
			
			for(int j=0;j<rects.size();j++)
			{  
				detectinfo body;

				rectangle(bgr_img,rects[j],Scalar(0,255,0),4);   //画出矩形框  
				
				string txt=num2str(label)+" : "+num2str(score[label][j]); 
				
				putText(bgr_img,txt,Point(rects[j].x,rects[j].y),CV_FONT_HERSHEY_SIMPLEX,0.5,Scalar(0,255,0)); //标记 类别：置信度  
				
				
			} 
			
		}  

		string name_save = str_img_name + "-----out.jpg" ;

	    cv::imwrite(name_save  , bgr_img);

		
	}
	else
	{
		printf("load model failed.\n");
	}
	
    return 0;
	
}

