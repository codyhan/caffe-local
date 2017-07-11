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


void detect_row_img(cv::Mat img,Caffe_Detector &my_detect,int location[2])
{

	int min_x =  img.cols + 1 ;

	int max_x =  -1  ;


	cv::Mat src_img = img;
	
	my_detect.SetThresh(0.8,0.3);
	
	map<int,vector<float> > score; 


	map<int,vector<cv::Rect> > label_objs = my_detect.detect(src_img,&score);


	for(map<int,vector<cv::Rect> >::iterator it=label_objs.begin();it!=label_objs.end();it++)
	{  
		int label=it->first;  //标签  
		
		vector<cv::Rect> rects=it->second;  //检测框  
		
		for(int j=0;j<rects.size();j++)
		{  
			if ( (rects[j].x + rects[j].width) > max_x )
				max_x = (rects[j].x + rects[j].width) ;
			if ( rects[j].x < min_x )
				min_x = rects[j].x ;
		} 
		
	}
	
	location[0] = min_x ;

	location[1] = max_x ;
	
}

#if 0
int main(int argc, char **argv)
{
	if(argc < 2 )
	{
		cout<<"usage: detector.bin	imagename"<<endl;
		return -1;
	}
	const std::string slover_file = "/usr/local/word_detector/frcnn/deploy.prototxt";
	const std::string model_file = "/usr/local/word_detector/frcnn/faster_rcnn.caffemodel";

	const std::string single_slover_file = "/media/media_share/linkfile/faster_rcnn/models/deploy.prototxt";
	const std::string single_model_file = "/media/media_share/linkfile/faster_rcnn/models/vgg16_faster_rcnn_iter_15000.caffemodel";
	
	string str_img_name(argv[1]);
	
	Caffe_Detector my_detect ;

	Caffe_Detector single_detect ;

	vector<cv::Rect>	obj_rects;

	vector<detectinfo> info;
	
	bool flag = my_detect.loadModel(slover_file,model_file);

	bool single_flag = single_detect.loadModel(single_slover_file,single_model_file);

	if (flag && single_flag)
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
			
			vector<cv::Rect> rects=it->second;	//检测框  
			
			for(int j=0;j<rects.size();j++)
			{  
				detectinfo body;

				body.x =  rects[j].x ;
				
				body.y =  rects[j].y ;
				
				body.width =  rects[j].width;
				
				body.height =  rects[j].height ;

				
				body.conf = round( score[label][j] * 100 );
				
				body.label =  label ;

				info.push_back(body);


				rects[j].x = body.x ;
				
				rects[j].y = body.y;

				rects[j].width = body.width ;

				rects[j].height = body.height;

				int w_h_ratio = rects[j].width	/ rects[j].height ;

				cout << "ratio : " <<rects[j].width  / rects[j].height << endl;


				if ( w_h_ratio > 1)
				{
					int crop_x = cvRound(rects[j].x -  0.5 * rects[j].width);
					
					if (crop_x <= 0)
					{
						crop_x = 0;
					}

					int crop_y = rects[j].y ;

					int crop_w = cvRound(rects[j].x + rects[j].width + 0.5 * rects[j].width - crop_x );

					if (( crop_w + crop_x + 1) >= w)
					{

						crop_w = w - crop_x - 1;
					}				

					int crop_h = rects[j].height;
					
					cv::Rect roi(crop_x,crop_y,crop_w,crop_h);
					
					cv::Mat save_mat(src_img,roi);
					
					string temp("");
					
					int2str(j,temp);

					int w = save_mat.cols;
							
					int h = save_mat.rows;

					double ratio  = 1.0;

					ratio = 60.000/ h;
					
					resize(save_mat, save_mat, Size(0,0),ratio, ratio);

					string crop_name = str_img_name + temp + "-----resize_crop.jpg" ;
					
					
					cv::imwrite(crop_name  , save_mat);

					int location[2] = {0};

					detect_row_img(save_mat,single_detect,location);
					
					if (location[1] > 0)
					{
						rects[j].x = location[0];
						rects[j].width = location[1] - rects[j].x;
					}

				}

				rectangle(bgr_img,rects[j],Scalar(0,255,0),4);	 //画出矩形框  
				
				string txt=num2str(label)+" : "+num2str(score[label][j]); 
				
				putText(bgr_img,txt,Point(rects[j].x,rects[j].y),CV_FONT_HERSHEY_SIMPLEX,0.5,Scalar(0,255,0)); //标记 类别：置信度	
				
				
			} 
			
		}  

		string name_save = str_img_name + "-----out.jpg" ;

		cv::imwrite(name_save  , bgr_img);

		//string json_str = ToJsonString(info);

		//cout << "json file is " << json_str <<endl;
		
	}
	else
	{
		printf("load model failed.\n");
	}
	
	return 0;
	
}
#endif 0

#if 1

bool test_collision(cv::Rect rect1, cv::Rect rect2)
{
	int x1 = rect1.x;
	int y1 = rect1.y;
	int x2 = rect1.x + rect1.width;
	int y2 = rect1.y + rect1.height;

	int c_y = rect1.y + rect1.height / 2;

	int x3 = rect2.x;
	int y3 = rect2.y;
	int x4 = rect2.x + rect2.width;
	int y4 = rect2.y + rect2.height;

	int c_y_t = rect2.y + rect2.height / 2;

	int distance = (rect1.height + rect2.height) / 2;

	if (abs(c_y_t - c_y) > float( distance*0.8 ))
		return false;


	return ( ( (x1 >=x3 && x1 < x4) || (x3 >= x1 && x3 <= x2) ) &&
	 ( (y1 >=y3 && y1 < y4) || (y3 >= y1 && y3 <= y2) ) ) ? true : false;

}


bool is_same_rect(vector<cv::Rect> &object,cv::Rect &rect)
{

	for(vector<cv::Rect>::iterator it=object.begin(); it!=object.end(); it++)
	{
		
		cv::Rect temp;
		temp.x = it->x;
		temp.y = it->y;
		temp.width = it->width;
		temp.height = it->height;
		
		//if ( abs(it->y - rect.y) < 5 && abs( (it->y + it->height) - (rect.y + rect.height)) < 5 && abs(it->x + it->width -(rect.x+rect.width) > 10) )
		if ( test_collision(temp,rect) )
		{
			int r_x = 0;

			if (it->y > rect.y)
			{
				it->y = rect.y;
			}
				
			if (it->height < rect.height)
			{
				it->height = rect.height;
			}

			if ( (it->x + it->width) < (rect.x+rect.width))
			{
				r_x  = (rect.x+rect.width)  ;
			}
			
			if (it->x > rect.x)
			{
				it->x = rect.x;
					
			}

			if (r_x > 0)
				it->width = r_x - it->x ;
		
			return true;
		}
	}

	return false;
}

void draw_rect(vector<cv::Rect> object,cv::Mat &img,string str_img_name)
{
	int index = 1;
	for(vector<cv::Rect>::iterator it=object.begin(); it!=object.end(); it++)
	{
		string temp("");

		int2str(index,temp);
				
		string lstm_img_name = str_img_name + temp + "-----lstm.jpg" ;

		cv::Rect lstm_roi(it->x,it->y, it->width ,it->height);
			
		cv::Mat lstm_mat(img,lstm_roi);

		cv::imwrite(lstm_img_name  , lstm_mat);

		cv::Rect draw_r ;
		
		draw_r.x = it->x;

		draw_r.y = it->y;

		draw_r.width = it->width;

		draw_r.height = it->height;
		
		rectangle(img,draw_r,Scalar(0,255,0),4);	 //画出矩形框  
		index++;
	}
}


int main(int argc, char **argv)
{
    if(argc < 2 )
    {
        cout<<"usage: detector.bin  imagename"<<endl;
		return -1;
    }
	const std::string slover_file = "/usr/local/word_detector/frcnn/deploy.prototxt";
	const std::string model_file = "/usr/local/word_detector/frcnn/faster_rcnn.caffemodel";
	
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
		
		my_detect.SetThresh(0.7,0.3);
		
		map<int,vector<float> > score; 


#if 0
		int w = bgr_img.cols;
		
		int h = bgr_img.rows;
		
		double w_ratio = 800.0 / w;  

		double h_ratio = 600.0 / h;
		
		resize(bgr_img, bgr_img, Size(800,600),w_ratio, h_ratio);

		printf("image size is %d , %d .\n",bgr_img.cols ,bgr_img.rows);
#endif

		map<int,vector<cv::Rect> > label_objs = my_detect.detect(bgr_img,&score);
		
		for(map<int,vector<cv::Rect> >::iterator it=label_objs.begin();it!=label_objs.end();it++)
		{  
			int label=it->first;  //标签  
			
			vector<cv::Rect> rects=it->second;  //检测框  
			
			for(int j=0;j<rects.size();j++)
			{  
				detectinfo body;

#if 0
				body.x =  cvRound(rects[j].x / w_ratio);
				
				body.y =  cvRound(rects[j].y / h_ratio) ;
				
				body.width =  cvRound(rects[j].width /w_ratio );
				
				body.height =  cvRound( rects[j].height / h_ratio );
#endif

#if 1
				body.x =  rects[j].x ;
				
				body.y =  rects[j].y ;
				
				body.width =  rects[j].width;
				
				body.height =  rects[j].height ;
#endif
				
				body.conf = round( score[label][j] * 100 );
				
				body.label =  label ;

				info.push_back(body);


				rects[j].x = body.x ;
				
				rects[j].y = body.y;

				rects[j].width = body.width ;

				rects[j].height = body.height;

				int w_h_ratio = rects[j].width  / rects[j].height ;

				cout << "ratio : " <<rects[j].width  / rects[j].height << endl;

				string temp("");
					
				int2str(j,temp);

				if ( w_h_ratio > 1)
				{
					
					int crop_x = cvRound(rects[j].x -  0.5 * rects[j].width);
					
					if (crop_x <= 0)
					{
						crop_x = 0;
					}

					int crop_y = rects[j].y ;

					int crop_w = cvRound(rects[j].x + rects[j].width + 0.5 * rects[j].width - crop_x );

					

					if (( crop_w + crop_x + 1) >= w)
					{

						crop_w = w - crop_x - 1;
					}				

					int crop_h = rects[j].height;
					
					cv::Rect roi(crop_x,crop_y,crop_w,crop_h);
					
					cv::Mat save_mat(src_img,roi);

					cv::Mat flip;
					
#if 1

					int w = save_mat.cols;
							
					int h = save_mat.rows;

					double ratio  = 1.0;
					
					if (h < 50.000)
					{
						ratio = 50.000/ h;
						resize(save_mat, save_mat, Size(0,0),ratio, ratio);
					}
					
					cv::flip(save_mat,flip,1);

					string crop_name = str_img_name + temp + "-----resize_crop.jpg" ;
					
					cv::imwrite(crop_name  , flip);
					
					int *r_temp = repeat_detect(my_detect,save_mat);

					//cout << "first is " << r_temp[0] << "second is " << r_temp[1] <<endl;

					r_temp[0] = cvRound(r_temp[0] / ratio ) + crop_x;

					r_temp[1] = cvRound(r_temp[1] / ratio) + crop_x;

					int crop_lx = r_temp[0];

					int crop_rx = r_temp[1];
					

					int *f_temp = repeat_detect(my_detect,flip);

				    int flip_lx = cvRound(( flip.cols - f_temp[1] )/ ratio) + crop_x;

					int flip_rx = cvRound((flip.cols - f_temp[0] )/ ratio) + crop_x;

					if (rects[j].x > r_temp[0] && (r_temp[0] >= 0) )
					{
						rects[j].x = r_temp[0];
						
						rects[j].width = r_temp[1] -  r_temp[0];
					}

					int min_x = body.x ;

					int max_x = body.x + body.width;


					if ( min_x > crop_lx && crop_lx >= 0 )
					{
						min_x = crop_lx;
					}

					
					if ( max_x < crop_rx && crop_rx >= 0)
					{
						max_x = crop_rx;
					}

				    if ( min_x > flip_lx && flip_lx >= 0 )
					{
						min_x = flip_lx;
					}

					
				    if ( max_x < flip_rx && flip_rx >= 0)
					{
						max_x = flip_rx;
					}
					
					rects[j].x = min_x;
				
					
					rects[j].width = max_x  - min_x + 1;

					if ((rects[j].width + rects[j].x) > bgr_img.cols)
						rects[j].width = bgr_img.cols - rects[j].x ;

				}
				//cout << "temp : " << r_temp[0]  << r_temp[1] << endl;
#endif

#if 0
				string lstm_img_name = str_img_name + temp + "-----lstm.jpg" ;

				cv::Rect lstm_roi(rects[j].x,rects[j].y, rects[j].width ,rects[j].height);
					
				cv::Mat lstm_mat(src_img,lstm_roi);

				cv::imwrite(lstm_img_name  , lstm_mat);

				
				rectangle(bgr_img,rects[j],Scalar(0,255,0),4);   //画出矩形框  
				
				string txt=num2str(label)+" : "+num2str(score[label][j]); 
				
				putText(bgr_img,txt,Point(rects[j].x,rects[j].y),CV_FONT_HERSHEY_SIMPLEX,0.5,Scalar(0,255,0)); //标记 类别：置信度  

#endif

				if (!is_same_rect(obj_rects,rects[j]))
					obj_rects.push_back(rects[j]);
				
			} 
			
		}  

		
		

		string name_save = str_img_name + "-----out.jpg" ;

		draw_rect(obj_rects,bgr_img,str_img_name);

	    cv::imwrite(name_save  , bgr_img);

		//string json_str = ToJsonString(info);

		//cout << "json file is " << json_str <<endl;
		
	}
	else
	{
		printf("load model failed.\n");
	}
	
    return 0;
	
}
#endif

