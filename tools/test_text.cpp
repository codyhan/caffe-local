#include "caffe/caffe_detector_text.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "caffe/layers/rpn_text_layer.hpp"
#include <opencv2/core/core.hpp>
#include <iostream>
#include <pthread.h>
#include <sys/types.h>
#include <unistd.h>
#include <memory>
#include <sys/sysinfo.h>
#include<sstream>  
#include<opencv2/opencv.hpp> 


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


int main(int argc, char **argv)
{
    if(argc < 2 )
    {
        cout<<"usage: detector.bin  imagename"<<endl;
		return -1;
    }
	const std::string slover_file = "/media/media_share/linkfile/faster_rcnn/ctpn_models/deploy.prototxt";
	const std::string model_file = "/media/media_share/linkfile/faster_rcnn/ctpn_models/ctpn_trained_model.caffemodel";
	
    string str_img_name(argv[1]);
	
	Caffe_Detector_Text my_detect ;

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
		
		map<int,float> score; 

		map<int,cv::Rect> label_objs = my_detect.detect(bgr_img,&score);

		string save_name = str_img_name  + "-----out.jpg" ;
		
		
		for(map<int,cv::Rect >::iterator it=label_objs.begin();it!=label_objs.end();it++)
		{
			cv::Rect rect = it->second;

			detectinfo body;

			body.x =  rect.x ;
				
			body.y =  rect.y ;
				
			body.width =  rect.width;
				
			body.height =  rect.height ;

			rectangle(bgr_img,rect,Scalar(0,255,255),2);   //»­³ö¾ØÐÎ¿ò  
			
		}

		cv::imwrite(save_name  , bgr_img);
	}
	else
	{
		printf("load model failed.\n");
	}
	
    return 0;
	
}

