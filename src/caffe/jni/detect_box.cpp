#include "com_lstm_detect_word_detector.h"
#include "caffe/caffe_detector_text.hpp"
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
#include <Python.h> 
#include<sstream>  
#include<opencv2/opencv.hpp> 
#include <exception> 
#include <caffe/json/json.h>

using namespace std;
using namespace caffe;
using namespace cv;
using namespace Json;


struct detectinfo{
	int x;
	int y;
	int width;
	int height;
	int conf;
	string label;
};

char labels[2][256] = {"background","text"};

string ToJsonString(const vector<detectinfo> &info);


static vector<Caffe_Detector_Text> detectors(get_nprocs()); //get_nprocs()
pthread_mutex_t             detector_mutex;
pthread_cond_t              detector_free_cond;

int GetDetector();

void FreeDetector(int index);



char* jstringTostring(JNIEnv* env, jstring jstr)
{
	char* rtn = NULL;
	jclass clsstring = env->FindClass("java/lang/String");
	jstring strencode = env->NewStringUTF("utf-8");
	jmethodID mid = env->GetMethodID(clsstring, "getBytes", "(Ljava/lang/String;)[B");
	jbyteArray barr= (jbyteArray)env->CallObjectMethod(jstr, mid, strencode);
	jsize alen = env->GetArrayLength(barr);
	jbyte* ba = env->GetByteArrayElements(barr, JNI_FALSE);
	if (alen > 0)
	{
		rtn = (char*)malloc(alen + 1);
		memcpy(rtn, ba, alen);
		rtn[alen] = 0;
	}
	env->ReleaseByteArrayElements(barr, ba, 0);
	return rtn;
}


string num2str(float i){  
    stringstream ss;  
    ss<<i;  
    return ss.str();  
} 


void init(vector<detectinfo> &info)
{
	detectinfo body;

	body.x =  0 ;

	body.y =  0 ;

	body.width =  0;

	body.height =  0 ;

	body.conf = 0;

	body.label =  string("");

	info.push_back(body);
}


JNIEXPORT jstring JNICALL Java_com_lstm_detect_word_1detector_detect_1box  (JNIEnv *env, jclass object, jstring image_path, jfloat thresh,jfloat nms_thresh)
{
	#ifdef CPU_ONLY
  		Caffe::set_mode(Caffe::CPU);
	#else
  		Caffe::set_mode(Caffe::GPU);
	#endif
	

	char* img_name = jstringTostring(env,image_path);
	
    string str_img_name(img_name);

	vector<detectinfo> info;

	vector<cv::Rect>  obj_rects;
	
	init(info);

	try 
	{

		int detector_index = GetDetector();
		
	    if(detector_index < 0 )
	    {
	        cout<<"Cannot load model files. "<<endl;
			
			jstring wrong_info;
			
	        return wrong_info;
	    }
		
		Caffe_Detector_Text my_detect = detectors[detector_index];

		cv::Mat bgr_img = cv::imread(str_img_name,1);

		my_detect.SetThresh(thresh,nms_thresh);

		map<int,float > score; 
		
		map<int,cv::Rect > label_objs = my_detect.detect(bgr_img,&score);

		int i = 0;
		
		for(map<int,cv::Rect >::iterator it = label_objs.begin() ; it != label_objs.end(); it++ )
		{
			cv::Rect rect = it->second;

			detectinfo body;

			body.x =  rect.x ;
				
			body.y =  rect.y ;
				
			body.width =  rect.width;
				
			body.height =  rect.height ;

			body.conf = round( score[i] * 100 );
				
			body.label =  string("text");

			i++;

			info.push_back(body);
		
		}

		FreeDetector(detector_index);


	}
	catch (exception& e)  
	{
		 cout << e.what() << endl;
	}


	string json_str = ToJsonString(info);


	jstring result = env->NewStringUTF(json_str.c_str());

	return result;

	
}



int GetDetector()
{
	
	const std::string slover_file = "/usr/local/word_detector/frcnn/deploy.prototxt";
	
	const std::string model_file = "/usr/local/word_detector/frcnn/ctpn_trained_model.caffemodel";

	pthread_mutex_lock (&detector_mutex);

    if(detectors.size() == 0)
    {
        detectors.resize( get_nprocs() );
    }


    int free_index = -1;
	
    while(1)
    {
        for(int k=0; k < detectors.size(); k++)
        {
            if( detectors[k].BeFree() )
            {
                free_index = k;
                detectors[k].Occupy();
                break;
            }
        }

        if(free_index >= 0)
        {
            break;
        }

        bool new_loaded = false;
        for(unsigned k=0; k<detectors.size(); k++)
        {
            if( false == detectors[k].Initialised() )
            {
                if( detectors[k].loadModel(slover_file,model_file) == false)
                {
                    cout<<"cannot initialize detector "<<endl;
                }
				else
                {
                    new_loaded = true;
				    detectors[k].SetThresh(0.85,0.2);
				    break;
                }
            }
        }

        if(false == new_loaded )
        {
            pthread_cond_wait(&detector_free_cond, &detector_mutex);
        }

    }//while
    pthread_mutex_unlock(&detector_mutex);
    return free_index;
}


void FreeDetector(int index)
{
    if(index < 0) return;

    pthread_mutex_lock (&detector_mutex);

    detectors[index].SetFree();
    pthread_cond_signal(&detector_free_cond);

    pthread_mutex_unlock(&detector_mutex);
}

string ToJsonString(const vector<detectinfo> &info)
{   
	Json::Value root;

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
	
	string json_str = root.toStyledString();  
	
	return json_str;
	
}


