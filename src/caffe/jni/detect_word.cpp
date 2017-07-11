#include "com_lstm_detect_word_detector.h"
#include "caffe/caffe_lstm.hpp"
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

static vector<Classifier> word_detectors(get_nprocs()); 	//get_nprocs()

pthread_mutex_t             word_detector_mutex;

pthread_cond_t              word_detector_free_cond;

int word_GetDetector();

void word_FreeDetector(int index);


char* word_jstringTostring(JNIEnv* env, jstring jstr)
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

string word_num2str(float i){  
    stringstream ss;  
    ss<<i;  
    return ss.str();  
} 


JNIEXPORT jstring JNICALL Java_com_lstm_detect_word_1detector_detect_1word(JNIEnv *env, jclass object, jlong data_addr,jint w, jint h, jint c)
{

	//char* img_name = word_jstringTostring(env,image_path);
	
    //string str_img_name(img_name);

	#ifdef CPU_ONLY
  		Caffe::set_mode(Caffe::CPU);
	#else
  		Caffe::set_mode(Caffe::GPU);
	#endif

	string predictions("");

	try{


			int detector_index = word_GetDetector();

			if(detector_index < 0 )
			{
				cout<<"Cannot load model files. "<<endl;
				return env->NewStringUTF(predictions.c_str());
			}

			Classifier my_detect = word_detectors[detector_index];

			cv::Mat bgr_img(h,w, CV_8UC3 ,(unsigned char *)data_addr);

			//cv::imwrite("/media/media_share/linkfile/out/out.jpg",bgr_img);

			
			//cout << "img channel is " << c << endl;
				
			int num_channels_ = 3;
			
			cv::Mat sample;
			if (bgr_img.channels() == 3 && num_channels_ == 1)
				cv::cvtColor(bgr_img, sample, cv::COLOR_BGR2GRAY);
			else if (bgr_img.channels() == 4 && num_channels_ == 1)
				cv::cvtColor(bgr_img, sample, cv::COLOR_BGRA2GRAY);
			else if (bgr_img.channels() == 4 && num_channels_ == 3)
				cv::cvtColor(bgr_img, sample, cv::COLOR_BGRA2BGR);
			else if (bgr_img.channels() == 1 && num_channels_ == 3)
				cv::cvtColor(bgr_img, sample, cv::COLOR_GRAY2BGR);
			else
				sample = bgr_img;

			bgr_img = sample;

			cv::imwrite("/media/media_share/linkfile/out/out.jpg",bgr_img);

			//cv::Mat bgr_img = cv::imread(str_img_name,-1);

			if (bgr_img.empty())
			{
				std::cout << "Unable to decode image " <<std::endl;
			}

			predictions = my_detect.Classify(bgr_img);
			
			word_FreeDetector(detector_index);

			//std::cout << prob <<std::endl;
		
	}
	catch(exception& e)
	{
		cout<<"catch the exception"<<endl;
		return env->NewStringUTF(predictions.c_str());
	}

	jstring result = env->NewStringUTF(predictions.c_str());

	return result;
	
}

int word_GetDetector()
{
	
	pthread_mutex_lock (&word_detector_mutex);

    if(word_detectors.size() == 0)
    {
        word_detectors.resize( get_nprocs() );
    }


    int free_index = -1;
	
    while(1)
    {
        for(int k=0; k < word_detectors.size(); k++)
        {
            if( word_detectors[k].BeFree() )
            {
                free_index = k;
                word_detectors[k].Occupy();
                break;
            }
        }

        if(free_index >= 0)
        {
            break;
        }

        bool new_loaded = false;
        for(unsigned k=0; k< word_detectors.size(); k++)
        {
            if( false == word_detectors[k].Initialised() )
            {
				/*flag : 0 detect box 1 : detect word*/
				
				bool rflag  = false;

				const std::string slover_file = "/usr/local/word_detector/lstm/deploy.prototxt";

				const std::string model_file = "/usr/local/word_detector/lstm/lstm.caffemodel";
				
				const std::string label_file = "/usr/local/word_detector/lstm/dict.txt";
				
				rflag = word_detectors[k].loadModel(slover_file,model_file,label_file);
		
				
                if(  rflag == false)
                {
                    cout<<"cannot initialize detector "<<endl;
                }
				else
                {
                    new_loaded = true;
				    break;
                }
            }
        }

        if(false == new_loaded )
        {
            pthread_cond_wait(&word_detector_free_cond, &word_detector_mutex);
        }

    }//while
    pthread_mutex_unlock(&word_detector_mutex);
    return free_index;
}


void word_FreeDetector(int index)
{
    if(index < 0) return;

    pthread_mutex_lock (&word_detector_mutex);

    word_detectors[index].SetFree();
    pthread_cond_signal(&word_detector_free_cond);

    pthread_mutex_unlock(&word_detector_mutex);
}

