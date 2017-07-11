#include "caffe/caffe_lstm.hpp"
#include <fstream>
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
#include "caffe/layers/continuation_indicator_layer.hpp"

using std::string;
using std::vector;
using std::max;
using std::min;

using namespace caffe;
using namespace std;




Classifier::Classifier()
{
    m_initialized = false;
	
    m_free        = false;
}


bool Classifier::loadModel(const string& model_file,const string& trained_file,const string& label_file)
{

#ifdef CPU_ONLY
	Caffe::set_mode(Caffe::CPU);
#else
	Caffe::set_mode(Caffe::GPU);
#endif

	Caffe::set_mode(Caffe::CPU);

	/* Load the network. */
	cout << "indicator load model" << endl;

	net_.reset(new Net<float>(model_file, TEST));

	net_->CopyTrainedLayersFrom(trained_file);

/*
	CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
	
	CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";
*/
	if (net_->num_inputs()!=1 || net_->num_outputs() != 1 )
	{
		std::cout << " Network should have exactly one input and one output." <<std::endl;
		return false;
	}

	//  Blob<float>* input_layer = net_->input_blobs()[0];
	
	num_channels_ = 3;
/*
	CHECK(num_channels_ == 3 || num_channels_ == 1)
		<< "Input layer should have 1 or 3 channels.";
*/
	if (num_channels_ != 3 && num_channels_ != 1)
	{
		std::cout << "Input layer should have 1 or 3 channels." <<std::endl;
		return false;
	}
	
	input_geometry_ = cv::Size(128, 32);

	/* Load labels. */
	std::ifstream labels(label_file.c_str());

	if (NULL == labels)
	{
		std::cout << "Unable to open labels file ." <<std::endl;
		return false;
	}
	//CHECK(labels) << "Unable to open labels file " << label_file;
	
	string line;
	
	while (std::getline(labels, line))
	{
		labels_.push_back(string(line));
	}

	Blob<float>* output_layer = net_->output_blobs()[0];
/*
	CHECK_EQ(33, output_layer->num())
		<< "Number of labels is different from the output layer dimension.";
*/
	if (33 != output_layer->num())
	{
		std::cout << "Number of labels is different from the output layer dimension." <<std::endl;
		return false;
	}

	m_initialized = true;
	
	m_free        = true;

	return true;
	
}


Classifier::~Classifier()
{
    m_initialized   = false;
    m_free          = false;
}


static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
  return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
  std::vector<std::pair<float, int> > pairs;
  for (size_t i = 0; i < v.size(); ++i)
    pairs.push_back(std::make_pair(v[i], i));
  std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

  std::vector<int> result;
  for (int i = 0; i < N; ++i)
    result.push_back(pairs[i].second);
  return result;
}

/* Return the top N predictions. */
float im_info[3];

string Classifier::Classify(const cv::Mat& img, int N) {

  im_info[0] = img.rows;
   im_info[1] = img.cols;
   im_info[2] = 2.0;
  
   std::cout << " height is " << im_info[1] << " width is " <<	 im_info[0]  << std::endl;


  std::vector<float> output = Predict(img);
  std::vector<float> temp;
  std::vector<int> maxid;
  std::vector<int> maxpathid;
  std::vector<string> prevpred;
  std::vector<string> newpred;
  std::vector<float> prevscore;

  for(int k=0;k<21019;k++)
  {
     temp.push_back(output[k]);
  }
  maxid = Argmax(temp,N);
  for(int i=0;i<N;i++){
     prevscore.push_back(output[maxid[i]]);
     prevpred.push_back(std::to_string(maxid[i]));
  }
  std::vector<int> curlabel;
  std::vector<float> curscore;
  std::vector<float> pathscore;
  string finalpreds("");
  std::vector<string> preduck(N);
  for(int i=21019;i<693627;i=i+21019){
     temp.resize(0);
     curlabel.resize(0);
     curscore.resize(0);
     pathscore.resize(0);
     newpred.resize(0);
     for (int j=0;j<21019;j++)
     {
	temp.push_back(output[i+j]);
     }
     maxid = Argmax(temp,N);
     for(int k=0;k<N;k++){
         curscore.push_back(temp[maxid[k]]);
         curlabel.push_back(maxid[k]);
     }
     for (int m=0;m<N;m++){
         for(int n=0;n<N;n++){
             pathscore.push_back(prevscore[m]+curscore[n]);
         }
     }
     maxpathid = Argmax(pathscore,N);
     for(int q=0;q<N;q++){
         newpred.push_back(prevpred[maxpathid[q]/N]+" "+std::to_string(curlabel[maxpathid[q]%N]));

     }
     for(int q=0;q<N;q++){
         prevpred[q]=newpred[q];
         prevscore[q]=pathscore[maxpathid[q]];
     }
  }
  for(int i=0;i<N;i++){
     std::istringstream f(newpred[i]);
     string s;
     string prev = "21018";
     string space = "21018";
     int x;
     while (getline(f, s, ' ')){
         if(s!=prev&&s!=space){
             stringstream(s)>>x;
	     preduck[i] = preduck[i] + labels_[x];
             //preduck[i] = preduck[i] + " " + s;
	 }
	 prev = s;
     }
  }
  std::vector<float> finalscore(N);
  for(int i=0;i<N;i++){
	finalscore[i] = prevscore[i];
	for(int j=i+1;j<N;j++){
		if (preduck[i]==preduck[j]){
			preduck[j] = "";
			finalscore[i] = finalscore[i] + prevscore[j];		
		}
	}
  }
  std::vector<int> rank = Argmax(finalscore,N);
  for (int i=0;i<rank.size();i++){
	if (preduck[rank[i]]!=""){
		finalpreds = finalpreds + preduck[rank[i]] + "\n";	
  	}
  }
  return finalpreds;
}

std::vector<float> Classifier::Predict(const cv::Mat& img) {
  boost::shared_ptr<Blob<float> >input_layer = net_->blob_by_name("data");
  
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */


  
  cout << "-------" << endl;
   net_->Reshape();
  //net_->blob_by_name("im_info")->set_cpu_data(im_info);
  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);
  
  Preprocess(img, &input_channels);
 #if 0
  Caffe::set_time_step(22);

	boost::shared_ptr<Layer<float> > indicator_layer =  net_->layer_by_name("indicator");
	LayerParameter* layer_param = const_cast<LayerParameter*>(&(indicator_layer->layer_param()));
	ContinuationIndicatorParameter* param = layer_param->mutable_continuation_indicator_param();
	param->set_time_step(33); 
	cout<< "indicate lay step is " << param->time_step() <<endl;
#endif
  net_->Forward();

  /* Copy the output layer to a std::vector */
  boost::shared_ptr<caffe::Blob<float> > output_layer = net_->blob_by_name("reshape2");
  const float* begin = output_layer->cpu_data(); 
  const float* end = begin + output_layer->count();
  return std::vector<float>(begin, end);
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
	

  Blob<float>* input_layer = net_->input_blobs()[0];
  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void Classifier::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

    /// processing imgs
  cv::Mat sample_resized;
  int height = sample.size().height;
  int width = sample.size().width;
  if(1.0*width/height >= 4) {
    cv::resize(sample, sample_resized, input_geometry_);
  }else {
     cv::Size input_geometry_small = cv::Size(int(32.0*width/height), 32);
     cv::resize(sample, sample_resized, input_geometry_small);
  }
  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);
  cv:: Mat sample_iv;
  //cv::subtract(255, sample_float, sample_iv);
  cv::divide(sample_float, 255, sample_iv, 1);
  cv::Mat final_img(input_geometry_,CV_32FC3,cv::Scalar(0));
  cv::Rect dest_rect(0, 0, sample_iv.cols, sample_iv.rows);
  sample_iv.copyTo(final_img(dest_rect));
  cv::split(final_img, *input_channels);
  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
   == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";

}



