#include <algorithm>
#include <vector>

#include "caffe/layers/rpn_text_layer.hpp"
#include "caffe/util/math_functions.hpp"


using namespace std;



namespace caffe {

    template <typename Dtype>
    void RPNTEXTLayer<Dtype>::LayerSetUp(
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
        
        //anchors_nums_ = 10;
        //anchors_ = new int[anchors_nums_ * 10];
        //memcpy(anchors_, tmp, 10 * 4 * sizeof(int));
      

		feat_stride_ = this->layer_param_.rpn_text_param().feat_stride();  //16
		
		base_size_ = this->layer_param_.rpn_text_param().basesize();   //16

        min_size_ = this->layer_param_.rpn_text_param().boxminsize();  // 16
		
        nms_thresh_ = this->layer_param_.rpn_text_param().nms_thresh();  //0.3
		
		generate_anchors();
		
        anchors_nums_ = gen_anchors_.size() ;

		//cout << "default anchors size is " << anchors_nums_ << endl;
		
        anchors_ = new int[anchors_nums_ * 4];
		
        for (int i = 0; i<gen_anchors_.size(); ++i)
        {
            for (int j = 0; j<gen_anchors_[i].size(); ++j)
            {
                anchors_[i*4+j] = gen_anchors_[i][j];
            }
        }

        top[0]->Reshape(1, 4, 1, 1);
		
        if (top.size() > 1)
        {
            top[1]->Reshape(1, 1, 1, 1);
        }
		
    }


	template <typename Dtype>
    void RPNTEXTLayer<Dtype>::Permute(const int count, Dtype* bottom_data, const bool forward,
    const int* permute_order, const int* old_steps, const int* new_steps,
    const int num_axes, Dtype* top_data) {
    for (int i = 0; i < count; ++i) {
      int old_idx = 0;
      int idx = i;
      for (int j = 0; j < num_axes; ++j) {
        int order = permute_order[j];
        old_idx += (idx / new_steps[j]) * old_steps[order];
        idx %= new_steps[j];
      }
      if (forward) {
        top_data[i] = bottom_data[old_idx];
      } else {
        bottom_data[old_idx] = top_data[i];
      }
    }
}
	
    template <typename Dtype>
    void RPNTEXTLayer<Dtype>::generate_anchors(){
        //generate base anchor
        vector<float> base_anchor;
        base_anchor.push_back(0);
        base_anchor.push_back(0);
        base_anchor.push_back(base_size_ - 1);
        base_anchor.push_back(base_size_ - 1);

        vector<vector<float> >hw_anchors = basic_anchors_();
		
        for (int i = 0; i < hw_anchors.size(); ++i)
        {
        	float h = hw_anchors[i][0];
			float w = hw_anchors[i][1];
			
            vector<float> tmp = scale_anchor(base_anchor,h,w);
            gen_anchors_.push_back(tmp);
        }
    }
	
	template <typename Dtype>
	vector<float> RPNTEXTLayer<Dtype>::scale_anchor(vector<float> anchor,float h,float w)
	{
		vector<float> scaled_anchor;
		
        float x_ctr = (anchor[0]+anchor[2])*0.5 ;
			
		float y_ctr = (anchor[1]+anchor[3])*0.5 ;

		float l_x = x_ctr - w * 0.5 ;
		scaled_anchor.push_back(l_x);
		float l_y = y_ctr - h * 0.5 ;
		scaled_anchor.push_back(l_y);
		float r_x = x_ctr + w * 0.5 ;
		scaled_anchor.push_back(r_x);
		float r_y = y_ctr + h * 0.5 ;
		scaled_anchor.push_back(r_y);
	
		return scaled_anchor;
		
	}

    template <typename Dtype>
    vector<vector<float> > RPNTEXTLayer<Dtype>::basic_anchors_(){
    	vector<vector<float> > result;
		float heights[10] = {11, 16 ,23, 33, 48, 68, 97, 139, 198, 283};
		float widths[1] = {16};

		for (int i =0 ;i < 10 ; i++)
		{
			vector<float> tmp;
			tmp.push_back(heights[i]);
			tmp.push_back(widths[0]);
			result.push_back(tmp);
		}

		return result;
    }

    template <typename Dtype>
    void RPNTEXTLayer<Dtype>::transpose(int a,int b,int c,int d ,shared_ptr<Blob<Dtype> > bottom,shared_ptr<Blob<Dtype> >top)
    {

		int num_axes_ = 4;
		
		vector<int> orders;
		orders.push_back(a);
		orders.push_back(b);
		orders.push_back(c);
		orders.push_back(d);
		
		for (int i = 0; i < num_axes_; ++i) {
		   if (std::find(orders.begin(), orders.end(), i) == orders.end()) {
			 orders.push_back(i);
			 std::cout << " orders("<<i<<")" << orders[i] <<std::endl;
		   }
		 }
		 CHECK_EQ(num_axes_, orders.size());

		vector<int> top_shape(num_axes_, 1);
		Blob<int> permute_order_;
		Blob<int> old_steps_;
		Blob<int> new_steps_;

		permute_order_.Reshape(num_axes_, 1, 1, 1);
		
		old_steps_.Reshape(num_axes_, 1, 1, 1);
		
		new_steps_.Reshape(num_axes_, 1, 1, 1);

		for (int i = 0; i < num_axes_; ++i) 
		{
			permute_order_.mutable_cpu_data()[i] = orders[i];
			
			top_shape[i] = bottom->shape(orders[i]);
		}
		
		top->Reshape(top_shape);

		vector<int> top_shape_;

		for (int i = 0; i < num_axes_; ++i) 
		{
			if (i == num_axes_ - 1) 
			{
				old_steps_.mutable_cpu_data()[i] = 1;
			}
			else 
			{
				old_steps_.mutable_cpu_data()[i] = bottom->count(i + 1);
			}
			
			top_shape_.push_back(bottom->shape(permute_order_.cpu_data()[i]));
		}
	  
		top->Reshape(top_shape_);

		for (int i = 0; i < num_axes_; ++i) 
		{
			if (i == num_axes_ - 1) 
			{
				new_steps_.mutable_cpu_data()[i] = 1;
			}
			else 
			{
				new_steps_.mutable_cpu_data()[i] = top->count(i + 1);
			}
		}


		Dtype* bottom_data = bottom->mutable_cpu_data();
	    Dtype* top_data = top->mutable_cpu_data();
	    const int top_count = top->count();
	    const int* permute_order = permute_order_.cpu_data();
	    const int* old_steps = old_steps_.cpu_data();
	    const int* new_steps = new_steps_.cpu_data();
	    bool forward = true;
	    Permute(top_count, bottom_data, forward, permute_order, old_steps,new_steps, num_axes_, top_data);
		
	}

	
    template <typename Dtype>
    void RPNTEXTLayer<Dtype>::proposal_local_anchor(){
        int length = mymax(map_width_, map_height_);  //57  38
        
        int step = map_width_*map_height_;
		
        Dtype *map_m = new Dtype[length];
		
        for (int i = 0; i < length; ++i)
        {
            map_m[i] = Dtype(i * feat_stride_);
        }
		
        Dtype *shift_x = new Dtype[step];
		
        Dtype *shift_y = new Dtype[step];
				
		local_anchors_->Reshape(1, 1, anchors_nums_*map_height_*map_width_ , 4);
		
		Dtype *a = local_anchors_->mutable_cpu_data();

		int anchors_index = 0;
		
        for (int i = 0; i < map_height_; ++i)
        {
            for (int j = 0; j < map_width_; ++j)
            {
                shift_x[i*map_width_ + j] = map_m[j];
                shift_y[i*map_width_ + j] = map_m[i];

				for (int h = 0; h < anchors_nums_; ++h)
				{
		            caffe_set(1, Dtype(anchors_[h * 4 + 0]), a + anchors_index);
		            caffe_set(1, Dtype(anchors_[h * 4 + 1]), a + anchors_index + 1);
		            caffe_set(1, Dtype(anchors_[h * 4 + 2]), a + anchors_index + 2);
		            caffe_set(1, Dtype(anchors_[h * 4 + 3]), a + anchors_index + 3);
					
		            caffe_add_scalar (1,  shift_x[i*map_width_ + j], a + anchors_index);
		            caffe_add_scalar (1,  shift_x[i*map_width_ + j], a + anchors_index + 2);
		            caffe_add_scalar (1,  shift_y[i*map_width_ + j], a + anchors_index + 1);
		            caffe_add_scalar (1,  shift_y[i*map_width_ + j], a + anchors_index + 3);
					
					anchors_index = anchors_index + 4;
				}
             
            }
        }

#if 0
		
        cout << " anchors index is " << anchors_index << endl;

		cout << local_anchors_->data_at(0,0,0 ,0) << " " << local_anchors_->data_at(0,0,0 ,1) << " " << local_anchors_->data_at(0,0,0 ,2) << " " << local_anchors_->data_at(0,0,0 ,3) << endl;
		cout << local_anchors_->data_at(0,0,1 ,0) << " " << local_anchors_->data_at(0,0,1 ,1) << " " << local_anchors_->data_at(0,0,1 ,2) << " " << local_anchors_->data_at(0,0,1 ,3) << endl;
		cout << local_anchors_->data_at(0,0,2 ,0) << " " << local_anchors_->data_at(0,0,2 ,1) << " " << local_anchors_->data_at(0,0,2 ,2) << " " << local_anchors_->data_at(0,0,2 ,3) << endl;
		cout << local_anchors_->data_at(0,0,3 ,0) << " " << local_anchors_->data_at(0,0,3 ,1) << " " << local_anchors_->data_at(0,0,3 ,2) << " " << local_anchors_->data_at(0,0,3 ,3) << endl;
		cout << local_anchors_->data_at(0,0,4 ,0) << " " << local_anchors_->data_at(0,0,4 ,1) << " " << local_anchors_->data_at(0,0,4 ,2) << " " << local_anchors_->data_at(0,0,4 ,3) << endl;

		cout << local_anchors_->data_at(0,0,anchors_nums_*map_height_*map_width_ -1 ,0) << " " << local_anchors_->data_at(0,0,anchors_nums_*map_height_*map_width_ -1 ,1) << " " << local_anchors_->data_at(0,0,anchors_nums_*map_height_*map_width_ -1 ,2) << " " << local_anchors_->data_at(0,0,anchors_nums_*map_height_*map_width_ -1 ,3) << endl;
#endif 
    }

    template<typename Dtype>
    void RPNTEXTLayer<Dtype>::clip_boxes(vector<abox>& aboxes)
    {
        float localMinSize = min_size_; //16
        aboxes.clear();
        int map_height = proposals_->height();
        const Dtype *box = proposals_->cpu_data();
		
        const Dtype *score = proposals__score_->cpu_data();

        int offset_x1, offset_y1, offset_x2, offset_y2, offset_s;
		
        for (int i  = 0; i < map_height; ++i)
        {
            
            offset_x1 = (i * 4 + 0);
			
            offset_y1 = (i * 4 + 1);
			
            offset_x2 = (i * 4 + 2);
			
            offset_y2 = (i * 4 + 3);
			
            offset_s = i;

            Dtype width = box[offset_x2] - box[offset_x1] + 1;
			
			//std::cout << "local width is " << width << std::endl;
			
            if ( width == localMinSize )
            {
                abox tmp;
                tmp.batch_ind = 0;
                tmp.x1 = box[offset_x1] ;
                tmp.y1 = box[offset_y1] ;
                tmp.x2 = box[offset_x2] ;
                tmp.y2 = box[offset_y2] ;
                tmp.x1 = mymin(mymax(tmp.x1, 0), src_width_);
                tmp.y1 = mymin(mymax(tmp.y1, 0), src_height_);
                tmp.x2 = mymin(mymax(tmp.x2, 0), src_width_);
                tmp.y2 = mymin(mymax(tmp.y2, 0), src_height_);
                tmp.score = score[offset_s];
                aboxes.push_back(tmp);
            }                
            
        }
    }

    template<typename Dtype>
    void RPNTEXTLayer<Dtype>::apply_deltas_to_anchors(){
		
        Dtype * boxes_delta_ptr = reshape_box_->mutable_cpu_data();
		
        Dtype * anchors_ptr = local_anchors_->mutable_cpu_data();

		Blob<Dtype> global_coords; 
		
		global_coords.Reshape(1,1,reshape_box_->count()/2,2);
	
		Dtype * global_coords_ptr = global_coords.mutable_cpu_data();

		proposals_->Reshape(1, 1, local_anchors_->height(), 4);
		
		Dtype * proposals_ptr = proposals_->mutable_cpu_data();
		

		shared_ptr<Blob<Dtype> > anchor_y_ctr;

		anchor_y_ctr.reset(new Blob<Dtype>());

		anchor_y_ctr->Reshape(1, 1, local_anchors_->height(), 1);

		Dtype *anchor_y_ctr_ptr = anchor_y_ctr->mutable_cpu_data();
		
		shared_ptr<Blob<Dtype> > anchor_h;

		anchor_h.reset(new Blob<Dtype>());

		anchor_h->Reshape(1, 1, local_anchors_->height(), 1);

		Dtype *anchor_h_ptr = anchor_h->mutable_cpu_data();

		Dtype *tem_ptr =new Dtype(1);

		caffe_set(1, Dtype(0.5),  tem_ptr);

		for (int i = 0; i < local_anchors_->height(); i++)
		{

			caffe_add(1, anchors_ptr + (i * 4 + 1), anchors_ptr + (i * 4 + 3), anchor_y_ctr_ptr + i); 
			
			caffe_mul(1, tem_ptr, anchor_y_ctr_ptr + i, anchor_y_ctr_ptr + i); 			//anchor_y_ctr

			
			caffe_set(1, *(anchors_ptr + (i * 4 + 3)),  anchor_h_ptr + i); 
			
			caffe_axpy(1, Dtype(-1), anchors_ptr + (i * 4 + 1), anchor_h_ptr + i);

			caffe_add_scalar(1, Dtype(1),  anchor_h_ptr + i); 								//anchor_h 


			caffe_exp(1, boxes_delta_ptr + (i * 2 + 1), global_coords_ptr + (i * 2 + 1));
			
			caffe_mul(1, anchor_h_ptr + i, global_coords_ptr + (i * 2 + 1), global_coords_ptr + (i * 2 + 1));  //global_coords[:, 1]
			
			
			caffe_mul(1, boxes_delta_ptr + (i * 2 + 0), anchor_h_ptr + i, global_coords_ptr + (i * 2 + 0)); 

			caffe_axpy(1, Dtype(1), anchor_y_ctr_ptr + i, global_coords_ptr + (i * 2 + 0));

			caffe_axpy(1, Dtype(-0.5), global_coords_ptr + (i * 2 + 1), global_coords_ptr + (i * 2 + 0));  //global_coords[:, 0]

			caffe_set(1, *(anchors_ptr + (i * 4 + 0)),  proposals_ptr + (i * 4 + 0)); 

			caffe_set(1, *(global_coords_ptr + (i * 2 + 0)),  proposals_ptr + (i * 4 + 1)); 

			caffe_set(1, *(anchors_ptr + (i * 4 + 2)),  proposals_ptr + (i * 4 + 2)); 

			caffe_set(1, *(global_coords_ptr + (i * 2 + 1)),  proposals_ptr + (i * 4 + 3)); 

			caffe_axpy(1, Dtype(1), global_coords_ptr + (i * 2 + 0), proposals_ptr + (i * 4 + 3));
		
		}


		if (NULL!= tem_ptr)
		{
			delete tem_ptr;
			tem_ptr = NULL;
		}
		


    }




    template<typename Dtype>
    void RPNTEXTLayer<Dtype>::nms(std::vector<abox> &input_boxes, float nms_thresh){
        std::vector<float>vArea(input_boxes.size());
        for (int i = 0; i < input_boxes.size(); ++i)
        {
            vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
                * (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
        }
        for (int i = 0; i < input_boxes.size(); ++i)
        {
            for (int j = i + 1; j < input_boxes.size();)
            {
                float xx1 = std::max(input_boxes[i].x1, input_boxes[j].x1);
                float yy1 = std::max(input_boxes[i].y1, input_boxes[j].y1);
                float xx2 = std::min(input_boxes[i].x2, input_boxes[j].x2);
                float yy2 = std::min(input_boxes[i].y2, input_boxes[j].y2);
                float w = std::max(float(0), xx2 - xx1 + 1);
                float   h = std::max(float(0), yy2 - yy1 + 1);
                float   inter = w * h;
                float ovr = inter / (vArea[i] + vArea[j] - inter);
                if (ovr >= nms_thresh)
                {
                    input_boxes.erase(input_boxes.begin() + j);
                    vArea.erase(vArea.begin() + j);
                }
                else
                {
                    j++;
                }
            }
        }
    }


    template<typename Dtype>
    void RPNTEXTLayer<Dtype>::get_proposal_scores(){
    
       // int channel = m_score_->channels();
		
        int height = m_score_->height();
		
        int width = m_score_->width();
		
		//int num = m_score_->num();
		
		int step = height * width;

		Dtype * a = m_score_->mutable_cpu_data() + step*anchors_nums_ ;
		
		proposals__score_->Reshape(1, 1, proposals__score_->count() , 1);
		
		Dtype * b =  proposals__score_->mutable_cpu_data() ;
		
		for (int i = 0; i < proposals__score_->count(); ++i)
		{
			
			caffe_set(1,*a,b);
			a++;
			b++;

		}
		
	}

    template <typename Dtype>
    void RPNTEXTLayer<Dtype>::Forward_cpu(
        const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {

        map_width_ = bottom[1]->width();    	// 51
        
        map_height_ = bottom[1]->height();  	// 38
        
		/////////////////////////////////////////////////////////////////////
		
        //get sores ，前面anchors_nums_个位bg的得分，后面anchors_nums_为fg得分，我们需要的是后面的。
        
        m_score_->CopyFrom(*(bottom[0]),false,true);
		
		reshape_score_->Reshape(m_score_->num(),m_score_->channels() / 2 , m_score_->height() , m_score_->width());

		Dtype *reshape_score_ptr = reshape_score_->mutable_cpu_data(); 
		
		Dtype *m_score__ptr = m_score_->mutable_cpu_data() + anchors_nums_*map_height_*map_width_; 

		for (int i =0 ; i < reshape_score_->count(); i++)
		{
			caffe_set(1, *m_score__ptr, reshape_score_ptr);
			
			m_score__ptr++;
			
			reshape_score_ptr++;
		
		}
		

		transpose(0, 2, 3, 1,reshape_score_,proposals__score_);

		proposals__score_->Reshape(1, 1, proposals__score_->count() , 1);

        //get boxs_delta。
        
        m_box_->CopyFrom(*(bottom[1]), false, true);

        //get im_info

        src_height_ = bottom[2]->data_at(0, 0,0,0);
		
        src_width_ = bottom[2]->data_at(0, 1,0,0);

		src_scale_ = bottom[2]->data_at(0, 2, 0, 0);

        proposal_local_anchor();
		
		transpose(0, 2, 3, 1,m_box_,reshape_box_);

		reshape_box_->Reshape(1, 1, reshape_box_->count() / 2  , 2);


        apply_deltas_to_anchors();

        vector<abox>aboxes;

        clip_boxes(aboxes);

        //nms(aboxes,nms_thresh_);
		
        top[0]->Reshape(aboxes.size(),4,1,1);
		
        Dtype *top0 = top[0]->mutable_cpu_data();
		
        for (int i = 0; i < aboxes.size(); ++i)
        {
            top0[0] = aboxes[i].x1;
            top0[1] = aboxes[i].y1; 
            top0[2] = aboxes[i].x2;
            top0[3] = aboxes[i].y2;
            top0 += top[0]->offset(1);
        }
		
        if (top.size()>1)
        {
            top[1]->Reshape(aboxes.size(), 1,1,1);
			
            Dtype *top1 = top[1]->mutable_cpu_data();
            for (int i = 0; i < aboxes.size(); ++i)
            {
                top1[0] = aboxes[i].score;
                top1 += top[1]->offset(1);
            }
        } 
		
    }

#ifdef CPU_ONLY
        STUB_GPU(RPNTEXTLayer);
#endif

    INSTANTIATE_CLASS(RPNTEXTLayer);
    REGISTER_LAYER_CLASS(RPNTEXT);

}  // namespace caffe
