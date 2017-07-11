#include <algorithm>
#include <vector>


#include "caffe/layers/rpn_text_layer.hpp"
#include "caffe/util/math_functions.hpp"

using namespace std;

namespace caffe 
{

	template <typename Dtype>
	__global__ void Permute_Cuda(const int nthreads, Dtype* const bottom_data, const bool forward, const int* permute_order,const int* old_steps, const int* new_steps, const int num_axes,Dtype* const top_data) 
	{
		CUDA_KERNEL_LOOP(index, nthreads) 
		{
			int temp_idx = index;
			int old_idx = 0;
			for (int i = 0; i < num_axes; ++i) 
			{
				int order = permute_order[i];
				old_idx += (temp_idx / new_steps[i]) * old_steps[order];
				temp_idx %= new_steps[i];
			}
			if (forward) 
			{
				top_data[index] = bottom_data[old_idx];
			} 
			else 
			{
				bottom_data[old_idx] = top_data[index];
			}
		}
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
		
		for (int i = 0; i < num_axes_; ++i) 
		{
			if (std::find(orders.begin(), orders.end(), i) == orders.end()) 
			{
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
			permute_order_.mutable_gpu_data()[i] = orders[i];
			
			top_shape[i] = bottom->shape(orders[i]);
		}
		
		top->Reshape(top_shape);

		vector<int> top_shape_;

		for (int i = 0; i < num_axes_; ++i) 
		{
			if (i == num_axes_ - 1) 
			{
				old_steps_.mutable_gpu_data()[i] = 1;
			}
			else 
			{
				old_steps_.mutable_gpu_data()[i] = bottom->count(i + 1);
			}
			
			top_shape_.push_back(bottom->shape(permute_order_.gpu_data()[i]));
		}
	  
		top->Reshape(top_shape_);

		for (int i = 0; i < num_axes_; ++i) 
		{
			if (i == num_axes_ - 1) 
			{
				new_steps_.mutable_gpu_data()[i] = 1;
			}
			else 
			{
				new_steps_.mutable_gpu_data()[i] = top->count(i + 1);
			}
		}


		Dtype* bottom_data = bottom->mutable_gpu_data();
	    Dtype* top_data = top->mutable_gpu_data();
	    const int top_count = top->count();
	    const int* permute_order = permute_order_.gpu_data();
	    const int* old_steps = old_steps_.gpu_data();
	    const int* new_steps = new_steps_.gpu_data();
	    bool forward = true;
	    Permute_Cuda<Dtype><<<CAFFE_GET_BLOCKS(top_count), CAFFE_CUDA_NUM_THREADS>>>(top_count, bottom_data, forward, permute_order, old_steps,new_steps, num_axes_, top_data);
		
	}



    template <typename Dtype>
    __global__ void initialize_score(Dtype *reshape_score_ptr,Dtype *m_score__ptr,int count)
   {
   		CUDA_KERNEL_LOOP(index, count)
   		{
			reshape_score_ptr[index] = m_score__ptr[index];
		}
   }

    template <typename Dtype>
    __global__ void initialize_commom_ptr(int count,Dtype *map_m,int feat_stride_)
    {
   		CUDA_KERNEL_LOOP(index, count)
   		{
   			map_m[index] = Dtype(index * feat_stride_);
		}
    }

    
	template <typename Dtype>
    __global__ void initialize_local_anchors(int map_height_,int map_width_,Dtype *shift_x ,Dtype *shift_y,Dtype *a,Dtype *map_m ,int anchors_nums_,int *anchors_)
    {
    	int anchors_index = 0;
    	
   		CUDA_KERNEL_LOOP(i, map_height_)
   		{
   		
			for (int j = 0; j < map_width_; ++j)
			{
				shift_x[i*map_width_ + j] = map_m[j];
				shift_y[i*map_width_ + j] = map_m[i];

				for (int h = 0; h < anchors_nums_; ++h)
				{
					*(a + anchors_index) = Dtype(anchors_[h * 4 + 0]);
					
					*(a + anchors_index + 1) = Dtype(anchors_[h * 4 + 1]);

					*(a + anchors_index + 2) = Dtype(anchors_[h * 4 + 2]);

					*(a + anchors_index + 3) = Dtype(anchors_[h * 4 + 3]);

					*(a + anchors_index) =  *(a + anchors_index) + shift_x[i*map_width_ + j];

					*(a + anchors_index + 2) = *(a + anchors_index + 2) + shift_x[i*map_width_ + j];

					*(a + anchors_index + 1) = *(a + anchors_index + 1) + shift_y[i*map_width_ + j];

					*(a + anchors_index + 3) = *(a + anchors_index + 3) +  shift_y[i*map_width_ + j];

					anchors_index = anchors_index + 4;
				}

			}
			
		}
    }

	
   	template <typename Dtype>
    void RPNTEXTLayer<Dtype>::proposal_local_anchor()
    {
        int length = mymax(map_width_, map_height_);  //57  38
        
        int step = map_width_*map_height_;
		
        Dtype *map_m = new Dtype[length];

        initialize_commom_ptr<Dtype> <<<CAFFE_GET_BLOCKS(length), CAFFE_CUDA_NUM_THREADS>>>(length,map_m,feat_stride_);
		
        Dtype *shift_x = new Dtype[step];
		
        Dtype *shift_y = new Dtype[step];
				
		local_anchors_->Reshape(1, 1, anchors_nums_*map_height_*map_width_ , 4);
		
		Dtype *a = local_anchors_->mutable_gpu_data();
		
		initialize_local_anchors<Dtype> <<<CAFFE_GET_BLOCKS(map_height_), CAFFE_CUDA_NUM_THREADS>>>(map_height_,map_width_,shift_x ,shift_y,a,map_m,anchors_nums_,anchors_);

    }

	template <typename Dtype>
    __global__ void get_deltas_proposal(int count,Dtype *tem_ptr,Dtype *anchor_y_ctr_ptr,Dtype *anchor_h_ptr, Dtype * anchors_ptr,Dtype * global_coords_ptr ,Dtype * proposals_ptr,Dtype * boxes_delta_ptr)
    {
   		CUDA_KERNEL_LOOP(i, count)
   		{

   			*(anchor_y_ctr_ptr + i) =  *(anchors_ptr + (i * 4 + 1)) + *(anchors_ptr + (i * 4 + 3));
   			
			//caffe_add(1, anchors_ptr + (i * 4 + 1), anchors_ptr + (i * 4 + 3), anchor_y_ctr_ptr + i); 

			*(anchor_y_ctr_ptr + i) =  *(tem_ptr) * (*( anchor_y_ctr_ptr + i));

			//caffe_mul(1, tem_ptr, anchor_y_ctr_ptr + i, anchor_y_ctr_ptr + i); 			//anchor_y_ctr

			*(anchor_h_ptr + i) = *(anchors_ptr + (i * 4 + 3));
			
			//caffe_set(1, *(anchors_ptr + (i * 4 + 3)),  anchor_h_ptr + i); 

			*(anchor_h_ptr + i) = Dtype(-1) * (*(anchors_ptr + (i * 4 + 1))) + *(anchor_h_ptr + i);

			//caffe_axpy(1, Dtype(-1), anchors_ptr + (i * 4 + 1), anchor_h_ptr + i);
			
			*(anchor_h_ptr + i) = *(anchor_h_ptr + i) + Dtype(1);
			
			//caffe_add_scalar(1, Dtype(1),  anchor_h_ptr + i); 								//anchor_h 


			*(global_coords_ptr + (i * 2 + 1)) = *(boxes_delta_ptr + (i * 2 + 1));

			//caffe_exp(1, boxes_delta_ptr + (i * 2 + 1), global_coords_ptr + (i * 2 + 1));
			
			*(global_coords_ptr + (i * 2 + 1)) = *(anchor_h_ptr + i) *(*( global_coords_ptr + (i * 2 + 1)));
			
			//caffe_mul(1, anchor_h_ptr + i, global_coords_ptr + (i * 2 + 1), global_coords_ptr + (i * 2 + 1));  //global_coords[:, 1]


			*(global_coords_ptr + (i * 2 + 0)) = *(boxes_delta_ptr + (i * 2 + 0)) * (*( anchor_h_ptr + i));

			//caffe_mul(1, boxes_delta_ptr + (i * 2 + 0), anchor_h_ptr + i, global_coords_ptr + (i * 2 + 0)); 


			*(global_coords_ptr + (i * 2 + 0)) = *(anchor_y_ctr_ptr + i) * Dtype(1) + *(global_coords_ptr + (i * 2 + 0));

			//caffe_axpy(1, Dtype(1), anchor_y_ctr_ptr + i, global_coords_ptr + (i * 2 + 0));

			*(global_coords_ptr + (i * 2 + 0)) = *(global_coords_ptr + (i * 2 + 1)) * Dtype(-0.5) + *(global_coords_ptr + (i * 2 + 0));

			//caffe_axpy(1, Dtype(-0.5), global_coords_ptr + (i * 2 + 1), global_coords_ptr + (i * 2 + 0));  //global_coords[:, 0]

			*(proposals_ptr + (i * 4 + 0)) = *(anchors_ptr + (i * 4 + 0));

			//caffe_set(1, *(anchors_ptr + (i * 4 + 0)),  proposals_ptr + (i * 4 + 0)); 
			
			*(proposals_ptr + (i * 4 + 1)) = *(global_coords_ptr + (i * 2 + 0));
			
			//caffe_set(1, *(global_coords_ptr + (i * 2 + 0)),  proposals_ptr + (i * 4 + 1)); 

			
			*(proposals_ptr + (i * 4 + 2)) = *(anchors_ptr + (i * 4 + 2));
			
			//caffe_set(1, *(anchors_ptr + (i * 4 + 2)),  proposals_ptr + (i * 4 + 2)); 

			*(proposals_ptr + (i * 4 + 3)) = *(global_coords_ptr + (i * 2 + 1));

	
			//caffe_set(1, *(global_coords_ptr + (i * 2 + 1)),  proposals_ptr + (i * 4 + 3)); 

			*(proposals_ptr + (i * 4 + 3)) = *(global_coords_ptr + (i * 2 + 0)) * Dtype(1) + *(proposals_ptr + (i * 4 + 3));

			//caffe_axpy(1, Dtype(1), global_coords_ptr + (i * 2 + 0), proposals_ptr + (i * 4 + 3));

   		}
   		
    }

    template<typename Dtype>
    void RPNTEXTLayer<Dtype>::apply_deltas_to_anchors(){
		
        Dtype * boxes_delta_ptr = reshape_box_->mutable_gpu_data();
		
        Dtype * anchors_ptr = local_anchors_->mutable_gpu_data();

		Blob<Dtype> global_coords; 
		
		global_coords.Reshape(1,1,reshape_box_->count()/2,2);
	
		Dtype * global_coords_ptr = global_coords.mutable_gpu_data();

		proposals_->Reshape(1, 1, local_anchors_->height(), 4);
		
		Dtype * proposals_ptr = proposals_->mutable_gpu_data();
		

		shared_ptr<Blob<Dtype> > anchor_y_ctr;

		anchor_y_ctr.reset(new Blob<Dtype>());

		anchor_y_ctr->Reshape(1, 1, local_anchors_->height(), 1);

		Dtype *anchor_y_ctr_ptr = anchor_y_ctr->mutable_gpu_data();
		
		shared_ptr<Blob<Dtype> > anchor_h;

		anchor_h.reset(new Blob<Dtype>());

		anchor_h->Reshape(1, 1, local_anchors_->height(), 1);

		Dtype *anchor_h_ptr = anchor_h->mutable_gpu_data();

		Dtype *tem_ptr =new Dtype(1);

		caffe_gpu_set(1, Dtype(0.5),  tem_ptr);

		get_deltas_proposal<Dtype> <<<CAFFE_GET_BLOCKS(local_anchors_->height()), CAFFE_CUDA_NUM_THREADS>>>(local_anchors_->height(),tem_ptr,anchor_y_ctr_ptr,anchor_h_ptr, anchors_ptr,global_coords_ptr ,proposals_ptr,boxes_delta_ptr);


		if (NULL!= tem_ptr)
		{
			delete tem_ptr;
			tem_ptr = NULL;
		}
		


    }

    template<typename Dtype>
    void RPNTEXTLayer<Dtype>::clip_boxes(vector<abox>& aboxes)
    {
        float localMinSize = min_size_; 
        
        aboxes.clear();
        
        int map_height = proposals_->height();
        
        const Dtype *box = proposals_->gpu_data();
		
        const Dtype *score = proposals__score_->gpu_data();

        int offset_x1, offset_y1, offset_x2, offset_y2, offset_s;
		
        for (int i  = 0; i < map_height; ++i)
        {
            
            offset_x1 = (i * 4 + 0);
			
            offset_y1 = (i * 4 + 1);
			
            offset_x2 = (i * 4 + 2);
			
            offset_y2 = (i * 4 + 3);
			
            offset_s = i;

            Dtype width = box[offset_x2] - box[offset_x1] + 1;
			
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

	
	template <typename Dtype>
	void RPNTEXTLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top)
	{
		
		cout<< "rpn text forward gpu" <<endl;

		map_width_ = bottom[1]->width();    	

		map_height_ = bottom[1]->height(); 

		m_score_->CopyFrom(*(bottom[0]),false,true);

		reshape_score_->Reshape(m_score_->num(),m_score_->channels() / 2 , m_score_->height() , m_score_->width());

		Dtype *reshape_score_ptr = reshape_score_->mutable_gpu_data(); 

		Dtype *m_score__ptr = m_score_->mutable_gpu_data() + anchors_nums_*map_height_*map_width_; 

		cout << "reshape score count is " <<reshape_score_->count() << endl;
		
		initialize_score<Dtype> <<<CAFFE_GET_BLOCKS(reshape_score_->count()), CAFFE_CUDA_NUM_THREADS>>>(reshape_score_ptr,m_score__ptr,reshape_score_->count());

		transpose(0, 2, 3, 1,reshape_score_,proposals__score_);

		proposals__score_->Reshape(1, 1, proposals__score_->count() , 1);

		//get boxs_delta¡£
        
        m_box_->CopyFrom(*(bottom[1]), false, true);

        //get im_info

        src_height_ = bottom[2]->data_at(0, 0,0,0);
		
        src_width_ = bottom[2]->data_at(0, 1,0,0);

		src_scale_ = bottom[2]->data_at(0, 2, 0, 0);

        proposal_local_anchor();
		
		transpose(0, 2, 3, 1,m_box_,reshape_box_);

		//cout << "before reshape box is " << reshape_box_->shape_string()<<endl;

		reshape_box_->Reshape(1, 1, reshape_box_->count() / 2  , 2);

		//cout << "after reshape box is " << reshape_box_->shape_string()<<endl;

		apply_deltas_to_anchors();

		vector<abox>aboxes;

        clip_boxes(aboxes);

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

        cout << "rpntext Forward_gpu finished." << endl;

	}

	INSTANTIATE_LAYER_GPU_FUNCS(RPNTEXTLayer);


}  // namespace caffe
