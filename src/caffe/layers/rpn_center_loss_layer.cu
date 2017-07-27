#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/rpn_center_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
//void Compute_distance_data_gpu(int nthreads, const int K, const Dtype* bottom,
	//      const Dtype* label,const Dtype* label_weight, const Dtype* center, Dtype* distance) 
//{
  //  for(int index = 0; index < nthreads; ++index) {
__global__ void Compute_distance_data_gpu(int nthreads, const int K, const Dtype* bottom,
	      const Dtype* label,const Dtype* label_weight, const Dtype* center, Dtype* distance) {
    CUDA_KERNEL_LOOP(index, nthreads) {
    int m = index / K;
    //int m = index;
    int k = index % K;  // no use here
    const int label_value = static_cast<int>(label[m]);
	//liu@0716 added
	const int label_weight_value = static_cast<int>(label_weight[m]);
    // distance(i) = x(i) - c_{y(i)}
    //liu@0716 changed
    //distance[index] = bottom[index] - center[label_value * K + k] ;
    // 0717 changed
    if(label_weight_value > 0)
    {
        distance[index] = (bottom[index] - center[label_value * K + k]);
    }
    // feat - center
    //caffe_gpu_sub(K, bottom + m * K, center + label_value * K, distance + m * K);
	// (feat - center) * label_weight (0 or 1)
	//caffe_gpu_scale(K, (Dtype)label_weight_value, distance + m * K, distance + m * K);
  }
}

template <typename Dtype>
__global__ void Compute_center_diff_gpu(int nthreads, const int M, const int K, 
        const Dtype* label,const Dtype* label_weight, const Dtype* distance, Dtype* variation_sum, 
        Dtype* center_diff) {
//void Compute_center_diff_gpu(int nthreads, const int M, const int K, 
//	  const Dtype* label,const Dtype* label_weight, const Dtype* distance, Dtype* variation_sum, 
//	  Dtype* center_diff) {
//  for(int index = 0; index < nthreads; ++index)
  CUDA_KERNEL_LOOP(index, nthreads) 
  {
    int count = 0;
    for (int m = 0; m < M; m++) {
      const int label_value = static_cast<int>(label[m]);
	  const int label_weight_value = static_cast<int>(label_weight[m]);
      if (label_weight_value > 0 && label_value == index) 
	  {
        ++count;
        for (int k = 0; k < K; k++) {
          variation_sum[index * K + k] -= distance[m * K + k];
        }
		//caffe_gpu_sub(K, variation_sum + index * K, distance + m * K, variation_sum + index * K);
      }
    }
    for (int k = 0; k < K; k++) {
      center_diff[index * K + k] = (Dtype)0.05 * variation_sum[index * K + k] /(count + (Dtype)1.);
    }
    //caffe_gpu_scale(K, (Dtype)1./(count + (Dtype)1.), variation_sum + index * K, center_diff + index * K);
  }
}


template <typename Dtype>
void RpnCenterLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int nthreads = M_ * K_;
  // liu 0719: initialized distance_ as 0 each time before using it
  caffe_gpu_set(M_ * K_, Dtype(0), distance_.mutable_gpu_data());
  //int nthreads = M_; // num x hei x wid
  //liu@0716 changed:  added bottom[2]->gpu_data()  (label_weight)
  Compute_distance_data_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
     CAFFE_CUDA_NUM_THREADS>>>(nthreads, K_, bottom[0]->gpu_data(), bottom[1]->gpu_data(),
      							bottom[2]->gpu_data(),
                                this->blobs_[0]->gpu_data(), distance_.mutable_gpu_data());
  Dtype dot;
  caffe_gpu_dot(M_ * K_, distance_.gpu_data(), distance_.gpu_data(), &dot);
  // ### liu: should change M_ to actual count
  Dtype valid_count;
  caffe_gpu_asum(M_, bottom[2]->gpu_data(), &valid_count);
  //Dtype loss = dot / M_ / Dtype(2);
  Dtype loss = dot / valid_count / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
  //LOG(INFO) << "dot = " << dot << ", valid_count = " << valid_count;
  //LOG(INFO) << "M_ = " << M_ << ", K_ = " << K_;
}

template <typename Dtype>
void RpnCenterLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  int nthreads = N_; // 2
  caffe_gpu_set(N_ * K_, Dtype(0), variation_sum_.mutable_gpu_data());
  //caffe_gpu_set(variation_sum_->count(), Dtype(0), variation_sum_.mutable_gpu_data());
  // liu: here do not need "label_weight" because distance is already 0 from forward_gpu computation
  // liu: NONONO: still need label weight to reduce computation
  Compute_center_diff_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, M_, K_, bottom[1]->gpu_data(),
                                bottom[2]->gpu_data(), distance_.gpu_data(), 
                                variation_sum_.mutable_gpu_data(), this->blobs_[0]->mutable_gpu_diff());
  // ### liu: should change M_ to actual count
  Dtype valid_count;
  //caffe_gpu_asum(M_, bottom[2]->gpu_data(), &valid_count);
  caffe_gpu_asum(bottom[2]->count(), bottom[2]->gpu_data(), &valid_count);
  if (propagate_down[0]) {
  	//liu@0716 changed
    //caffe_gpu_scale(M_ * K_, top[0]->cpu_diff()[0] / M_, 
    caffe_gpu_scale(M_ * K_, top[0]->cpu_diff()[0] / valid_count, 
                             distance_.gpu_data(), bottom[0]->mutable_gpu_diff());
  }
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(RpnCenterLossLayer);

}  // namespace caffe
