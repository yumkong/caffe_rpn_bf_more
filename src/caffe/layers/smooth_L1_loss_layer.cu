// --------------------------------------------------------
// Fast R-CNN
// Copyright (c) Microsoft. All rights reserved.
// Written by Ross Girshick, 2015.
// Licensed under the BSD 2-clause "Simplified" license.
// See LICENSE in the Fast R-CNN project root for license
// information.
// --------------------------------------------------------

#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {
  template <typename Dtype>
  __global__ void SmoothL1ForwardGPU(const int n, const Dtype* in, Dtype* out) {
    // f(x) = 0.5 * x^2    if |x| < 1
    //        |x| - 0.5    otherwise
    CUDA_KERNEL_LOOP(index, n) {
      Dtype val = in[index];
      Dtype abs_val = abs(val);
      if (abs_val < 1) {
        out[index] = 0.5 * val * val;
      }
      else {
        out[index] = abs_val - 0.5;
      }
    }
  }

  template <typename Dtype>
  __global__ void kernel_channel_sum(const int num, const int channels,
    const int spatial_dim, const Dtype* data, Dtype* channel_sum) {
    CUDA_KERNEL_LOOP(index, num * spatial_dim) {
      int n = index / spatial_dim;  // index
      int s = index % spatial_dim;  //  0
      Dtype sum = 0;
	  // channels = 8, currently
      for (int c = 0; c < channels; ++c) {
        sum += data[(n * channels + c) * spatial_dim + s];
      }
      channel_sum[index] = sum;
    }
  }

  template <typename Dtype>
  void SmoothL1LossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    int count = bottom[0]->count();
    caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),
      bottom[1]->gpu_data(),
      diff_.mutable_gpu_data());    // d := b0 - b1
    if (has_weights_) {
      caffe_gpu_mul(
        count,
        bottom[2]->gpu_data(),
        diff_.gpu_data(),
        diff_.mutable_gpu_data());  // d := w * (b0 - b1)
    }
    SmoothL1ForwardGPU<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
      count, diff_.gpu_data(), errors_.mutable_gpu_data());
    CUDA_POST_KERNEL_CHECK;

    
    Dtype loss;
    caffe_gpu_asum(count, errors_.gpu_data(), &loss);
    int spatial_dim = diff_.height() * diff_.width();

    //0308 added: sum of all weights value as normalizer
    Dtype valid_count = -1;
    const int nthreads = bottom[2]->count(); //outer_num_ * inner_num_;
	Dtype* counts = bottom[2]->mutable_gpu_data();
    if (normalization_ == LossParameter_NormalizationMode_VALID) {
        caffe_gpu_asum(nthreads, counts, &valid_count);
    }
    // 0308 added: if all weights are zeros, set as invalid
    if (valid_count == 0)
    { valid_count = 1; }

    Dtype pre_fixed_normalizer = this->layer_param_.loss_param().pre_fixed_normalizer();
    top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_,
      pre_fixed_normalizer, valid_count);
    //LOG(INFO) << "SmoothL1 Forward: normalization_ = " << normalization_ << ", pre_fixed_normalizer = " << pre_fixed_normalizer;
    // Output per-instance loss
    if (top.size() >= 2) {
      kernel_channel_sum<Dtype> << <CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS >> >
        (outer_num_, bottom[0]->channels(), inner_num_, errors_.gpu_data(),
          top[1]->mutable_gpu_data());
    }
  }

  template <typename Dtype>
  __global__ void SmoothL1BackwardGPU(const int n, const Dtype* in, Dtype* out) {
    // f'(x) = x         if |x| < 1
    //       = sign(x)   otherwise
    CUDA_KERNEL_LOOP(index, n) {
      Dtype val = in[index];
      Dtype abs_val = abs(val);
      if (abs_val < 1) {
        out[index] = val;
      }
      else {
        out[index] = (Dtype(0) < val) - (val < Dtype(0));
      }
    }
  }

  template <typename Dtype>
  void SmoothL1LossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    int count = diff_.count();
    SmoothL1BackwardGPU<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
      count, diff_.gpu_data(), diff_.mutable_gpu_data());
    CUDA_POST_KERNEL_CHECK;
    for (int i = 0; i < 2; ++i) {
      if (propagate_down[i]) {
        const Dtype sign = (i == 0) ? 1 : -1;
        int spatial_dim = diff_.height() * diff_.width();
        
	//0308 added: sum of all weights value as normalizer
        Dtype valid_count = -1;
		//Dtype valid_count2 = -1;
		//Dtype valid_count3 = -1;
		Dtype* counts = bottom[2]->mutable_gpu_data();
        const int nthreads = bottom[2]->count(); //outer_num_ * inner_num_;
        if (normalization_ == LossParameter_NormalizationMode_VALID) {
			//LOG(INFO) << "Valid has come";
            caffe_gpu_asum(nthreads, counts, &valid_count);
			//caffe_gpu_asum(bottom[2]->count(), bottom[2]->cpu_data(), &valid_count2);
			//caffe_gpu_asum(bottom[2]->count(), bottom[2]->mutable_gpu_data(), &valid_count3);
        }
		//LOG(INFO) << "SmoothL1 BP: outer_num_ = " << outer_num_ << ", inner_num_ = " << inner_num_;
		//LOG(INFO) << "SmoothL1 BP: bot2 cnt = " << bottom[2]->count() << ", valid_count = " << valid_count;
		//LOG(INFO) << "valid_count2 = " << valid_count2 << ", valid_count3 = " << valid_count3;
	    // 0308 added: if all weights are zeros, set as invalid
        if (valid_count == 0)
	    { valid_count = 1; }

        Dtype pre_fixed_normalizer = this->layer_param_.loss_param().pre_fixed_normalizer();

        Dtype normalizer =  get_normalizer(normalization_, pre_fixed_normalizer, valid_count);
		//LOG(INFO) << "SmoothL1 BP: normalizer = " << normalizer <<", normalization_ = " << normalization_<<", pre_fixed_normalizer = " << pre_fixed_normalizer;
        Dtype alpha = sign * top[0]->cpu_diff()[0] / normalizer;

        caffe_gpu_axpby(
          bottom[i]->count(),              // count
          alpha,                           // alpha
          diff_.gpu_data(),                // x
          Dtype(0),                        // beta
          bottom[i]->mutable_gpu_diff());  // y
      }
    }
  }

	INSTANTIATE_LAYER_GPU_FUNCS(SmoothL1LossLayer);

}  // namespace caffe
