#ifndef CAFFE_RPN_CENTER_LOSS_LAYER_HPP_
#define CAFFE_RPN_CENTER_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/loss_layers.hpp"
//#include "caffe/layers/loss_layer.hpp"

namespace caffe {

template <typename Dtype>
class RpnCenterLossLayer : public LossLayer<Dtype> {
 public:
  explicit RpnCenterLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "RpnCenterLoss"; }
  virtual inline int ExactNumBottomBlobs() const { return 3; } //0716: 2->3
  virtual inline int ExactNumTopBlobs() const { return -1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int M_; // number of batch samples
  int K_; // feature length
  int N_; // number of centers
  // liu@0811 added
  Dtype lr_;
  
  Blob<Dtype> distance_;
  Blob<Dtype> variation_sum_;
};

}  // namespace caffe

#endif  // CAFFE_RPN_CENTER_LOSS_LAYER_HPP_
