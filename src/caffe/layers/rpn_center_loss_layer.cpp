#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/rpn_center_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void RpnCenterLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num_output = this->layer_param_.center_loss_param().num_output();  // 10
  N_ = num_output; // 2: face, non-face
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.center_loss_param().axis()); // 1
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_  vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N inner products with dimension CHW are performed.
  K_ = bottom[0]->count(axis); // = chl
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(1);
    // Intialize the weight
    vector<int> center_shape(2);
    center_shape[0] = N_; // 10
    //###liu### need to change for rpn loss layer
    center_shape[1] = K_; // CHW (== 2 for mnist)
    this->blobs_[0].reset(new Blob<Dtype>(center_shape)); // reset ip1 ?
    // fill the weights
    shared_ptr<Filler<Dtype> > center_filler(GetFiller<Dtype>(
        this->layer_param_.center_loss_param().center_filler()));
    center_filler->Fill(this->blobs_[0].get());

  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void RpnCenterLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //CHECK_EQ(bottom[1]->channels(), 1); // chl 
  //CHECK_EQ(bottom[1]->height(), 1); // hei
  //CHECK_EQ(bottom[1]->width(), 1); // wid
  CHECK_EQ(bottom[0]->num(), bottom[1]->num()); // output blob dim[0] == label dim[0]
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels()); // output blob dim[1] == label dim[1] (original hei)
  CHECK_EQ(bottom[0]->height(), bottom[1]->height()); // output blob dim[2] == label dim[2] (original wid)
  CHECK_EQ(bottom[1]->width(), 1); //label's chl
  CHECK_EQ(bottom[2]->width(), 1); // label_weight's chl
  //###liu### change for rpn loss layer
  //M_ = bottom[0]->num();
  M_ = bottom[0]->num() * bottom[0]->channels() * bottom[0]->height(); //
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  LossLayer<Dtype>::Reshape(bottom, top);
  distance_.ReshapeLike(*bottom[0]); // liu: num x hei x wid x chl
  variation_sum_.ReshapeLike(*this->blobs_[0]); // liu: 2 (center_nums) x feat_len
}

template <typename Dtype>
void RpnCenterLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  // liu: 0716 added
  const Dtype* label_weight = bottom[2]->cpu_data();
  const Dtype* center = this->blobs_[0]->cpu_data();
  Dtype* distance_data = distance_.mutable_cpu_data();
  // liu 0716: initialized as 0
  memset(distance_data, (Dtype)0, sizeof(Dtype) * M_ * K_);
  int cnt = 0;
  // the i-th distance_data
  for (int i = 0; i < M_; i++) {
  	// cast from Dtype to int
    const int label_value = static_cast<int>(label[i]);
	//liu added
	const int label_weight_value = static_cast<int>(label_weight[i]);
    // D(i,:) = X(i,:) - C(y(i),:)
    // liu@0716 changed
    if(label_weight_value > 0) // selected fg or bg
    {
         caffe_sub(K_, bottom_data + i * K_, center + label_value * K_, distance_data + i * K_);
		 ++cnt; // number of valid data points in the output feature map
    }
  }
  Dtype dot = caffe_cpu_dot(M_ * K_, distance_.cpu_data(), distance_.cpu_data());
  // 0716 liu
  //Dtype loss = dot / M_ / Dtype(2);
  Dtype loss = dot / cnt / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void RpnCenterLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) 
{
  // liu@0716 added
	Dtype count_sum = (Dtype)0;
  // Gradient with respect to centers
  if (this->param_propagate_down_[0]) {
    const Dtype* label = bottom[1]->cpu_data();
	// liu: 0716 added
    const Dtype* label_weight = bottom[2]->cpu_data();
    Dtype* center_diff = this->blobs_[0]->mutable_cpu_diff();
    Dtype* variation_sum_data = variation_sum_.mutable_cpu_data();
	// liu 0716: initialized as 0
    memset(variation_sum_data, (Dtype)0, sizeof(Dtype) * N_ * K_);
    const Dtype* distance_data = distance_.cpu_data();

    // \sum_{y_i==j}
    caffe_set(N_ * K_, (Dtype)0., variation_sum_.mutable_cpu_data());
	
    for (int n = 0; n < N_; n++) {
      int count = 0;
      for (int m = 0; m < M_; m++) 
	  {
        const int label_value = static_cast<int>(label[m]);
		//0716 added
		const int label_weight_value = static_cast<int>(label_weight[m]);
		// liu@0716 changed 
        //if (label_value == n) 
        if (label_weight_value > 0 && label_value == n) 
		{
          count++;
          caffe_sub(K_, variation_sum_data + n * K_, distance_data + m * K_, variation_sum_data + n * K_);
        }
      }
	  // a * x + y
	  count_sum += count;
      caffe_axpy(K_, (Dtype)1./(count + (Dtype)1.), variation_sum_data + n * K_, center_diff + n * K_);
    }
  }
  // Gradient with respect to bottom data 
  if (propagate_down[0]) {
    caffe_copy(M_ * K_, distance_.cpu_data(), bottom[0]->mutable_cpu_diff());
	//liu 0617 changed
    //caffe_scal(M_ * K_, top[0]->cpu_diff()[0] / M_, bottom[0]->mutable_cpu_diff());
    caffe_scal(M_ * K_, top[0]->cpu_diff()[0] / count_sum, bottom[0]->mutable_cpu_diff());
  }
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
}

#ifdef CPU_ONLY
STUB_GPU(RpnCenterLossLayer);
#endif

INSTANTIATE_CLASS(RpnCenterLossLayer);
REGISTER_LAYER_CLASS(RpnCenterLoss);

}  // namespace caffe
