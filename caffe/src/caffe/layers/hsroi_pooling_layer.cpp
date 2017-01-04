#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/hsroi_pooling_layer.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {

template <typename Dtype>
void HSROIPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  HSROIPoolingParameter hsroi_pool_param = this->layer_param_.hsroi_pooling_param();

  //CHECK_EQ(hsroi_pool_param.output_width(), 4)
  //    << "output_width must be = 4";
  CHECK_GT(hsroi_pool_param.output_dim(), 0)
      << "output_dim must be > 0";

  output_dim_ = hsroi_pool_param.output_dim();
  output_height_ = 1;
  //output_width_ = hsroi_pool_param.output_width();
  output_width_ = 4;
  spatial_scale_ = hsroi_pool_param.spatial_scale();
  LOG(INFO) << "Spatial scale: " << spatial_scale_;
}

template <typename Dtype>
void HSROIPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  channels_ = bottom[0]->channels();
  CHECK_EQ(channels_, output_dim_*output_height_*output_width_)
      << "input channel number does not match layer parameters";
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  top[0]->Reshape(bottom[1]->num(), output_dim_, output_height_, output_width_);
  mapping_channel_.Reshape(
      bottom[1]->num(), output_dim_, output_height_, output_width_);
}

template <typename Dtype>
void HSROIPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
const vector<Blob<Dtype>*>& top) {
	/*const Dtype* bottom_data = bottom[0]->cpu_data();
	const Dtype* bottom_rois = bottom[1]->cpu_data();

	// number of ROIs
	int num_rois = bottom[1]->num();
	int batch_size = bottom[0]->num();
	int top_count = top[0]->count();
	Dtype* top_data = top[0]->mutable_cpu_data();
	caffe_set(top_count, Dtype(0), top_data);

	// for each ROI R = [batch_index x1 y1 x2 y2]: head&shoulders-sensitive average pool over R
	for(int n = 0; n < num_rois; n++) {
		int roi_batch_ind = bottom_rois[0];
	    int roi_start_w = round(bottom_rois[1] * spatial_scale_);
	    int roi_start_h = round(bottom_rois[2] * spatial_scale_);
	    int roi_end_w = round(bottom_rois[3] * spatial_scale_);
	    int roi_end_h = round(bottom_rois[4] * spatial_scale_);
	    CHECK_GE(roi_batch_ind, 0);
	    CHECK_LT(roi_batch_ind, batch_size);

	    int roi_height = max(roi_end_h - roi_start_h + 1, 1);
    	int roi_width = max(roi_end_w - roi_start_w + 1, 1);

    	const Dtype* batch_data = bottom_data + bottom[0]->offset(roi_batch_ind);

    	// entire region pool
	}*/
	NOT_IMPLEMENTED;
}

template <typename Dtype>
void HSROIPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(HSROIPoolingLayer);
#endif

INSTANTIATE_CLASS(HSROIPoolingLayer);
REGISTER_LAYER_CLASS(HSROIPooling);

}  // namespace caffe