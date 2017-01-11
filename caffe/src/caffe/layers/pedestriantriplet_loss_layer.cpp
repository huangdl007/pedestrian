#include <algorithm>
#include <vector>
#include "caffe/layer.hpp"
#include "caffe/layers/pedestriantriplet_loss_layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include <map>
using std::vector;
namespace caffe {
template <typename Dtype>
void PedestrianTripletLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  LOG(INFO) << " satart up PedestrianTripletLossLayer";
  int dim = bottom[0]->count()/bottom[0]->num();
  int batch_size = bottom[0]->num();
  // limit each person has two pics
  int full_size = 1 ; //(batch_size-2) * batch_size;

  diff_all.Reshape(batch_size,dim,1,1);

  diff_pa.Reshape(full_size, dim, 1, 1);
  diff_sq_pa.Reshape(full_size,1,1,1);

  diff_an.Reshape(full_size, dim, 1, 1);
  diff_sq_an.Reshape(full_size,1,1,1);

  diff_np.Reshape(1, dim, 1, 1);
  LOG(INFO) << " satart up PedestrianTripletLossLayer done";
}

template <typename Dtype>
void PedestrianTripletLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //LOG(INFO) << " reshape PedestrianTripletLossLayer";
  int dim = bottom[0]->count()/bottom[0]->num();
  int batch_size = bottom[0]->num();
  // limit each person has two pics
  int full_size = 1 ; //(batch_size-2) * batch_size;

  diff_all.Reshape(batch_size,dim,1,1);

  diff_pa.Reshape(full_size, dim, 1, 1);
  diff_sq_pa.Reshape(full_size,1,1,1);

  diff_an.Reshape(full_size, dim, 1, 1);
  diff_sq_an.Reshape(full_size,1,1,1);

  diff_np.Reshape(1, dim, 1, 1);
  top[0]->Reshape(1, 1, 1, 1);
  //LOG(INFO) << " reshape PedestrianTripletLossLayer done";
}

template <typename Dtype>
void PedestrianTripletLossLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  // first you should get the batch num
  // push the person togethor // get label
  // get feature
  int dim = bottom[0]->count()/bottom[0]->num();

  const Dtype* feature_data = bottom[0]->cpu_data();
  const Dtype* label_data = bottom[1]->cpu_data();

  PedestrianTripletLossParameter pedestriantriplet_loss_param = this->layer_param_.pedestriantriplet_loss_param();
  const bool has_ignore_label = pedestriantriplet_loss_param.has_ignore_label();
  const bool use_ignore_as_negative = pedestriantriplet_loss_param.use_ignore_as_negative();
  const int ignore_label = pedestriantriplet_loss_param.ignore_label();
  const float margin = pedestriantriplet_loss_param.margin();

  int batch_size = bottom[0]->num();

  // prepare to gen triplet
  // for each person in personList:
  //    cal the distance which ||A - P|| > ||A - N||
  // same person A1 A2 C
  hardTripletCnt = 0;
  Dtype loss(0.0);

  caffe_set(bottom[0]->count(), Dtype(0.0), diff_all.mutable_cpu_diff());

  int PosEndIndex = 0;
  for(int i = 0; i < bottom[1]->count(); i++) {
  	if(int(label_data[i]) == 0) {
  		PosEndIndex = i - 1;
  	}
  }

  for ( int i = 0; i < PosEndIndex; i++ ) {
      int curLabel = label_data[i];
      if (has_ignore_label && ignore_label == curLabel ) {
//          LOG(INFO) << "ignore label cpu " << curLabel;
          continue;
      }
      int A = i;
      for (int P = A+1; P < PosEndIndex; P++) {
	      // P - A
	      caffe_sub(dim,
	              feature_data + P*dim,
	              feature_data + A*dim,
	              diff_pa.mutable_cpu_data());
	      diff_sq_pa.mutable_cpu_data()[0] = caffe_cpu_dot(dim,
	              diff_pa.cpu_data(),
	              diff_pa.cpu_data());

	      for ( int N = PosEndIndex; N < batch_size; N++ ) {
	        if ( N == A || N == P ) {
	            continue;
	        }

	        int negative_label = label_data[N];
	        if (has_ignore_label && ignore_label == negative_label
	                && !use_ignore_as_negative) {
	          continue;
	        }

	        // A - N
	        caffe_sub(dim,
	                feature_data + A * dim,
	                feature_data + N * dim,
	                diff_an.mutable_cpu_data());
	        diff_sq_an.mutable_cpu_data()[0] = caffe_cpu_dot(dim,
	                diff_an.cpu_data(),
	                diff_an.cpu_data());
	      //  float margin = 0.5;
	        if (diff_sq_pa.cpu_data()[0] + margin  > diff_sq_an.cpu_data()[0]) {
	            hardTripletCnt += 1;
	            loss += diff_sq_pa.cpu_data()[0] - diff_sq_an.cpu_data()[0] + margin;
	            // accmulate
	            // if you create loss you should update
	            // N-P
	            caffe_sub(dim,
	                    feature_data + N*dim,
	                    feature_data + P*dim,
	                    diff_np.mutable_cpu_data());
	            // for A
	            caffe_add(dim,
	                    diff_np.cpu_data(),
	                    diff_all.cpu_data() + A*dim,
	                    diff_all.mutable_cpu_data() + A*dim);
	            // for P
	            caffe_add(dim,
	                    diff_pa.cpu_data(),
	                    diff_all.cpu_data() + P *dim,
	                    diff_all.mutable_cpu_data() + P*dim);

	            // for N
	            caffe_add(dim,
	                    diff_an.cpu_data(),
	                    diff_all.cpu_data() + N*dim,
	                    diff_all.mutable_cpu_data() + N*dim);
	        }
	      }
	  }
  }

  if (hardTripletCnt != 0 ) {
    top[0]->mutable_cpu_data()[0] = 1.0 * loss
        / static_cast<Dtype>(hardTripletCnt);
//    caffe_cpu_axpby(batch_size*dim,
//            Dtype(1.0)/hardTripletCnt,
//            diff_all.cpu_data(),
//            Dtype(0.0),
//            diff_all.mutable_cpu_data());
    caffe_scal(batch_size*dim,
            Dtype(1.0)/hardTripletCnt,
            diff_all.mutable_cpu_data());


  } else {
    top[0]->mutable_cpu_data()[0] = 1.0 * loss;
  }
}

template <typename Dtype>
void PedestrianTripletLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                           const vector<bool>& propagate_down,
                                           const vector<Blob<Dtype>*>& bottom) {
    if (propagate_down[0]) {
        const Dtype alpha = top[0]->cpu_diff()[0] /
            static_cast<Dtype>(bottom[0]->num());
        caffe_cpu_axpby(bottom[0]->count(),
                alpha,
                diff_all.cpu_data(),
                Dtype(0),
                bottom[0]->mutable_cpu_diff());
    }

}
#ifdef CPU_ONLY
    STUB_GPU(PedestrianTripletLossLayer);
#endif
    INSTANTIATE_CLASS(PedestrianTripletLossLayer);
    REGISTER_LAYER_CLASS(PedestrianTripletLoss);
}  // namespace caffe
