#ifndef CAFFE_PEDESTRIAN_TRIPLET_LOSS_LAYER_HPP_
#define CAFFE_PEDESTRIAN_TRIPLET_LOSS_LAYER_HPP_
#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/loss_layer.hpp"
#include "caffe/proto/caffe.pb.h"
namespace caffe {
template <typename Dtype>
  class PedestrianTripletLossLayer : public LossLayer<Dtype> {
   public:
    explicit PedestrianTripletLossLayer(const LayerParameter& param)
    : LossLayer<Dtype>(param)/*, diff_()*/ {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
    virtual inline int ExactNumBottomBlobs() const { return 2; }
    virtual inline int ExactNumTopBlobs() const { return 1; }
    virtual inline const char* type() const { return "PedestrianTripletLoss"; }
    /**
     * Unlike most loss layers, in the PedestrianTripletLossLayer we can backpropagate
     * to the first three inputs.
     */
    virtual inline bool AllowForceBackward(const int bottom_index) const {
      return bottom_index != 1;
    }

   protected:
    /// @copydoc PedestrianTripletLossLayer
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
    /*
     \begin{eqnarray}
     \mathcal{L}_{tri}(s_i,s_j,s_k) = max(0,1-\frac{||f(x_i)-f(x_k)||_2^2}{||f(x_i)-f(x_j)||_2^2+m})
     \end{eqnarray}®
     where $ f(x) $ is the input of the loss layer for sample $ x $ and m is the margin for triplet.
     Denote that $D_{ij}=||f(x_i)-f(x_j)||_2^2$ and $D_{ik}=||f(x_i)-f(x_k)||_2^2$,
     so the partial differential equations for the input of triplet loss layer are:
     \begin{eqnarray}
     \dfrac{\partial \mathcal{L}_{tri}}{\partial f(x_i)}=
     &\frac{D_{ik}(f(x_i)-f(x_j))-(D_{ij}+m)(f(x_i)-f(x_k))}{(D_{ij}+m)^2} \nonumber \\
     \dfrac{\partial \mathcal{L}_{tri}}{\partial f(x_j)}=
     &\frac{D_{ik}(f(x_j)-f(x_i))}{(D_{ij}+m)^2} \nonumber \\
     \dfrac{\partial \mathcal{L}_{tri}}{\partial f(x_k)}=
     &\frac{f(x_i)-f(x_k)}{D_{ij}+m}
     \end{eqnarray}®
     */
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down,
                              const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down,
                              const vector<Blob<Dtype>*>& bottom);
    Blob<Dtype> diff_all;  // cached for backward pa
    Blob<Dtype> diff_pa;
    Blob<Dtype> diff_an;
    Blob<Dtype> diff_np;  // cached for backward pas
    Blob<Dtype> diff_sq_pa;
    Blob<Dtype> diff_sq_an;

    //Blob<Dtype> diff_;  // cached for backward pass

    int hardTripletCnt;
  };

}  // namespace caffe

#endif  // CAFFE_PEDESTRIAN_TRIPLET_LOSS_LAYER_HPP_
