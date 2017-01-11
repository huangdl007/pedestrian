#include <algorithm>
#include <vector>
#include <iostream>

#include "caffe/layer.hpp"
#include "caffe/layers/pedestriantriplet_loss_layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#define mdy_debug 0
namespace caffe {

template <typename Dtype>
void PedestrianTripletLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    //LOG(INFO) << "bottom asum_data: "  << bottom[0]->num() << "===" << bottom[0]->sumsq_data();
    int dim = bottom[0]->count()/bottom[0]->num();
    //LOG(INFO) << "dim is " << dim;
    const Dtype* feature_data = bottom[0]->gpu_data();
    const Dtype* label_data = bottom[1]->cpu_data();

    PedestrianTripletLossParameter pedestriantriplet_loss_param = this->layer_param_.pedestriantriplet_loss_param();
    const bool has_ignore_label = pedestriantriplet_loss_param.has_ignore_label();
    const bool use_ignore_as_negative = pedestriantriplet_loss_param.use_ignore_as_negative();
    const int ignore_label = pedestriantriplet_loss_param.ignore_label();
    const float margin = pedestriantriplet_loss_param.margin();


    int batch_size = bottom[0]->num();
    hardTripletCnt = 0;
    Dtype loss(0.0);
    caffe_gpu_set(bottom[0]->count(), Dtype(0.0), diff_all.mutable_gpu_data());
    
#if mdy_debug
    int tmp;
    std::cout << " pause " << std::endl;
    std::cin >> tmp;

 LOG(INFO) << " start  online ";
#endif
    //for ( int j = 0; j < dim; j++ ) {
    //    LOG(INFO) << cpu_data[j];
    //}
    int PosEndIndex = 0;
    for(int i = 0; i < bottom[1]->count(); i++) {
        if(int(label_data[i]) == 0) {
            PosEndIndex = i - 1;
        }
    }
    for ( int i = 0; i < PosEndIndex; i++ ) {
        int curLabel = label_data[i];
        if (has_ignore_label && ignore_label == curLabel ) {
//            std::cout << "ignore label " << curLabel  << std::endl;
            continue;
        }

        int A = i;
        for (int P = A+1; P < PosEndIndex; P++) {
            // P - A
            caffe_gpu_sub(dim,
                    feature_data + P*dim,
                    feature_data + A*dim,
                    diff_pa.mutable_gpu_data());

            Dtype dotPA;
            caffe_gpu_dot(dim,
                    diff_pa.gpu_data(),
                    diff_pa.gpu_data(),
                    &dotPA);

            for ( int N = PosEndIndex; N < batch_size; N++ ) {
                if ( N == A || N == P ) {
                    continue;
                }
         //       LOG(INFO) << " A P N : " << A << " " <<  P << " "<<  N ;

                int negative_label = label_data[N];
                if (has_ignore_label && ignore_label == negative_label
                        && !use_ignore_as_negative) {
    //                std::cout << "ignore label " << negative_label  << std::endl;
                    continue;
                }

                // A - N
                caffe_gpu_sub(dim,
                        feature_data + A*dim,
                        feature_data + N*dim,
                        diff_an.mutable_gpu_data());

                Dtype dotNA;
                caffe_gpu_dot(dim,
                        diff_an.gpu_data(),
                        diff_an.gpu_data(),
                        &dotNA);
    #if mdy_debug
               // LOG(INFO) << " try a  triplet " << hardTripletCnt << " " << dotPA << " "<< dotNA ;
    #endif
            //    float margin = 0.5;
                if (dotPA + margin > dotNA) {

    #if mdy_debug
                    LOG(INFO) << " find a hard triplet "
                        << A << " "
                        << P << " "
                        << N << " " << dotPA << " "<< dotNA ;
    #endif
                    hardTripletCnt += 1;
                    loss += dotPA - dotNA + margin;
                    // N - P
                    caffe_gpu_sub(dim,
                            feature_data + N*dim,
                            feature_data + P*dim,
                            diff_np.mutable_gpu_data());
                    // for A
                    caffe_gpu_add(dim,
                            diff_np.gpu_data(),
                            diff_all.gpu_data() + A*dim,
                            diff_all.mutable_gpu_data() + A*dim);
                    // for P
                    caffe_gpu_add(dim,
                            diff_pa.gpu_data(),
                            diff_all.gpu_data() + P*dim,
                            diff_all.mutable_gpu_data() + P*dim);
                    // for N
                    caffe_gpu_add(dim,
                            diff_an.gpu_data(),
                            diff_all.gpu_data() + N*dim,
                            diff_all.mutable_gpu_data() + N*dim);
    #if mdy_debug
                    std::cout << " check for loss \n";
                    for ( int i = 0; i < 512; i++ ) {
                        std::cout << diff_all.cpu_data()[i] << " ";
                    }
                    std::cout << "\n";
    #endif
                }
            }
        }
    }
#if mdy_debug
    LOG(INFO) << " over  online ";
#endif
    if (hardTripletCnt != 0 ) {
        top[0]->mutable_cpu_data()[0] = 1.0 * loss
            / static_cast<Dtype>(hardTripletCnt);
//        std::cout << " scale is " << Dtype(1.0)/hardTripletCnt;
//        caffe_gpu_axpby(batch_size*dim,
//        Dtype(1.0)/hardTripletCnt,
//        diff_all.gpu_data(),
//        Dtype(1.0),
//        diff_all.mutable_gpu_data());
//
        caffe_gpu_scal(batch_size*dim,
                Dtype(1.0)/hardTripletCnt,                                                                                                                                                                                                 
                diff_all.mutable_gpu_data());

#if mdy_debug
            std::cout << " after scale \n";
                for ( int i = 0; i < 512; i++ ) {
                    std::cout << diff_all.cpu_data()[i] << " ";
                }
                std::cout << "\n";
#endif
    } else {
        top[0]->mutable_cpu_data()[0] = 1.0 * loss;
    }
}

template <typename Dtype>
void PedestrianTripletLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

     if (propagate_down[0]) {
        const Dtype alpha = top[0]->cpu_diff()[0] /
        static_cast<Dtype>(bottom[0]->num());
#if mdy_debug
        std::cout <<  " bottom count is " << bottom[0]-> count() << std::endl;
        std::cout << "alpha = " << alpha << std::endl;
        for ( int i = 0; i < bottom[0]->count(); i++ ) {
            std::cout << " " << diff_all.cpu_data()[i] << " ";
        }
        std::cout << std::endl;
#endif
        caffe_gpu_axpby(bottom[0]->count(),
                alpha,
                diff_all.gpu_data(),
                Dtype(0),
                bottom[0]->mutable_gpu_diff());
        
     }
}

INSTANTIATE_LAYER_GPU_FUNCS(PedestrianTripletLossLayer);

}  // namespace caffe
