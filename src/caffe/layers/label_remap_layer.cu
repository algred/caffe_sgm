#include <algorithm>
#include <vector>
#include <cstdio>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {
template <typename Dtype>
__global__ void LabelRemapForward(const int n, const int* label_map_data,
    const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = static_cast<Dtype>(
        label_map_data[static_cast<int>(in[index])]);
    //printf("******%d %d\n", static_cast<int>(in[index]),
    //    label_map_data[static_cast<int>(in[index])]);
  }
}

template <typename Dtype>
void LabelRemapLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  const int* label_map_data = (int*) label_map->gpu_data();
  // NOLINT_NEXT_LINE(whitespace/operators)
  LabelRemapForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, label_map_data, bottom_data, top_data);
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FORWARD(LabelRemapLayer);


}  // namespace caffe
