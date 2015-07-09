#include <vector>
#include <cstdio>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"


namespace caffe {

template <typename Dtype>
void LabelRemapLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  int n = this->layer_param_.label_remap_param().new_label_size();
  label_map.reset(new SyncedMemory(n * sizeof(int)));
  new_label = new int[n];
  printf("###### n = %d\n", n); 
  for (int i = 0; i < n; i++) {
    new_label[i] = this->layer_param_.label_remap_param().new_label(i);
    printf("###### new_label[%d] = %d\n", i, 
        this->layer_param_.label_remap_param().new_label(i)); 
  }
  label_map->set_cpu_data(new_label);
}

template <typename Dtype>
void LabelRemapLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const int* label_map_data = (int*) label_map->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
    top_data[i] = static_cast<Dtype>(
        label_map_data[static_cast<int>(bottom_data[i])]);
  }
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(LabelRemapLayer, Forward);
#endif

INSTANTIATE_CLASS(LabelRemapLayer);
REGISTER_LAYER_CLASS(LabelRemap);

}  // namespace caffe
