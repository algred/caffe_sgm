#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
FlowDataLayer<Dtype>::~FlowDataLayer<Dtype>() {
  this->JoinPrefetchThread();
}

template <typename Dtype>
void FlowDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Read the file with filenames and labels
  const string& source = this->layer_param_.flow_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string filename;
  int label;
  while (infile >> filename >> label) {
    lines_.push_back(std::make_pair(filename, label));
  } 
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  lines_id_ = 0; 
  // Read an image, and use it to initialize the top blob.
  cv::Mat cv_img = ReadImageToCVMat(lines_[lines_id_].first, 0, 0 , true);
  const int channels = this->layer_param_.flow_data_param().stack_size() * 2;
  const int height = cv_img.rows;
  const int width = cv_img.cols;
  // image
  const int crop_size = this->layer_param_.transform_param().crop_size();
  const int batch_size = this->layer_param_.flow_data_param().batch_size();
  // flow
  flow_field_ = shared_ptr<Blob<Dtype> >(new Blob<Dtype>());
  flow_stack_ = shared_ptr<Blob<Dtype> >(new Blob<Dtype>());
  flow_field_->Reshape(1, 2, height, width);
  flow_stack_->Reshape(1, channels, height, width);
  
  if (crop_size > 0) {
    top[0]->Reshape(batch_size, channels, crop_size, crop_size);
    this->prefetch_data_.Reshape(batch_size, channels, crop_size, crop_size);
    this->transformed_data_.Reshape(1, channels, crop_size, crop_size);
  } else {
    top[0]->Reshape(batch_size, channels, height, width);
    this->prefetch_data_.Reshape(batch_size, channels, height, width);
    this->transformed_data_.Reshape(1, channels, height, width);
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  top[1]->Reshape(batch_size, 1, 1, 1);
  this->prefetch_label_.Reshape(batch_size, 1, 1, 1);
}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void FlowDataLayer<Dtype>::InternalThreadEntry() {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double decompress_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(this->prefetch_data_.count());
  CHECK(this->transformed_data_.count());
  FlowDataParameter flow_data_param = this->layer_param_.flow_data_param();
  const int batch_size = flow_data_param.batch_size();
  const int stack_size = flow_data_param.stack_size();
  const int height = flow_field_->height();
  const int width = flow_field_->width();
  const int data_dim = height * width * 2;

  Dtype* prefetch_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* prefetch_label = this->prefetch_label_.mutable_cpu_data();
  Dtype* flow_stack_data = flow_stack_->mutable_cpu_data();

  // datum scales
  const int lines_size = lines_.size();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    CHECK_GT(lines_size, lines_id_ + stack_size - 1);
    prefetch_label[item_id] = lines_[lines_id_].second;
    for (int flow_id = 0; flow_id < stack_size; ++flow_id) {
      // reads a compressed flow field.
      timer.Start();
      cv::Mat cv_img = ReadImageToCVMat(lines_[lines_id_].first,
          height, width, true);
      CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
      read_time += timer.MicroSeconds();

      // Decompress the flow.
      timer.Start();
      flow_field_->set_cpu_data(flow_stack_data + data_dim * flow_id);
      Decompress(cv_img, flow_field_.get());
      decompress_time += timer.MicroSeconds();
      lines_id_++;
    }
    // Apply transformations (mirror, crop...) to the flow stack.
    int offset = this->prefetch_data_.offset(item_id);
    this->transformed_data_.set_cpu_data(prefetch_data + offset);
    this->data_transformer_->Transform(flow_stack_.get(), 
        &(this->transformed_data_));
    trans_time += timer.MicroSeconds();

    // go to the next iter
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "     Decompress time: " << decompress_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

template<typename Dtype>
void FlowDataLayer<Dtype>::Decompress(const cv::Mat& cv_img,
                                       Blob<Dtype>* transformed_blob) {
  const int img_channels = cv_img.channels();
  const int img_height = cv_img.rows;
  const int img_width = cv_img.cols;

  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int count = height * width;

  const bool subtract_mean = this->layer_param_.
    flow_data_param().subtract_mean();

  CHECK_EQ(channels, 2) << "Flow fileds must have 2 channels";
  CHECK_EQ(img_channels, 3) << "Flow Image must have 3 channels";
  CHECK_EQ(height, img_height);
  CHECK_EQ(width, img_width); 

  CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();
  int u_index, v_index;
  Dtype total_u = 0.0;
  Dtype total_v = 0.0;
  for (int h = 0; h < height; ++h) {
    const uchar* ptr = cv_img.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < width; ++w) {
      // Computes the fractions. 
      Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
      Dtype u_frac = static_cast<Dtype>(floor(static_cast<double>(pixel) / 10.0));
      Dtype v_frac = (pixel - 10 * u_frac) / 10.0;
      u_frac = u_frac / 10.0;
      
      // Horizontal flow.
      v_index = (height + h) * width + w;
      pixel = static_cast<Dtype>(ptr[img_index++]);
      transformed_data[v_index] = pixel - 127.0;

      // Vertical flow.
      u_index = h * width + w;
      pixel = static_cast<Dtype>(ptr[img_index++]);
      transformed_data[u_index] = pixel - 127.0;

      // Adds in the fraction part.
      transformed_data[u_index] += u_frac;
      transformed_data[v_index] += v_frac;

      if (subtract_mean) {
        // Computes the total.
        total_u += transformed_data[u_index];
        total_v += transformed_data[v_index];
      }
    }
  }
  // Subtracts the mean flow vector if required.
  if (subtract_mean) {
    Dtype mean_u = total_u / count, mean_v = total_v / count;
    caffe_add_scalar(height * width, (Dtype)(-1.0 * mean_u), transformed_data);
    caffe_add_scalar(height * width, (Dtype)(-1.0 * mean_v), 
        transformed_data + height * width);
// 
//     for (int h = 0; h < height; ++h) {
//       for (int w = 0; w < width; ++w) { 
//         transformed_data[h*width + w] -= mean_u;
//         transformed_data[(height + h)*width + w] -= mean_v;
//       }
//     }
  }
}
INSTANTIATE_CLASS(FlowDataLayer);
REGISTER_LAYER_CLASS(FlowData);

}  // namespace caffe
