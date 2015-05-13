#include <opencv2/core/core.hpp>

#include <stdint.h>

#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
ImageStackLayer<Dtype>::~ImageStackLayer<Dtype>() {
  this->JoinPrefetchThread();
}

template <typename Dtype>
void ImageStackLayer<Dtype>::DataLayerSetUp(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // Checks the paramters.
  image_stack_param_ = ImageStackParameter(
      this->layer_param_.image_stack_param());
  const int stack_size = image_stack_param_.stack_size();
  if (!image_stack_param_.is_flow()) {
    CHECK(!image_stack_param_.subtract_mean_size() > 0)
      << "Subtract mean is only supported for flow data.";
  }
  CHECK_EQ(image_stack_param_.subtract_mean_size(), stack_size)
      << "Must provide flag for subtract mean for each flow image.";

  // Initialize DB
  db_.reset(db::GetDB(image_stack_param_.backend()));
  db_->Open(image_stack_param_.source(), db::READ);
  cursor_.reset(db_->NewCursor());

  // Check if we should randomly skip a few data points
  if (image_stack_param_.rand_skip()) {
    unsigned int skip = caffe_rng_rand() % image_stack_param_.rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    while (skip-- > 0) {
      cursor_->Next();
    }
  }

  // Read a data point, and use it to initialize the top blob.
  DatumVector datum_vector;
  datum_vector.ParseFromString(cursor_->value());
  CHECK_EQ(datum_vector.data_size(), stack_size) 
    << "Number of images in each sample mush be equal to stack_size";

  bool force_color = image_stack_param_.force_encoded_color();
  Datum datum = datum_vector.data(0);
  int image_nchs;
  if (image_stack_param_.is_flow()) {
    CHECK_EQ(datum.channels(), 3) << "Flow image must have 3 channels";
    image_nchs = 2;
  } else {
    image_nchs = datum.channels();
  }
  int nchs = image_nchs * stack_size;
  std::cout << "width " << datum.width() << " height " << datum.height();
  // image
  const int batch_size = image_stack_param_.batch_size();
  int crop_size = this->layer_param_.transform_param().crop_size();
  if (crop_size > 0) {
    top[0]->Reshape(batch_size, nchs, crop_size, crop_size);
    this->prefetch_data_.Reshape(batch_size, nchs, crop_size, crop_size);
    this->transformed_data_.Reshape(1, nchs, crop_size, crop_size);
  } else {
    top[0]->Reshape(batch_size, nchs, datum.height(), datum.width());
    this->prefetch_data_.Reshape(batch_size, nchs, datum.height(), datum.width());
    this->transformed_data_.Reshape(1, nchs, datum.height(), datum.width());
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
 
  // stack for reading flow data.
  field_ = shared_ptr<Blob<Dtype> >(new Blob<Dtype>());
  stack_ = shared_ptr<Blob<Dtype> >(new Blob<Dtype>());
  field_->Reshape(1, image_nchs, datum.height(), datum.width());
  stack_->Reshape(1, nchs, datum.height(), datum.width());

  // label
  if (this->output_labels_) {
    top[1]->Reshape(batch_size, 1, 1, 1);
    this->prefetch_label_.Reshape(batch_size, 1, 1, 1);
  }
}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void ImageStackLayer<Dtype>::InternalThreadEntry() {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;

  CPUTimer timer;
  CHECK(this->prefetch_data_.count());
  CHECK(this->transformed_data_.count());

  // Reshape on single input batches for inputs of varying dimension.
  const int batch_size = image_stack_param_.batch_size();
  const int stack_size = image_stack_param_.stack_size();
  bool force_color = image_stack_param_.force_encoded_color();
  bool is_flow = image_stack_param_.is_flow();
  const int height = field_->height();
  const int width = field_->width();
  const int image_nchs = field_->channels();
  const int data_dim = height * width * image_nchs; 

  Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables
  Dtype* stack_data = stack_->mutable_cpu_data();
   
  if (this->output_labels_) {
    top_label = this->prefetch_label_.mutable_cpu_data();
  }
  
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    // get a blob
    DatumVector datum_vector;
    datum_vector.ParseFromString(cursor_->value());

    vector<cv::Mat> cv_imgs;
    cv::Mat cv_img;
    for (int image_id = 0; image_id < stack_size; ++image_id) {
      if (force_color) {
        cv_img = DecodeDatumToCVMat(datum_vector.data(image_id), true);
      } else {
        cv_img = DecodeDatumToCVMatNative(datum_vector.data(image_id));
      }
      if (is_flow){
        field_->set_cpu_data(stack_data + data_dim * image_id);
        bool subtract_mean = image_stack_param_.subtract_mean(image_id);
        FlowImageToFlow(cv_img, field_.get(), subtract_mean);
      } else {
        cv_imgs.push_back(cv_img);
      }
    }
    read_time += timer.MicroSeconds();
    timer.Start();

    // Apply data transformations (mirror, scale, crop...)
    int offset = this->prefetch_data_.offset(item_id);
    this->transformed_data_.set_cpu_data(top_data + offset);
    if (is_flow) {
      this->data_transformer_->Transform(stack_.get(), &(this->transformed_data_));
    } else {
      Datum datum;
      CVMatStackToDatum(cv_imgs, &datum, stack_size);
      this->data_transformer_->Transform(datum, &(this->transformed_data_));
    }
    if (this->output_labels_) {
      top_label[item_id] = datum_vector.data(0).label();
    }
    trans_time += timer.MicroSeconds();
  
    // Moves the cursor forward.
    if (image_stack_param_.rand_step()) {
      unsigned int skip = caffe_rng_rand() % image_stack_param_.rand_step();
      while (skip-- > 0) {
        cursor_->Next();
      }
      
    } else {
      cursor_->Next();
    }
    if (!cursor_->valid()) {
      DLOG(INFO) << "Restarting data prefetching from start.";
      cursor_->SeekToFirst();
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(ImageStackLayer);
REGISTER_LAYER_CLASS(ImageStack);

}  // namespace caffe
