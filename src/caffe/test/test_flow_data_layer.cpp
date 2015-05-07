#include <map>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename TypeParam>
class FlowDataLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  FlowDataLayerTest():
        blob_top_data_(new Blob<Dtype>()),
        blob_top_label_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    blob_top_vec_.push_back(blob_top_data_);
    blob_top_vec_.push_back(blob_top_label_);
    // Create test input file.
    MakeTempFilename(&filename_);
    std::ofstream outfile(filename_.c_str(), std::ofstream::out);
    LOG(INFO) << "Using temporary file " << filename_;
    for (int i = 0; i < 5; ++i) {
      for (int j = 1; j <= 5; ++j) {
        outfile << EXAMPLES_SOURCE_DIR "images/000" << j << "_flow.png " << i;
      }
    }
    outfile.close();
    
    // Reads the flow data.
    data_dim = 256 * 256 * 5;
    string flowname = string(EXAMPLES_SOURCE_DIR) + "images/flow_u.txt";
    std::ifstream infile(flowname.c_str(), std::ifstream::in);
    flow_u = new Dtype[data_dim];
    for (int i = 0; i < data_dim; i++) {
      infile >> flow_u[i];
    }
    infile.close();
    
    flowname = string(EXAMPLES_SOURCE_DIR) + "images/flow_v.txt";
    infile.open(flowname.c_str(), std::ifstream::in);
    flow_v = new Dtype[data_dim];
    for (int i = 0; i < data_dim; i++) {
      infile >> flow_v[i];
    }
    infile.close();

    flowname = string(EXAMPLES_SOURCE_DIR) + "images/flow_u_ms.txt";
    infile.open(flowname.c_str(), std::ifstream::in);
    flow_u_ms = new Dtype[data_dim];
    for (int i = 0; i < data_dim; i++) {
      infile >> flow_u_ms[i];
    }
    infile.close();

    flowname = string(EXAMPLES_SOURCE_DIR) + "images/flow_v_ms.txt";
    infile.open(flowname.c_str(), std::ifstream::in);
    flow_v_ms = new Dtype[data_dim];
    for (int i = 0; i < data_dim; ++i) {
      infile >> flow_v_ms[i];
    }
    infile.close();
  }

  virtual ~FlowDataLayerTest() {
    delete blob_top_data_;
    delete blob_top_label_;
    delete flow_u;
    delete flow_v;
    delete flow_u_ms;
    delete flow_v_ms;
  }

  string filename_;
  Dtype* flow_u;
  Dtype* flow_v;
  Dtype* flow_u_ms;
  Dtype* flow_v_ms;
  int data_dim;
  Blob<Dtype>* const blob_top_data_;
  Blob<Dtype>* const blob_top_label_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(FlowDataLayerTest, TestDtypesAndDevices);

TYPED_TEST(FlowDataLayerTest, TestRead) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param;
  FlowDataParameter* flow_data_param = param.mutable_flow_data_param();
  flow_data_param->set_batch_size(5);
  flow_data_param->set_stack_size(5);
  flow_data_param->set_source(this->filename_.c_str());
  FlowDataLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 5);
  EXPECT_EQ(this->blob_top_data_->channels(), 10);
  EXPECT_EQ(this->blob_top_data_->height(), 256);
  EXPECT_EQ(this->blob_top_data_->width(), 256);
  EXPECT_EQ(this->blob_top_label_->num(), 5);
  EXPECT_EQ(this->blob_top_label_->channels(), 1);
  EXPECT_EQ(this->blob_top_label_->height(), 1);
  EXPECT_EQ(this->blob_top_label_->width(), 1);
  // Go through the data twice
  for (int iter = 0; iter < 2; ++iter) {
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype* flow_data = this->blob_top_data_->cpu_data();
    int idx = 0;
    int flow_idx = 0;
    for (int n = 0; n < 5; ++n) {
      for (int h = 0; h < 256; ++h) {
        for (int w = 0; w < 256; ++w) {
          flow_idx = n * 256 * 256 + h * 256 + w;
          
          idx = n * 256 * 256 * 2 + h * 256 + w;
          EXPECT_LE(abs(flow_data[idx] - this->flow_u_ms[flow_idx]), 0.1)
            << flow_data[idx] << " " << this->flow_u_ms[flow_idx] 
            << " " << idx << " " << flow_idx << endl;

          idx = n * 256 * 256 * 2 + 256 * 256 + h * 256 + w;
          EXPECT_LE(abs(flow_data[idx] - this->flow_v_ms[flow_idx]), 0.1)
            << flow_data[idx] << " " << this->flow_v_ms[flow_idx] 
            << " " << idx << " " << flow_idx << endl;
        }
      }
    }
    for (int i = 0; i < 5; ++i) {
      EXPECT_EQ(i, this->blob_top_label_->cpu_data()[i]);
    }
  }
}

TYPED_TEST(FlowDataLayerTest, TestNoSubtractMean) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param;
  FlowDataParameter* flow_data_param = param.mutable_flow_data_param();
  flow_data_param->set_batch_size(5);
  flow_data_param->set_subtract_mean(false);
  flow_data_param->set_stack_size(5);
  flow_data_param->set_source(this->filename_.c_str());
  FlowDataLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 5);
  EXPECT_EQ(this->blob_top_data_->channels(), 10);
  EXPECT_EQ(this->blob_top_data_->height(), 256);
  EXPECT_EQ(this->blob_top_data_->width(), 256);
  EXPECT_EQ(this->blob_top_label_->num(), 5);
  EXPECT_EQ(this->blob_top_label_->channels(), 1);
  EXPECT_EQ(this->blob_top_label_->height(), 1);
  EXPECT_EQ(this->blob_top_label_->width(), 1);
  // Go through the data twice
  for (int iter = 0; iter < 2; ++iter) {
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype* flow_data = this->blob_top_data_->cpu_data();
    int idx = 0;
    int flow_idx = 0;
    for (int n = 0; n < 5; ++n) {
      for (int h = 0; h < 256; ++h) {
        for (int w = 0; w < 256; ++w) {
          flow_idx = n * 256 * 256 + h * 256 + w;
          
          idx = n * 256 * 256 * 2 + h * 256 + w;
          EXPECT_LE(abs(flow_data[idx] - this->flow_u[flow_idx]), 0.1)
            << flow_data[idx] << " " << this->flow_u[flow_idx] 
            << " " << idx << " " << flow_idx << endl;

          idx = n * 256 * 256 * 2 + 256 * 256 + h * 256 + w;
          EXPECT_LE(abs(flow_data[idx] - this->flow_v[flow_idx]), 0.1)
            << flow_data[idx] << " " << this->flow_v[flow_idx] 
            << " " << idx << " " << flow_idx << endl;
        }
      }
    }
    for (int i = 0; i < 5; ++i) {
      EXPECT_EQ(i, this->blob_top_label_->cpu_data()[i]);
    }
  }
}
}  // namespace caffe
