#include <string>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/filler.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

using boost::scoped_ptr;

template <typename TypeParam>
class ImageStackLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  ImageStackLayerTest()
      : backend_(ImageStackParameter_DB_LEVELDB),
        blob_top_data_(new Blob<Dtype>()),
        blob_top_label_(new Blob<Dtype>()),
        seed_(1701) {}
  virtual void SetUp() {
    filename_.reset(new string());
    MakeTempDir(filename_.get());
    *filename_ += "/db";
    blob_top_vec_.push_back(blob_top_data_);
    blob_top_vec_.push_back(blob_top_label_);
  }

  //TODO: fill the test case with real data.
  void Fill(const bool unique_pixels, ImageStackParameter_DB backend) {
    backend_ = backend;
    LOG(INFO) << "Using temporary dataset " << *filename_;
    scoped_ptr<db::DB> db(db::GetDB(backend));
    db->Open(*filename_, db::NEW);
    scoped_ptr<db::Transaction> txn(db->NewTransaction());
    for (int i = 0; i < 5; ++i) {
      DatumVector datum_vector;
      for (int n = 0; n < 5; ++n) {
        Datum* datum = datum_vector.add_data();
        datum->set_label(i);
        datum->set_channels(3);
        datum->set_height(2);
        datum->set_width(3);
        std::string* data = datum->mutable_data();
        for (int j = 0; j < 18; ++j) {
          int datum = unique_pixels ? j : i;
          data->push_back(static_cast<uint8_t>(datum));
        }
      }
      stringstream ss;
      ss << i;
      string out;
      CHECK(datum_vector.SerializeToString(&out));
      txn->Put(ss.str(), out);
    }
    txn->Commit();
    db->Close();
  }

  void TestRead() {
    LayerParameter param;
    param.set_phase(TRAIN);
    ImageStackParameter* image_stack_param = param.mutable_image_stack_param();
    image_stack_param->set_batch_size(5);
    image_stack_param->set_stack_size(5);
    image_stack_param->set_is_flow(true);
    image_stack_param->set_source(filename_->c_str());
    image_stack_param->set_backend(backend_);
    for (int n = 0; n < 4; ++n) {
      image_stack_param->add_subtract_mean(false);
    }
    image_stack_param->add_subtract_mean(false);
    std::cout << "subtract_mean " << image_stack_param->subtract_mean(1);

    ImageStackLayer<Dtype> layer(param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    EXPECT_EQ(blob_top_data_->num(), 5);
    EXPECT_EQ(blob_top_data_->channels(), 10);
    EXPECT_EQ(blob_top_data_->height(), 2);
    EXPECT_EQ(blob_top_data_->width(), 3);
    EXPECT_EQ(blob_top_label_->num(), 5);
    EXPECT_EQ(blob_top_label_->channels(), 1);
    EXPECT_EQ(blob_top_label_->height(), 1);
    EXPECT_EQ(blob_top_label_->width(), 1);

    for (int iter = 0; iter < 100; ++iter) {
      layer.Forward(blob_bottom_vec_, blob_top_vec_);
      for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(i, blob_top_label_->cpu_data()[i]);
      }
      for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 12; ++j) {
          //EXPECT_DOUBLE_EQ(0.0, blob_top_data_->cpu_data()[i * 60 + j]) 
          //  << "debug: iter " << iter << " n 1 i " << i << " j " << j;
          EXPECT_DOUBLE_EQ(i + ((double)i / 10), 
              blob_top_data_->cpu_data()[i * 60 + j])
              << "debug: iter " << iter << " n 1 i " << i << " j " << j;
        }
      }
    }
  } 

  virtual ~ImageStackLayerTest() { delete blob_top_data_; delete blob_top_label_; }

  ImageStackParameter_DB backend_;
  shared_ptr<string> filename_;
  Blob<Dtype>* const blob_top_data_;
  Blob<Dtype>* const blob_top_label_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  int seed_;
};

TYPED_TEST_CASE(ImageStackLayerTest, TestDtypesAndDevices);

TYPED_TEST(ImageStackLayerTest, TestReadLevelDB) {
  const bool unique_pixels = false;  // all pixels the same; images different
  this->Fill(unique_pixels, ImageStackParameter_DB_LEVELDB);
  this->TestRead();
}

TYPED_TEST(ImageStackLayerTest, TestReadLMDB) {
  const bool unique_pixels = false;  // all pixels the same; images different
  this->Fill(unique_pixels, ImageStackParameter_DB_LMDB);
  this->TestRead();
}

}  // namespace caffe
