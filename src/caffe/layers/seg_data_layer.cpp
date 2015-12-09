#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
SegDataLayer<Dtype>::~SegDataLayer<Dtype>() {
}

template <typename Dtype>
void SegDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Seg file format
  // repeated:
  //    image_name
  LOG(INFO) << "Seg data layer:" << std::endl
      << "  image root_folder: "
      << this->layer_param_.seg_data_param().image_root_folder() << std::endl
      << "  label root_folder: "
      << this->layer_param_.seg_data_param().label_root_folder();
  string image_root_folder = this->layer_param_.seg_data_param().image_root_folder();
  string label_root_folder = this->layer_param_.seg_data_param().label_root_folder();

  sample_cnt_ = 0;
  const unsigned int prefetch_rng_seed = caffe_rng_rand();
  prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));

  std::ifstream infile(this->layer_param_.seg_data_param().source().c_str());
  CHECK(infile.good()) << "Failed to open source file "
      << this->layer_param_.seg_data_param().source() << std::endl;
  string linestr, image_path, label_path;
  while (std::getline(infile, linestr)) {
    std::istringstream iss(linestr);
    string imgfn;
    iss >> imgfn;
    //LOG(INFO) << imgfn;
    image_path = image_root_folder + imgfn + ".jpg";
    label_path = label_root_folder + imgfn + ".png";
    image_database_.push_back(std::make_pair(image_path,label_path));
  }
  if (this->layer_param_.seg_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    ShuffleImages();
  }
  LOG(INFO) <<  "# images : " << image_database_.size();
  // prepare blobs' shape
  const int crop_size = this->layer_param_.seg_data_param().crop_size();
  const int batch_size = this->layer_param_.seg_data_param().batch_size();
  // image
  top[0]->Reshape(batch_size,3,crop_size,crop_size);
  LOG(INFO) << "input image data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  top[1]->Reshape(batch_size,1,crop_size,crop_size);
  LOG(INFO) << "input label data size: " << top[1]->num() << ","
      << top[1]->channels() << "," << top[1]->height() << ","
      << top[1]->width();
  // data mean
  has_mean_values_ = this->layer_param_.seg_data_param().mean_value_size() > 0;
  if (has_mean_values_) {
    for (int c = 0; c < this->layer_param_.seg_data_param().mean_value_size(); ++c) {
      mean_values_.push_back(this->layer_param_.seg_data_param().mean_value(c));
    }
    CHECK(mean_values_.size() == 1 || mean_values_.size() == top[0]->channels() ) <<
     "Specify either 1 mean_value or as many as channels: " << top[0]->channels() ;
    if (top[0]->channels() > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < 3; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }
}

template <typename Dtype>
void SegDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(image_database_.begin(), image_database_.end(), prefetch_rng);
}

template <typename Dtype>
unsigned int SegDataLayer<Dtype>::PrefetchRand() {
  CHECK(prefetch_rng_);
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  return (*prefetch_rng)();
}

template <typename Dtype>
void SegDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // seg data layer parameters
  const Dtype scale = this->layer_param_.seg_data_param().scale();
  const int batch_size = this->layer_param_.seg_data_param().batch_size();
  const int new_height = this->layer_param_.seg_data_param().new_height();
  const int new_width = this->layer_param_.seg_data_param().new_width();
  const int crop_size = this->layer_param_.seg_data_param().crop_size();
  const bool mirror = this->layer_param_.seg_data_param().mirror();
  const bool is_shuffle = this->layer_param_.seg_data_param().shuffle();
  //const int stride = this->layer_param_.seg_data_param().stride();
  //const int num_class = this->layer_param_.seg_data_param().num_class();
  // image data
  cv::Mat cv_image;
  // label data
  cv::Mat cv_label; 
  cv::Size cv_crop_size(crop_size,crop_size);
  
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    std::string image_path = image_database_[sample_cnt_].first;
    std::string label_path = image_database_[sample_cnt_].second;
    bool do_mirror = mirror && PrefetchRand() % 2;
    //LOG(INFO) << do_mirror;
	//LOG(INFO) << image_path;
    cv_image = ReadImageToCVMat( image_path, new_height, new_width, true );
    cv_label = ReadLabelToCVMat( label_path, new_height, new_width );
    if(do_mirror) {
      cv::flip(cv_image, cv_image, 1);
      cv::flip(cv_label, cv_label, 1);
    }
/*
    cv::Mat cv_mask = cv::Mat::zeros( cv_label.rows, cv_label.cols, CV_8UC1);
    for (int h = 0; h < cv_label.rows; ++h) {
      const uchar* label_ptr = cv_label.ptr<uchar>(h);
      int label_index = 0;
      for (int w = 0; w < cv_label.cols; ++w) {
        Dtype pixel = static_cast<Dtype>(label_ptr[label_index++]);
        if(pixel!=0) {
          cv_mask.at<uchar>(h,w) = 255;
        }
      }
    } 
	cv::namedWindow("image",1);
	cv::imshow("image", cv_image);
	cv::namedWindow("label",1);
	cv::imshow("label", cv_mask);
	cv::waitKey(0);
*/
    // image
    const int ih = cv_image.rows;
    const int iw = cv_image.cols;
    const int channels = cv_image.channels();
    Dtype* top_data = top[0]->mutable_cpu_data();
    for (int h = 0; h < cv_image.rows; ++h) {
      const uchar* ptr = cv_image.ptr<uchar>(h);
      int img_index = 0;
      for (int w = 0; w < cv_image.cols; ++w) {
        for (int c = 0; c < channels; ++c) {
          int top_index = ((item_id * channels + c) * ih + h) * iw + w;
          Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
          if (this->has_mean_values_) {
            top_data[top_index] = (pixel - this->mean_values_[c]) * scale;
          } else {
            top_data[top_index] = pixel * scale;
          }
        }
      }
    }
    // label
    Dtype* top_label = top[1]->mutable_cpu_data();
    const int lh = cv_label.rows;
    const int lw = cv_label.cols;
    for (int h = 0; h < cv_label.rows; ++h) {
      const uchar* label_ptr = cv_label.ptr<uchar>(h);
      int label_index = 0;
      for (int w = 0; w < cv_label.cols; ++w) {
        int top_index = ((item_id) * lh + h) * lw + w;
        Dtype pixel = static_cast<Dtype>(label_ptr[label_index++]);
        top_label[top_index] = pixel;
      }
    }
    sample_cnt_++;
    if( sample_cnt_ >= image_database_.size() ) {
      sample_cnt_ = 0;
      if (is_shuffle) {
        ShuffleImages();
      }
    }
  }
}

INSTANTIATE_CLASS(SegDataLayer);
REGISTER_LAYER_CLASS(SegData);

} // namespace caffe
