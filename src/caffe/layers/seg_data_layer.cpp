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

    cv_image = ReadImageToCVMat( image_path, new_height, new_width, true );
    cv_label = ReadLabelToCVMat( label_path, new_height, new_width );
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
          //LOG(INFO) << top_data[top_index];
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
        //if( top_label[top_index]>20 && top_label[top_index]!=255 ) LOG(INFO) << top_label[top_index];
      }
    }
    sample_cnt_++;
    if( sample_cnt_ >= image_database_.size() ) {
      sample_cnt_ = 0;
      ShuffleImages();
    }
  }
}

INSTANTIATE_CLASS(SegDataLayer);
REGISTER_LAYER_CLASS(SegData);

} // namespace caffe
/*
    for (int c=1; c<=num_class;c++) {
      bool is_class = false;
      cv::Mat cv_mask = cv::Mat::zeros( cv_label.rows, cv_label.cols, CV_8UC1);
      for (int h = 0; h < cv_label.rows; ++h) {
        const uchar* label_ptr = cv_label.ptr<uchar>(h);
        const uchar* mask_ptr = cv_mask.ptr<uchar>(h);
        int label_index = 0;
        for (int w = 0; w < cv_label.cols; ++w) {
          Dtype pixel = static_cast<Dtype>(label_ptr[label_index++]);
          if(pixel==c) {
            is_class = true;
            cv_mask.at<uchar>(h,w) = 255;
          }
        }
     }
     if( is_class ) {
      // blob detection in a binary label image
      cv::Mat cv_cropped_img;
      cv::Mat cv_cropped_label;
      std::vector<std::vector<cv::Point> > contours;
      cv::findContours( cv_mask, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE );
      //Draw the contours
      cv::Mat scale_map(cv_label.size(), CV_8UC1, cv::Scalar(0));
      cv::Scalar colors[5];
      colors[0] = cv::Scalar(128);  // scale 1 
      colors[1] = cv::Scalar(128);  // scale 2
      colors[2] = cv::Scalar(128);  // scale 3
      colors[3] = cv::Scalar(128);  // scale 4
      colors[4] = cv::Scalar(128);  // scale 5
      for (size_t idx = 0; idx < contours.size(); idx++) {
        double area = cv::contourArea(contours[idx], false);
        cv::Rect bbox = cv::boundingRect(contours[idx]);
        int nw = (bbox.width*10)/7;
        int nh = (bbox.height*10)/7;
        double obj_scale = area/(cv_label.rows*cv_label.cols);
        LOG(INFO) << "scale: " << obj_scale;
        LOG(INFO) << "pos_x: " << bbox.x + bbox.width/2;
        LOG(INFO) << "pos_y: " << bbox.y + bbox.height/2;
        LOG(INFO) << "new_w: " << nw;
        LOG(INFO) << "new_h: " << nh;
        int ih = cv_label.rows;
        int iw = cv_label.cols;
        int x1 = bbox.x + bbox.width/2 - nw/2;
        int y1 = bbox.y + bbox.height/2- nh/2;
        int x2 = bbox.x + bbox.width/2+ nw/2;
        int y2 = bbox.y + bbox.height/2+ nh/2;
	    // compute margin in case bbox is larger than image size.
	    int margin_l_x = 0 - x1;
	    int margin_t_y = 0 - y1;
	    int margin_r_x = x2 - iw + 1;
	    int margin_b_y = y2 - ih + 1;
	    if ( margin_l_x > 0 || margin_t_y > 0 || margin_r_x > 0 || margin_b_y > 0 ) {
		  if( x1 < 0 ) x1 = 0; else margin_l_x = 0;
		  if( y1 < 0 ) y1 = 0; else margin_t_y = 0;
		  if( x2 >= iw ) x2 = iw-1; else margin_r_x = 0;
		  if( y2 >= ih ) y2 = ih-1; else margin_b_y = 0;
		  cv::Rect roi(x1, y1, x2-x1+1, y2-y1+1);
		  cv::Mat cv_sub_img = cv_image(roi);
		  cv::Mat cv_sub_label = cv_label(roi);
		  // mean padding
		  cv::Scalar value = cv::Scalar( mean_values_[0], mean_values_[1], mean_values_[2] );
		  cv::Scalar value_l = cv::Scalar( 255 );
      	  cv::copyMakeBorder( cv_sub_img, cv_cropped_img, margin_t_y, margin_b_y, margin_l_x, margin_r_x, cv::BORDER_CONSTANT, value );
      	  cv::copyMakeBorder( cv_sub_label, cv_cropped_label, margin_t_y, margin_b_y, margin_l_x, margin_r_x, cv::BORDER_CONSTANT, value_l );
		  // warping
    	  cv::resize(cv_cropped_img, cv_cropped_img, cv_crop_size, 0, 0, cv::INTER_LINEAR);
    	  cv::resize(cv_cropped_label, cv_cropped_label, cv_crop_size, 0, 0, cv::INTER_LINEAR);
	    } else { // if inner region in an image
		  cv::Rect roi(x1, y1, x2-x1+1, y2-y1+1);
		  cv_cropped_img = cv_image(roi);
		  cv_cropped_label = cv_label(roi);
    	  cv::resize(cv_cropped_img, cv_cropped_img, cv_crop_size, 0, 0, cv::INTER_LINEAR);
    	  cv::resize(cv_cropped_label, cv_cropped_label, cv_crop_size, 0, 0, cv::INTER_LINEAR);
        }
    cv::Mat label_binary;
    cv::threshold(cv_cropped_label, label_binary, 0.0, 255.0, cv::THRESH_BINARY);
      cv::namedWindow("image",1);
	  cv::imshow("image", cv_image);
      cv::namedWindow("cimage",1);
	  cv::imshow("cimage", cv_cropped_img);
      cv::namedWindow("clabel",1);
	  cv::imshow("clabel", label_binary);
	  cv::waitKey(0);

        cv::drawContours(scale_map, contours, idx, colors[0], CV_FILLED);
        cv::rectangle(scale_map, bbox, cv::Scalar(255));
      }*/
