#include <opencv2/highgui/highgui_c.h>
#include <stdint.h>

#include <algorithm>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

// caffe.proto > LayerParameter > AttentionDataParameter
//   'source' field specifies the window_file

namespace caffe {

template <typename Dtype>
AttentionDataLayer<Dtype>::~AttentionDataLayer<Dtype>() {
  //this->StopInternalThread();
}

template <typename Dtype>
//void AttentionDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
void AttentionDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // attention_file format
  // repeated:
  //    # image_index
  //    img_path (abs path)
  //    channels
  //    height
  //    width
  //    num_windows
  //    x1 y1 x2 y2 TL BR TLFlip BRFlip ... class_index

  LOG(INFO) << "Attention data layer:" << std::endl
      << "  cache_images: "
      << this->layer_param_.attention_data_param().cache_images() << std::endl
      << "  root_folder: "
      << this->layer_param_.attention_data_param().root_folder();

  cache_images_ = this->layer_param_.attention_data_param().cache_images();
  string root_folder = this->layer_param_.attention_data_param().root_folder();
  num_class_ = this->layer_param_.attention_data_param().num_class();
  CHECK_EQ( 2*num_class_+2, top.size() ); // check configuration
  patch_id_ = 0;
  total_patch_ = 0;
  
  const bool prefetch_needs_rand = this->layer_param_.attention_data_param().mirror();
  if (prefetch_needs_rand) {
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
  } else {
    prefetch_rng_.reset();
  }

  std::ifstream infile(this->layer_param_.attention_data_param().source().c_str());
  CHECK(infile.good()) << "Failed to open window file "
      << this->layer_param_.attention_data_param().source() << std::endl;

  string hashtag;
  int image_index, channels;
  if (!(infile >> hashtag >> image_index)) {
    LOG(FATAL) << "Window file is empty";
  }
  do {
    CHECK_EQ(hashtag, "#");
    // read image path
    string image_path;
    infile >> image_path;
    image_path = root_folder + image_path;
    // read image dimensions
    vector<int> image_size(3);
    infile >> image_size[0] >> image_size[1] >> image_size[2];
	channels = image_size[0];
    image_database_.push_back(std::make_pair(image_path, image_size));

    if (cache_images_) {
      Datum datum;
      if (!ReadFileToDatum(image_path, &datum)) {
        LOG(ERROR) << "Could not open or find file " << image_path;
        return;
      }
      image_database_cache_.push_back(std::make_pair(image_path, datum));
    }
    // read each box
    int num_windows;
    infile >> num_windows;
	LOG(INFO) << image_path << ": " << image_size[0] << "," << image_size[1] << "," << image_size[2];
	LOG(INFO) << num_windows;
	total_patch_ += num_windows;

    for (int i = 0; i < num_windows; ++i) {
      int x1, y1, x2, y2;
      int TL, BR, TLF, BRF, CLS;
	  vector<float> target_info;
	  target_info.push_back(image_index);
      infile >> x1 >> y1 >> x2 >> y2;
	  target_info.push_back(x1);  target_info.push_back(y1);
	  target_info.push_back(x2);  target_info.push_back(y2);
	  //LOG(INFO) << target_info[0] << ", " << target_info[1] << ", " << target_info[2] << ", " << target_info[3];
	  for (int c = 0; c < num_class_; ++c ) {
	    infile >> TL >> BR >> TLF >> BRF;
		target_info.push_back(TL);  target_info.push_back(BR);
		target_info.push_back(TLF);  target_info.push_back(BRF);
	    //LOG(INFO) << target_info[4+2*c] << ", " << target_info[4+2*c+1];
	  }
	  infile >> CLS;
	  target_info.push_back(CLS);
	  //LOG(INFO) << target_info[2*num_class_+4];
	  target_attention_.push_back(target_info); 
    }

	//vector<float> info = target_attention_[image_index];
	//LOG(INFO) << info[1] << ", " << info[2] << ", " << info[3] << ", " << info[4];
	//for (int c = 0; c < num_class_; ++c ) {
	//  LOG(INFO) << info[5+2*c] << ", " << info[5+2*c+1];
	//}
	//LOG(INFO) << info[2*num_class_+5];
	
    if (image_index % 100 == 0) {
      LOG(INFO) << "num: " << image_index << " "
          << image_path << " "
          << image_size[0] << " "
          << image_size[1] << " "
          << image_size[2] << " "
          << "attention data parsing... " << num_windows;
    }
  } while (infile >> hashtag >> image_index);

  LOG(INFO) << "Number of images  : " << image_index+1;
  LOG(INFO) << "Number of windows : " << total_patch_;
  // prepare blobs' shape
  // image
  const int input_size = this->layer_param_.attention_data_param().input_size();
  CHECK_GT(input_size, 0);
  const int batch_size = this->layer_param_.attention_data_param().batch_size();
  top[0]->Reshape(batch_size, channels, input_size, input_size);
  //this->transformed_data_.Reshape(batch_size, channels, input_size, input_size);
  //for (int i = 0; i < this->PREFETCH_COUNT; ++i)
  //  this->prefetch_[i].data_.Reshape(
  //      batch_size, channels, input_size, input_size);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  
  // label
  vector<int> label_shape(1, batch_size);
  for (int c = 0; c < 2*num_class_+1; ++c ) {
    top[c+1]->Reshape(label_shape);
    //for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    //  this->prefetch_[i].label_.Reshape(label_shape);
    //}
  }

  // data mean
  has_mean_values_ = this->layer_param_.attention_data_param().mean_value_size() > 0;
  LOG(INFO) << has_mean_values_;
  if (has_mean_values_) {
    for (int c = 0; c < this->layer_param_.attention_data_param().mean_value_size(); ++c) {
      mean_values_.push_back(this->layer_param_.attention_data_param().mean_value(c));
    }
    CHECK(mean_values_.size() == 1 || mean_values_.size() == channels) <<
     "Specify either 1 mean_value or as many as channels: " << channels;
    if (channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }
}

template <typename Dtype>
unsigned int AttentionDataLayer<Dtype>::PrefetchRand() {
  CHECK(prefetch_rng_);
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  return (*prefetch_rng)();
}

template <typename Dtype>
void AttentionDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  const Dtype scale = this->layer_param_.attention_data_param().scale();
  const int batch_size = this->layer_param_.attention_data_param().batch_size();
  const int input_size = this->layer_param_.attention_data_param().input_size();
  const bool mirror = this->layer_param_.attention_data_param().mirror();
  
  cv::Size cv_crop_size(input_size, input_size);
  int curr_image_id, prev_image_id;
  curr_image_id = prev_image_id = 0;
  bool image_reload = true;
  cv::Mat cv_img; // original image
  cv::Mat cv_cropped_img; // patch image

  for (int item_id = 0; item_id < batch_size; ++item_id) {
    //sample a window
    timer.Start();
    vector<float> patch = target_attention_[patch_id_];
	bool do_mirror = mirror && PrefetchRand() % 2;
	// check a current image index
	if ( item_id==0 ) {
	  curr_image_id = prev_image_id = patch[0];
	  image_reload = true;
	}
	else {	
	  curr_image_id = patch[0];
	  if( prev_image_id == curr_image_id ) image_reload = false;
	  else	image_reload = true;
	}
    pair<std::string, vector<int> > image = image_database_[curr_image_id];
	//LOG(INFO) << prev_image_id << ", " << curr_image_id;
	if ( image_reload ) {
      // load the image containing the window
      if (this->cache_images_) { // if an image is already loaded. (in memory)
        pair<std::string, Datum> image_cached = image_database_cache_[curr_image_id];
        cv_img = DecodeDatumToCVMat(image_cached.second, true);
      } else {
        cv_img = cv::imread(image.first, CV_LOAD_IMAGE_COLOR);
		//LOG(INFO) << "Image reload";
        if (!cv_img.data) {
          LOG(ERROR) << "Could not open or find file " << image.first;
          return;
        }
      }
    }
    read_time += timer.MicroSeconds();
    timer.Start();
	
	// crop window out of image and warp it
    const int channels = cv_img.channels();
    const int ih = cv_img.rows;
	const int iw = cv_img.cols;
    int x1 = patch[1];
    int y1 = patch[2];
    int x2 = patch[3];
    int y2 = patch[4];
	//LOG(INFO) << ih << ", " << iw;

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
		cv::Mat cv_sub_img = cv_img(roi);
		// mean padding
		//LOG(INFO) << margin_l_x << margin_t_y << margin_r_x << margin_b_y;
		//LOG(INFO) << this->mean_values_[0] << this->mean_values_[1] << this->mean_values_[2];
		cv::Scalar value = cv::Scalar( mean_values_[0], mean_values_[1], mean_values_[2] );
      	cv::copyMakeBorder( cv_sub_img, cv_cropped_img, margin_t_y, margin_b_y, margin_l_x, margin_r_x, cv::BORDER_CONSTANT, value );
		// warping
    	cv::resize(cv_cropped_img, cv_cropped_img, cv_crop_size, 0, 0, cv::INTER_LINEAR);
	} else { // if inner region in an image
		cv::Rect roi(x1, y1, x2-x1+1, y2-y1+1);
		cv_cropped_img = cv_img(roi);
    	cv::resize(cv_cropped_img, cv_cropped_img, cv_crop_size, 0, 0, cv::INTER_LINEAR);
    }

    // horizontal flip at random
    if (do_mirror) {
      cv::flip(cv_cropped_img, cv_cropped_img, 1);
    }
	LOG(INFO) << "mirroring... " << do_mirror;
	cv::namedWindow("patch",1);
	cv::imshow("patch", cv_cropped_img);
	cv::waitKey(0);
    
	// copy the warped patch into top_data
	Dtype* top_data = top[0]->mutable_cpu_data();
    for (int h = 0; h < cv_cropped_img.rows; ++h) {
      const uchar* ptr = cv_cropped_img.ptr<uchar>(h);
      int img_index = 0;
      for (int w = 0; w < cv_cropped_img.cols; ++w) {
        for (int c = 0; c < channels; ++c) {
          int top_index = ((item_id * channels + c) * input_size + h)
                   * input_size + w;
          // int top_index = (c * height + h) * width + w;
          Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
          if (this->has_mean_values_) {
            top_data[top_index] = (pixel - this->mean_values_[c]) * scale;
          } else {
            top_data[top_index] = pixel * scale;
          }
        }
      }
    }

    trans_time += timer.MicroSeconds();
    
	// get patch direction label
  	for (int c = 0; c < this->num_class_; ++c ) {
	  Dtype* top_label_TL = top[2*c+1]->mutable_cpu_data();
	  Dtype* top_label_BR = top[2*c+2]->mutable_cpu_data();
	  if ( do_mirror ) {
		top_label_TL[item_id] = patch[5+4*c+2];
		top_label_BR[item_id] = patch[5+4*c+3];
	  } else {
		top_label_TL[item_id] = patch[5+4*c+0];
		top_label_BR[item_id] = patch[5+4*c+1];
	  }
	}
	Dtype* top_label = top[top.size()-1]->mutable_cpu_data();
	top_label[item_id] = patch[5+4*this->num_class_];
	// get next patch
	patch_id_++;
	prev_image_id = curr_image_id;
	if( patch_id_ >= total_patch_ ) patch_id_ = 0;
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms."; 
}
/*
// This function is called on prefetch thread
template <typename Dtype>
void AttentionDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  Dtype* top_data = batch->data_.mutable_cpu_data();
  Dtype* top_label = batch->label_.mutable_cpu_data();
  const int batch_size = this->layer_param_.attention_data_param().batch_size();
  const int input_size = this->layer_param_.attention_data_param().input_size();
  const bool mirror = this->layer_param_.attention_data_param().mirror();
  Dtype* mean = NULL;
  int mean_off = 0;
  int mean_width = 0;
  int mean_height = 0;
  if (this->has_mean_file_) {
    mean = this->data_mean_.mutable_cpu_data();
    mean_off = (this->data_mean_.width() - input_size) / 2;
    mean_width = this->data_mean_.width();
    mean_height = this->data_mean_.height();
  }
  cv::Size cv_crop_size(input_size, input_size);
 
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    //sample a window
    timer.Start();
    vector<float> patch = target_attention_[patch_id_];
    // load the image containing the window
    pair<std::string, vector<int> > image = image_database_[patch[0]];
    cv::Mat cv_img;
    if (this->cache_images_) {
      pair<std::string, Datum> image_cached =
        image_database_cache_[patch[0]];
      cv_img = DecodeDatumToCVMat(image_cached.second, true);
    } else {
      cv_img = cv::imread(image.first, CV_LOAD_IMAGE_COLOR);
      if (!cv_img.data) {
        LOG(ERROR) << "Could not open or find file " << image.first;
        return;
      }
    }
    read_time += timer.MicroSeconds();
    timer.Start();
    //const int channels = cv_img.channels();
    
	// crop window out of image and warp it
    int x1 = patch[1];
    int y1 = patch[2];
    int x2 = patch[3];
    int y2 = patch[4];
    
	cv::Rect roi(x1, y1, x2-x1+1, y2-y1+1);
    cv::Mat cv_cropped_img = cv_img(roi);
    cv::resize(cv_cropped_img, cv_cropped_img,
        cv_crop_size, 0, 0, cv::INTER_LINEAR);

    // horizontal flip at random
    //if (do_mirror) {
    //  cv::flip(cv_cropped_img, cv_cropped_img, 1);
    //}

    // Apply transformations (mirror, crop...) to the image
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(top_data + offset);
    this->data_transformer_->Transform(cv_cropped_img, &(this->transformed_data_));
    trans_time += timer.MicroSeconds();
    // get window label
    top_label[item_id] = patch[5];

	// get next patch
	patch_id_++;
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms."; 
}*/

INSTANTIATE_CLASS(AttentionDataLayer);
REGISTER_LAYER_CLASS(AttentionData);

}  // namespace caffe