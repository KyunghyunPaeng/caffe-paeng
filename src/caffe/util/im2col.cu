#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include "caffe/common.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/gpu_util.cuh"

namespace caffe {

template <typename Dtype>
__global__ void im2col_gpu_kernel(const int n, const Dtype* data_im,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int height_col, const int width_col,
	const int hole_h, const int hole_w,
    Dtype* data_col) {
  CUDA_KERNEL_LOOP(index, n) {
    int w_out = index % width_col;
    int h_index = index / width_col;
    int h_out = h_index % height_col;
    int channel_in = h_index / height_col;
    int channel_out = channel_in * kernel_h * kernel_w;
    int h_in = h_out * stride_h - pad_h;
    int w_in = w_out * stride_w - pad_w;
    Dtype* data_col_ptr = data_col;
    data_col_ptr += (channel_out * height_col + h_out) * width_col + w_out;
    const Dtype* data_im_ptr = data_im;
    data_im_ptr += (channel_in * height + h_in) * width + w_in;
    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        int h = h_in + i * hole_h;
        int w = w_in + j * hole_w;
        *data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ?
            data_im_ptr[(i * hole_h) * width + j * hole_w ] : 0;
        data_col_ptr += height_col * width_col;
      }
    }
  }
}

template <typename Dtype>
void im2col_gpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
	const int hole_h, const int hole_w,
    Dtype* data_col) {
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  const int kernel_h_eff = kernel_h + (kernel_h - 1) * (hole_h -1);
  const int kernel_w_eff = kernel_w + (kernel_w - 1) * (hole_w -1);
  int height_col = (height + 2 * pad_h - kernel_h_eff) / stride_h + 1;
  int width_col = (width + 2 * pad_w - kernel_w_eff) / stride_w + 1;
  int num_kernels = channels * height_col * width_col;
  // NOLINT_NEXT_LINE(whitespace/operators)
  im2col_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, data_im, height, width, kernel_h, kernel_w, pad_h,
      pad_w, stride_h, stride_w, height_col,
      width_col, hole_h, hole_w, data_col);
  CUDA_POST_KERNEL_CHECK;
}


// Explicit instantiation
template void im2col_gpu<float>(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
	const int hole_h, const int hole_w,
    float* data_col);
template void im2col_gpu<double>(const double* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
	const int hole_h, const int hole_w,
    double* data_col);

template <typename Dtype>
__global__ void col2im_gpu_kernel(const int n, const Dtype* data_col,
    const int height, const int width, const int channels,
    const int patch_h, const int patch_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int height_col, const int width_col,
	const int hole_h, const int hole_w,
    Dtype* data_im) {
  CUDA_KERNEL_LOOP(index, n) {
	int w = index % width_col;
    int h = (index/width_col) % height_col;
    int c_im = (index / width_col / height_col) % channels;
    int h_im = h * stride_h - pad_h;
    int w_im = w * stride_w - pad_w;
    Dtype* data_im_ptr = data_im;
    data_im_ptr += (c_im * height + h_im) * width + w_im;
    int c = c_im * patch_h * patch_w;
    const Dtype* data_col_ptr = data_col;
    data_col_ptr += (c * height_col + h) * width_col + w;
    for (int i = 0; i < patch_h; ++i) {
      for (int j = 0; j < patch_w; ++j) {
        int hh = h_im + i * hole_h;
        int ww = w_im + j * hole_w;
		if (hh >= 0 && hh < height && ww >= 0 && ww < width) {
			caffe_gpu_atomic_add(*data_col_ptr, &data_im_ptr[(i * hole_h) * width + j * hole_w] );
		}
		data_col_ptr += height_col * width_col;
      }
    }
  }
}

template <typename Dtype>
void col2im_gpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int hole_h, const int hole_w, Dtype* data_im) {
  const int kernel_h_eff = patch_h + (patch_h - 1) * (hole_h -1);
  const int kernel_w_eff = patch_w + (patch_w - 1) * (hole_w -1);
  int height_col = (height + 2 * pad_h - kernel_h_eff) / stride_h + 1;
  int width_col = (width + 2 * pad_w - kernel_w_eff) / stride_w + 1;
  int num_kernels = channels * height_col * width_col;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  // NOLINT_NEXT_LINE(whitespace/operators)
  caffe_gpu_set(channels * height * width, Dtype(0), data_im);
  col2im_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, data_col, height, width, channels, patch_h, patch_w,
      pad_h, pad_w, stride_h, stride_w,
      height_col, width_col, hole_h, hole_w, data_im);
  CUDA_POST_KERNEL_CHECK;
}

// Explicit instantiation
template void col2im_gpu<float>(const float* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int hole_h, const int hole_w, float* data_im);
template void col2im_gpu<double>(const double* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int hole_h, const int hole_w, double* data_im);

}  // namespace caffe
