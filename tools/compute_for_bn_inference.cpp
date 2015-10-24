
#include "caffe/caffe.hpp"

int main(int argc, char** argv) {
  LOG(FATAL) << "Deprecated. Use caffe compute_for_bn_inference --model=... "
      "--weights=... --iterations=... instead.";
  return 0;
}
