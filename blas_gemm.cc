/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// DOC:
//      是否使用多线程
#define EIGEN_USE_THREADS

// DOC:
//      判断 GOOGLE_CUDA 或 TENSORFLOW_USE_ROCM 是否存在
//      存在则引入相关头文件
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/core/platform/stream_executor.h"
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/rnn/blas_gemm.h"
// DOC:命名空间tensorflow
namespace tensorflow {

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
namespace {
    // DOC：
    //  定义模版
    //  目的:获取设备的可用内存信息
    //  T -- 指针:记录cuda(GPU)可用的memory信息
template <typename T>
se::DeviceMemory<T> AsDeviceMemory(const T* cuda_memory) {
  se::DeviceMemoryBase wrapped(const_cast<T*>(cuda_memory));
  se::DeviceMemory<T> typed(wrapped);
  return typed;
}
}  // namespace
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

// DOC:命名空间 functor
namespace functor {
    // DOC:
    //  实现矩阵的GEMM
    //  定义泛型:方便处理不同size的矩阵
    // 参数:
    //  ctx -- 设定核方法内容的指针
    //  transa -- 设定矩阵A是否转置
    //  transb -- 设定矩阵B是否转置
    //  m -- 矩阵A和矩阵C的行数
    //  n -- 矩阵B和矩阵C的列数
    //  k -- 矩阵A和矩阵B的列数
    //  alpha -- 数乘系数
    //  a -- 矩阵A
    //  lda -- 矩阵A的递增步长
    //  b -- 矩阵B
    //  ldb -- 矩阵B的递增步长
    //  beta -- 数乘系数
    //  c -- 矩阵C 并将计算结果写入矩阵C
    //  ldc -- 矩阵C的递增步长
template <typename T>
void TensorCuBlasGemm<T>::operator()(OpKernelContext* ctx, bool transa,
                                     bool transb, uint64 m, uint64 n, uint64 k,
                                     float alpha, const T* a, int lda,
                                     const T* b, int ldb, float beta, T* c,
                                     int ldc) {
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  se::blas::Transpose trans[] = {se::blas::Transpose::kNoTranspose,
                                 se::blas::Transpose::kTranspose};

  // DOC:
  // 多个设备信息指针
  auto a_ptr = AsDeviceMemory(a);
  auto b_ptr = AsDeviceMemory(b);
  auto c_ptr = AsDeviceMemory(c);

  // DOC:
  //    采用ThenBlasGemm对矩阵进行计算
  //    定义blas_launch_status 以标识计算是否成功
  bool blas_launch_status =
      ctx->op_device_context()
          ->stream()
          ->ThenBlasGemm(trans[transa], trans[transb], m, n, k, alpha, a_ptr,
                         lda, b_ptr, ldb, beta, &c_ptr, ldc)
          .ok();
  OP_REQUIRES(ctx, blas_launch_status, errors::Aborted("CuBlasGemm failed!"));
#else
  ctx->SetStatus(errors::InvalidArgument("CuBlasGemm needs CUDA.")); // 在没有安装CUDA时信息显示报错信息
#endif
}

template struct TensorCuBlasGemm<Eigen::half>;
template struct TensorCuBlasGemm<float>;

}  // end namespace functor
}  // end namespace tensorflow
