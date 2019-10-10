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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/kernels/eigen_activations.h"
#include "tensorflow/core/kernels/rnn/lstm_ops.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {
namespace functor {

// DOC:
// 定义一个GPU处理器
typedef Eigen::GpuDevice GPUDevice;

namespace {

// DOC:
// CUDA float转换为half数据类型
struct FloatToHalf {
  __host__ __device__ EIGEN_STRONG_INLINE Eigen::half operator()(
      const float& x) const {
    return Eigen::half_impl::float_to_half_rtne(x);
  }
};

// DOC:
// 如果数据类型不相同，则强制转换
//
// 参数:
//      U ---- 标准数据类型
//      T ---- 待检测数据类型
template <typename U, typename T>
__host__ __device__ EIGEN_STRONG_INLINE
    typename std::enable_if<!std::is_same<T, U>::value, U>::type
    strict_cast(T t);

// DOC:
// 如果数据类型相同，则返回其值
//
// 参数:
//      U ---- 标准数据
//      T ---- 待检测数据
//
// 返回值:
//      t ---- 比较处理之后的数据
template <typename U, typename T>
__host__ __device__ EIGEN_STRONG_INLINE
    typename std::enable_if<std::is_same<T, U>::value, U>::type
    strict_cast(T t) {
  return t;
}

// DOC:
// 将输入的float转为half数据类型输出
//
// 返回值:
//      t ---- 转换为half后的值
template <>
__host__ __device__ EIGEN_STRONG_INLINE Eigen::half
strict_cast<Eigen::half, float>(float t) {
  return FloatToHalf()(t);
}

}  // namespace

// DOC:
// 参数初始化为0
template <typename T>
struct TensorZero<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat t) {
    t.device(d) = t.constant(strict_cast<T>(0.f));
  }
};

// DOC:
// 维度大小未对齐的参数的初始化
template <typename T>
struct TensorUnalignedZero<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::UnalignedFlat t) {
    t.device(d) = t.constant(strict_cast<T>(0.f));
  }
};

namespace {

// Adds bias, applies non-linearities and gates.
//
// Launch with a 2D setup such that there is one thread per (example,
// activation) with 'x' governing example index and 'y' governing activation.
//
// Launch with blocks of (batch x 32)
//
// TODO(b/67600500): Try making 'use_peephole' a template parameter.
// DOC:
// 加入偏差值，并将其加入非线性关系和门中
// x代表样本序号，y代表激活值
// 参数：
//      gates ---- 门值
//      b     ---- 偏差值
//      cs_prev ---- 前一个记忆细胞的值
//      wci ---- 更新门的权重矩阵
//      wcf ---- 遗忘门的权重矩阵
//      wco ---- 输出门的权重矩阵
//      o   ---- 输出门值
//      h   ---- 假设值
//      i   ---- 更新门值
//      f   ---- 遗忘门值
//      ci  ---- 更新记忆细胞值
//      cs  ---- 候选记忆细胞值
//      co  ---- 输出记忆细胞值
//      forget_bias ---- 遗忘门偏差值
//      cell_clip ---- 记忆细胞偏移量
//      batch_size ---- 批量大小
//      cell_size ---- 记忆细胞容量
template <typename T, bool use_peephole, GateLayout gate_layout>
__global__ void lstm_gates(const T* gates, const T* b, const T* cs_prev,
                           const T* wci, const T* wcf, const T* wco, T* o, T* h,
                           T* ci, T* cs, T* co, T* i, T* f,
                           const float forget_bias, const float cell_clip,
                           const int batch_size, const int cell_size) {
  // DOC:
  // 计算批次序号和行为序号
  const int batch_id = blockIdx.x * blockDim.x + threadIdx.x;
  const int act_id = blockIdx.y * blockDim.y + threadIdx.y;

  // 将遗忘值偏差量和记忆细胞之间差值强制转换为泛型
  T forget_bias_t = strict_cast<T>(forget_bias);
  T cell_clip_t = strict_cast<T>(cell_clip);

  // 如果批次序号大于批量大小，或者行为id大于记忆细胞容量，则函数返回
  if (batch_id >= batch_size || act_id >= cell_size) return;

  // The following code assumes the input arrays are of the following
  // shapes and interpretations.
  //
  // 1) 'gates' is a matrix such that,
  //
  //   cell_size  cell_size  cell_size  cell_size
  //  +----------+----------+----------+----------+
  //  |          |          |          |          |
  //  |    i     |    c     |    f     |    o     |  batch_size
  //  |          |          |          |          |
  //  +----------+----------+----------+----------+
  //
  // 'gid' is the index assigned to this thread for 'gates' in the 'i'
  // submatrix.
  //
  // 2) 'b' is a vector such that,
  //
  //   cell_size  cell_size  cell_size  cell_size
  //  +----------+----------+----------+----------+
  //  |    i     |    c     |    f     |    o     |  1
  //  +----------+----------+----------+----------+
  //
  // 'act_id' is the index assigned to this thread for 'b' in the 'i' subvector.
  //
  // 3) 'wc{i,f,o}' are vectors such that,
  //
  //   cell_size
  //  +----------+
  //  |    i     |  1
  //  +----------+
  //
  //  'act_id' is the index to this thread.
  //
  // 4) All other matrices have the form,
  //
  //   cell_size
  //  +----------+
  //  |          |
  //  |    i     |  batch_size
  //  |          |
  //  +----------+
  //
  // 'cid' is the index assigned to this thread.
  //
  // DOC:
  // 定义单条神经网络门值矩阵的序号，定义单条神经网络的记忆细胞序号
  const int gid = batch_id * cell_size * 4 + act_id;
  const int cid = batch_id * cell_size + act_id;
  // 定义sigmoid函数
  Eigen::internal::scalar_logistic_op<T> sigmoid_op;
  // 定义tanh函数
  Eigen::internal::scalar_tanh_op<T> tanh_op;
  // 定义记忆细胞偏差量
  Eigen::scalar_clip_op<T> clip_op;

  // DOC:
  // 定义局部序号值
  T i_local;
  // 如果使用偷窥孔连接，则计算算式如下（考虑上一阶段的cell值）
  if (use_peephole) {
    i_local =
        sigmoid_op(gates[0 * cell_size + gid] + b[0 * cell_size + act_id] +
                   cs_prev[cid] * wci[act_id]);
  } else {
      // 如果不使用偷窥孔连接，则不考虑上一阶段的cell值，计算算式如下
    i_local =
        sigmoid_op(gates[0 * cell_size + gid] + b[0 * cell_size + act_id]);
  }
  // 记录cid
  i[cid] = i_local;

  // DOC:
  // 定义门值的偏差量
  const int c_offset = gate_c_offset(gate_layout, cell_size);
  const int f_offset = gate_f_offset(gate_layout, cell_size);

  // 计算当前记忆细胞的更新值
  const T ci_local = tanh_op(gates[c_offset + gid] + b[c_offset + act_id]);
  ci[cid] = ci_local;

  // 定义遗忘门的当前值
  T f_local;
  // 如果使用偷窥孔连接，则计算算式如下（考虑上一阶段的cell值）
  if (use_peephole) {
    f_local = sigmoid_op(gates[f_offset + gid] + b[f_offset + act_id] +
                         forget_bias_t + cs_prev[cid] * wcf[act_id]);
  } else {
      // 如果不使用偷窥孔连接，则不考虑上一阶段的cell值，计算算式如下
    f_local = sigmoid_op(gates[f_offset + gid] + b[f_offset + act_id] +
                         forget_bias_t);
  }
  // 记录cid
  f[cid] = f_local;

  // 计算候选记忆细胞的当前值
  T cs_local = i_local * ci_local + f_local * cs_prev[cid];
  if (cell_clip > 0.0f) {
    cs_local = clip_op(cs_local, cell_clip_t);
  }
  // 记录cid
  cs[cid] = cs_local;

  // 计算输出记忆细胞的当前值
  const T co_local = tanh_op(cs_local);
  // 记录cid
  co[cid] = co_local;

  // 定义输出记忆细胞的当前值
  T o_local;
  // 如果使用偷窥孔连接，则计算算式如下（考虑上一阶段的cell值）
  if (use_peephole) {
    o_local = sigmoid_op(gates[3 * cell_size + gid] +
                         b[3 * cell_size + act_id] + cs_local * wco[act_id]);
  } else {
      // 如果不使用偷窥孔连接，则不考虑上一阶段的cell值，计算算式如下
    o_local =
        sigmoid_op(gates[3 * cell_size + gid] + b[3 * cell_size + act_id]);
  }
  // 记录cid
  o[cid] = o_local;

  // 计算假设值
  h[cid] = o_local * co_local;
}

// Concatenate 'x' and 'h' and copy their contents into 'xh'.
// DOC:
// 连接x和h，把它们的内容合并到xh中
template <typename T>
__global__ void concat_xh(T* xh, const T* x, const T* h_prev,
                          const int batch_size, const int cell_size,
                          const int input_size) {
  // Assumes 'x', 'h', and 'xh' are of the following shape,
  //
  //   input_size  cell_size
  //  +----------+----------+
  //  |          |          |
  //  |    x     |    h     |  batch_size
  //  |          |          |
  //  +----------+----------+
  // DOC:
  // 计算门值序号和宽度
  const int gid = blockDim.x * blockIdx.x + threadIdx.x;
  const int width = input_size + cell_size;

  // 如果们只需好大于宽度乘批量大小，则函数返回
  if (gid >= width * batch_size) return;

  // 定义输出的行列容量
  const int output_row = gid / width;
  const int output_col = gid % width;

  // 如果输出列数比输入列数小，则xh维度与x一致
  if (output_col < input_size) {  // x
    xh[gid] = x[output_row * input_size + output_col];
  } else {  // h
      // 如果输出列数比输入列数大，则xh维度与h一致
    xh[gid] = h_prev[output_row * cell_size + output_col - input_size];
  }
}

// DOC:
// LSTM细胞模块向前传播算法（使用CUDA）
//
// 参数：
//      ctx ---- 输入内容
//      d   ---- GPU处理器
//      forget_bias ---- 遗忘门偏差值
//      cell_clip   ---- 记忆细胞偏移量
//      use_peephole    ---- 是否使用偷窥孔连接
//      x   ---- 输入值
//      cs_prev ---- 前一个记忆细胞的值
//      w   ---- 权重矩阵
//      wci ---- 更新门的权重矩阵
//      wcf ---- 遗忘门的权重矩阵
//      wco ---- 输出门的权重矩阵
//      b   ---- 偏差值
//      xh  ---- x和h的组合值
//      i   ---- 更新门值
//      cs  ---- 候选记忆细胞值
//      f   ---- 遗忘门值
//      o   ---- 输出门值
//      ci  ---- 更新记忆细胞值
//      co  ---- 输出记忆细胞值
//      gates   ---- 门值
//      h   ---- 假设值
//      cell_size   ---- 记忆细胞容量
//      input_size  ---- 输入大小
template <typename T, GateLayout gate_layout>
void LSTMBlockCellFpropWithCUDA(
    OpKernelContext* ctx, const GPUDevice& d, const float forget_bias,
    const float cell_clip, bool use_peephole, typename TTypes<T>::ConstMatrix x,
    typename TTypes<T>::ConstMatrix cs_prev,
    typename TTypes<T>::ConstMatrix h_prev, typename TTypes<T>::ConstMatrix w,
    typename TTypes<T>::ConstVec wci, typename TTypes<T>::ConstVec wcf,
    typename TTypes<T>::ConstVec wco, typename TTypes<T>::ConstVec b,
    typename TTypes<T>::Matrix xh, typename TTypes<T>::Matrix i,
    typename TTypes<T>::Matrix cs, typename TTypes<T>::Matrix f,
    typename TTypes<T>::Matrix o, typename TTypes<T>::Matrix ci,
    typename TTypes<T>::Matrix co, typename TTypes<T>::Matrix gates,
    typename TTypes<T>::Matrix h, int batch_size, int cell_size,
    int input_size) {
    // 定义GPU流
  const auto& cu_stream = GetGpuStream(ctx);

  // Concatenate xh = [x, h].
  //
  // Each block is assigned 128 threads. Good values are in [128, 1024] and are
  // divisible by 32 (the size of a warp). The number of blocks is such that
  // there are enough to process all the data.
  // DOC:
  // 定义模块维度
  const int block_dim = 128;
  // 定义网格维度
  const int grid_dim =
      Eigen::divup(batch_size * (cell_size + input_size), block_dim);
  // 确定GPU内核成功启动
  TF_CHECK_OK(GpuLaunchKernel(concat_xh<T>, grid_dim, block_dim, 0, cu_stream,
                              xh.data(), x.data(), h_prev.data(), batch_size,
                              cell_size, input_size));

  // states1 = xh * w
  typename TTypes<T>::ConstMatrix const_xh(xh.data(), xh.dimensions());
  TensorBlasGemm<GPUDevice, T, true /* USE_CUBLAS */>::compute(
      ctx, d, false, false, typename gemm_compute_type<T>::type(1.f), const_xh,
      w, typename gemm_compute_type<T>::type(0.f), gates);

  // Add bias, apply non-linearities and gating.
  //
  // Use 2D blocks. The number of threads per block is equal to x * y, where x =
  // min(batch_size, 8) and y = 32. See above for guidance on number of
  // threads.
  // DOC:
  // 定义二维模块和网格
  dim3 block_dim_2d(std::min(batch_size, 8), 32);
  dim3 grid_dim_2d(Eigen::divup(batch_size, static_cast<int>(block_dim_2d.x)),
                   Eigen::divup(cell_size, static_cast<int>(block_dim_2d.y)));
  // 如果使用偷窥孔连接
  if (use_peephole) {
      // 确定GPU内核成功启动
    TF_CHECK_OK(GpuLaunchKernel(
        lstm_gates<T, true, gate_layout>, grid_dim_2d, block_dim_2d, 0,
        cu_stream, gates.data(), b.data(), cs_prev.data(), wci.data(),
        wcf.data(), wco.data(), o.data(), h.data(), ci.data(), cs.data(),
        co.data(), i.data(), f.data(), forget_bias, cell_clip, batch_size,
        cell_size));
  } else {
      // 确定GPU内核成功启动
    TF_CHECK_OK(GpuLaunchKernel(
        lstm_gates<T, false, gate_layout>, grid_dim_2d, block_dim_2d, 0,
        cu_stream, gates.data(), b.data(), cs_prev.data(), wci.data(),
        wcf.data(), wco.data(), o.data(), h.data(), ci.data(), cs.data(),
        co.data(), i.data(), f.data(), forget_bias, cell_clip, batch_size,
        cell_size));
  }
}

// DOC:
// LSTM反向传播算法
//
// 参数：
//      cs_prev ---- 前一个记忆细胞的值
//      h_prev  ---- 前一个假设值
//      w   ---- 权重矩阵
//      wci ---- 更新门的权重矩阵
//      wcf ---- 遗忘门的权重矩阵
//      wco ---- 输出门的权重矩阵
//      b   ---- 偏差值
//      i   ---- 更新门值
//      cs  ---- 候选记忆细胞值
//      f   ---- 遗忘门值
//      o   ---- 输出门值
//      ci  ---- 更新记忆细胞值
//      co  ---- 输出记忆细胞值
//      cs_grad ---- 记忆细胞值的梯度
//      h_grad  ----  假设值的梯度
//      do_ ---- 输出值的偏导值
//      dcs ---- 候选记忆细胞值的偏导值
//      dci ---- 更新记忆细胞偏导值
//      df ---- 遗忘门偏导值
//      di ----  更新门偏导值
//      dgates  ----  门值偏导值
//      cs_prev_grad    ---- 前一个记忆细胞值的梯度
//      batch_size  ---- 批量大小
//      cell_size   ---- 细胞大小
//      use_peephole    ---- 是否使用偷窥孔连接
template <typename T, GateLayout gate_layout>
__global__ void lstm_gates_bprop(
    // 定义反向传播算法所需要的一系列参数
    const T* cs_prev,  // [batch_size, cell_size]
    const T* h_prev,   // [batch_size, cell_size]
    const T* w,        // [input_size + cell_size, 4 * cell_size]
    const T* wci,      // [cell_size]
    const T* wcf,      // [cell_size]
    const T* wco,      // [cell_size]
    const T* b,        // [4 * cell_size]
    const T* i,        // [batch_size, cell_size]
    const T* cs,       // [batch_size, cell_size]
    const T* f,        // [batch_size, cell_size]
    const T* o,        // [batch_size, cell_size]
    const T* ci,       // [batch_size, cell_size]
    const T* co,       // [batch_size, cell_size]
    const T* cs_grad,  // [batch_size, cell_size]
    const T* h_grad,   // [batch_size, cell_size]
    // 偏导值
    T* do_,            // [batch_size, cell_size]
    T* dcs,            // [batch_size, cell_size]
    T* dci,            // [batch_size, cell_size]
    T* df,             // [batch_size, cell_size]
    T* di,             // [batch_size, cell_size]
    T* dgates,         // [input_size + cell_size, 4 * cell_size]
    T* cs_prev_grad,   // [batch_size, cell_size]
    const int batch_size, const int cell_size, const bool use_peephole) {
  // DOC:
  // 定义批次序号和行为序号
  const int batch_id = blockIdx.x * blockDim.x + threadIdx.x;
  const int act_id = blockIdx.y * blockDim.y + threadIdx.y;

  // 如果批次序号大于等于批量大小，或者行为序号大于等于记忆细胞容量，则函数返回
  if (batch_id >= batch_size || act_id >= cell_size) return;

  // 计算gid和cid
  const int gid = batch_id * cell_size * 4 + act_id;
  const int cid = batch_id * cell_size + act_id;

  // 初始化一个全为1的矩阵
  const T one = static_cast<T>(1.0f);

  // DOC:
  // 反向传播算法实现
  // do[t] = sigm'(o[t]) .* dh[t] .* co[t]
  const T o_local = o[cid];
  const T h_grad_local = h_grad[cid];
  const T co_local = co[cid];
  const T ci_local = ci[cid];
  const T do_local = o_local * (one - o_local) * h_grad_local * co_local;
  const T i_local = i[cid];
  const T f_local = f[cid];

  do_[cid] = do_local;

  // dcs[t] += tanh'(cs[t]) .* dh[t] .* o[t] + dcs[t + 1] .* f[t + 1]
  T dcs_local =
      (one - co_local * co_local) * h_grad_local * o_local + cs_grad[cid];
  if (use_peephole) {
    dcs_local += do_local * wco[act_id];
  }
  dcs[cid] = dcs_local;

  // dci[t] = tanh'(ci[t]) dcs[t] i[t]
  const T dci_local = (one - ci_local * ci_local) * dcs_local * i_local;
  dci[cid] = dci_local;

  // df[t] = sigm'(f[t]) dcs[t] cs[t - 1]
  const T df_local = f_local * (one - f_local) * dcs_local * cs_prev[cid];
  df[cid] = df_local;

  // di[t] = sigm'(i[t]) dcs[t] ci[t]
  const T di_local = i_local * (one - i_local) * dcs_local * ci_local;
  di[cid] = di_local;

  dgates[gid + 0 * cell_size] = di_local;
  dgates[gate_c_offset(gate_layout, cell_size)] = dci_local;
  dgates[gate_f_offset(gate_layout, cell_size)] = df_local;
  dgates[gid + 3 * cell_size] = do_local;

  cs_prev_grad[cid] = dcs_local * f_local;
  if (use_peephole) {
    cs_prev_grad[cid] += di_local * wci[act_id] + df_local * wcf[act_id];
  }
}

// DOC:
// LSTM细胞模块向后传播算法（使用CUDA）
//
// 参数：
//      ctx ---- 输入内容
//      d   ---- GPU处理器
//      x   ---- 输入值
//      cs_prev ---- 前一个记忆细胞的值
//      h_prev  ---- 前一个假设值
//      w   ---- 权重矩阵
//      wci ---- 更新门的权重矩阵
//      wcf ---- 遗忘门的权重矩阵
//      wco ---- 输出门的权重矩阵
//      b   ---- 偏差值
//      i   ---- 更新门值
//      cs  ---- 候选记忆细胞值
//      f   ---- 遗忘门值
//      o   ---- 输出门值
//      ci  ---- 更新记忆细胞值
//      co  ---- 输出记忆细胞值
//      cs_grad ---- 记忆细胞值的梯度
//      h_grad  ----  假设值的梯度
//      do_ ---- 输出值的偏导值
//      dcs ---- 候选记忆细胞值的偏导值
//      dci ---- 更新记忆细胞偏导值
//      df ---- 遗忘门偏导值
//      di ----  更新门偏导值
//      dgates  ----  门值偏导值
//      cs_prev_grad    ---- 前一个记忆细胞值的梯度
//      batch_size  ---- 批量大小
//      cell_size   ---- 细胞大小
//      use_peephole    ---- 是否使用偷窥孔连接
template <typename T, GateLayout gate_layout>
void LSTMBlockCellBpropWithCUDA(
    OpKernelContext* ctx, const GPUDevice& d, typename TTypes<T>::ConstMatrix x,
    typename TTypes<T>::ConstMatrix cs_prev,
    typename TTypes<T>::ConstMatrix h_prev, typename TTypes<T>::ConstMatrix w,
    typename TTypes<T>::ConstVec wci, typename TTypes<T>::ConstVec wcf,
    typename TTypes<T>::ConstVec wco, typename TTypes<T>::ConstVec b,
    typename TTypes<T>::ConstMatrix i, typename TTypes<T>::ConstMatrix cs,
    typename TTypes<T>::ConstMatrix f, typename TTypes<T>::ConstMatrix o,
    typename TTypes<T>::ConstMatrix ci, typename TTypes<T>::ConstMatrix co,
    typename TTypes<T>::ConstMatrix cs_grad,
    typename TTypes<T>::ConstMatrix h_grad, typename TTypes<T>::Matrix do_,
    typename TTypes<T>::Matrix dcs, typename TTypes<T>::Matrix dci,
    typename TTypes<T>::Matrix df, typename TTypes<T>::Matrix di,
    typename TTypes<T>::Matrix dgates, typename TTypes<T>::Matrix cs_prev_grad,
    typename TTypes<T>::Vec wci_grad, typename TTypes<T>::Vec wcf_grad,
    typename TTypes<T>::Vec wco_grad, const int batch_size, const int cell_size,
    const bool use_peephole) {
  const auto& cu_stream = GetGpuStream(ctx);
  // DOC:
  // 定义二维模块和网格
  dim3 block_dim_2d(std::min(batch_size, 8), 32);
  dim3 grid_dim_2d(Eigen::divup(batch_size, static_cast<int>(block_dim_2d.x)),
                   Eigen::divup(cell_size, static_cast<int>(block_dim_2d.y)));

  // 确定GPU内核成功启动
  TF_CHECK_OK(GpuLaunchKernel(
      lstm_gates_bprop<T, gate_layout>, grid_dim_2d, block_dim_2d, 0, cu_stream,
      cs_prev.data(), h_prev.data(), w.data(), wci.data(), wcf.data(),
      wco.data(), b.data(), i.data(), cs.data(), f.data(), o.data(), ci.data(),
      co.data(), cs_grad.data(), h_grad.data(), do_.data(), dcs.data(),
      dci.data(), df.data(), di.data(), dgates.data(), cs_prev_grad.data(),
      batch_size, cell_size, use_peephole));

  // 如果使用偷窥孔连接，则计算算式如下（考虑上一阶段的cell值）
  if (use_peephole) {
    Eigen::array<Eigen::DenseIndex, 2> p_shape({1, cell_size});
    Eigen::array<Eigen::DenseIndex, 2> p_broadcast_shape({batch_size, 1});
    cs_prev_grad.device(d) =
        cs_prev_grad + di * wci.reshape(p_shape).broadcast(p_broadcast_shape) +
        df * wcf.reshape(p_shape).broadcast(p_broadcast_shape);
    wci_grad.device(d) = (di * cs_prev).sum(Eigen::array<int, 1>({0}));
    wcf_grad.device(d) = (df * cs_prev).sum(Eigen::array<int, 1>({0}));
    wco_grad.device(d) = (do_ * cs).sum(Eigen::array<int, 1>({0}));
  }
}

}  // namespace

// DOC:
// 声明使用GPU时LSTM向前向后传播算法
#define DECLARE_GPU_FBPROP(T, GATE_LAYOUT)                                    \
// 定义LSTM向前传播算法块
//
// 成员变量：
//      ctx ---- 输入内容
//      d ---- CPU处理器
//      forget_bias ---- 遗忘门偏差值
//      cell_clip   ---- 记忆细胞偏移量
//      use_peephole    ---- 是否使用偷窥孔连接
//      x   ---- 输入值
//      cs_prev ---- 前一个记忆细胞的值
//      h_prev  ---- 前一个假设值
//      w   ---- 权重矩阵
//      wci ---- 更新门的权重矩阵
//      wcf ---- 遗忘门的权重矩阵
//      wco ---- 输出门的权重矩阵
//      b   ---- 偏差值
//      xh  ---- x和h的组合值
//      i   ---- 更新门值
//      cs  ---- 候选记忆细胞值
//      f   ---- 遗忘门值
//      o   ---- 输出门值
//      ci  ---- 更新记忆细胞值
//      co  ---- 输出记忆细胞值
//      gates   ---- 门值
//      h   ---- 假设值
  template <>                                                                 \
  void LSTMBlockCellFprop<GPUDevice, T, true /* USE_CUBLAS */, GATE_LAYOUT>:: \
  operator()(                                                                 \
      OpKernelContext* ctx, const GPUDevice& d, const float forget_bias,      \
      const float cell_clip, bool use_peephole,                               \
      typename TTypes<T>::ConstMatrix x,                                      \
      typename TTypes<T>::ConstMatrix cs_prev,                                \
      typename TTypes<T>::ConstMatrix h_prev,                                 \
      typename TTypes<T>::ConstMatrix w, typename TTypes<T>::ConstVec wci,    \
      typename TTypes<T>::ConstVec wcf, typename TTypes<T>::ConstVec wco,     \
      typename TTypes<T>::ConstVec b, typename TTypes<T>::Matrix xh,          \
      typename TTypes<T>::Matrix i, typename TTypes<T>::Matrix cs,            \
      typename TTypes<T>::Matrix f, typename TTypes<T>::Matrix o,             \
      typename TTypes<T>::Matrix ci, typename TTypes<T>::Matrix co,           \
      typename TTypes<T>::Matrix gates, typename TTypes<T>::Matrix h) {       \
    LSTMBlockCellFpropWithCUDA<T, GATE_LAYOUT>(                               \
        ctx, d, forget_bias, cell_clip, use_peephole, x, cs_prev, h_prev, w,  \
        wci, wcf, wco, b, xh, i, cs, f, o, ci, co, gates, h, batch_size_,     \
        cell_size_, input_size_);                                             \
  }                                                                           \
// 定义LSTM反向传播算法块
// 成员变量：
//      cell ---- 记忆细胞值
//      ctx ---- 输入内容
//      d --- CPU处理器
//      use_peephole    ---- 是否使用偷窥孔连接
//      x   ---- 输入值
//      cs_prev ---- 前一个记忆细胞的值
//      h_prev  ---- 前一个假设值
//      w   ---- 权重矩阵
//      wci ---- 更新门的权重矩阵
//      wcf ---- 遗忘门的权重矩阵
//      wco ---- 输出门的权重矩阵
//      b   ---- 偏差值
//      i   ---- 更新门值
//      cs  ---- 候选记忆细胞值
//      f   ---- 遗忘门值
//      o   ---- 输出门值
//      ci  ---- 更新记忆细胞值
//      co  ---- 输出记忆细胞值
//      cs_grad ---- 记忆细胞值的梯度
//      h_grad  ----  假设值的梯度
//      do_ ---- 输出值的偏导值
//      dcs ---- 候选记忆细胞值的偏导值
//      dci ---- 更新记忆细胞偏导值
//      df ---- 遗忘门偏导值
//      di ----  更新门偏导值
//      dgates  ----  门值偏导值
//      cs_prev_grad    ---- 前一个记忆细胞值的梯度
//      wci_grad    ---- 更新门权重矩阵的梯度
//      wcf_grad    ---- 遗忘门权重矩阵的梯度
//      wco_grad    ---- 输出门权重矩阵的梯度
  template <>                                                                 \
  void LSTMBlockCellBprop<GPUDevice, T, true /* USE_CUBLAS */, GATE_LAYOUT>:: \
  operator()(                                                                 \
      OpKernelContext* ctx, const GPUDevice& d, bool use_peephole,            \
      typename TTypes<T>::ConstMatrix x,                                      \
      typename TTypes<T>::ConstMatrix cs_prev,                                \
      typename TTypes<T>::ConstMatrix h_prev,                                 \
      typename TTypes<T>::ConstMatrix w, typename TTypes<T>::ConstVec wci,    \
      typename TTypes<T>::ConstVec wcf, typename TTypes<T>::ConstVec wco,     \
      typename TTypes<T>::ConstVec b, typename TTypes<T>::ConstMatrix i,      \
      typename TTypes<T>::ConstMatrix cs, typename TTypes<T>::ConstMatrix f,  \
      typename TTypes<T>::ConstMatrix o, typename TTypes<T>::ConstMatrix ci,  \
      typename TTypes<T>::ConstMatrix co,                                     \
      typename TTypes<T>::ConstMatrix cs_grad,                                \
      typename TTypes<T>::ConstMatrix h_grad, typename TTypes<T>::Matrix do_, \
      typename TTypes<T>::Matrix dcs, typename TTypes<T>::Matrix dci,         \
      typename TTypes<T>::Matrix df, typename TTypes<T>::Matrix di,           \
      typename TTypes<T>::Matrix dgates,                                      \
      typename TTypes<T>::Matrix cs_prev_grad,                                \
      typename TTypes<T>::Vec wci_grad, typename TTypes<T>::Vec wcf_grad,     \
      typename TTypes<T>::Vec wco_grad) {                                     \
    LSTMBlockCellBpropWithCUDA<T, GATE_LAYOUT>(                               \
        ctx, d, x, cs_prev, h_prev, w, wci, wcf, wco, b, i, cs, f, o, ci, co, \
        cs_grad, h_grad, do_, dcs, dci, df, di, dgates, cs_prev_grad,         \
        wci_grad, wcf_grad, wco_grad, batch_size_, cell_size_, use_peephole); \
  }                                                                           \
  template struct LSTMBlockCellFprop<GPUDevice, T, true /* USE_CUBLAS */,     \
                                     GATE_LAYOUT>;                            \
  template struct LSTMBlockCellBprop<GPUDevice, T, true /* USE_CUBLAS */,     \
                                     GATE_LAYOUT>;                            \
  template struct BlockLSTMBprop<GPUDevice, T, true /* USE_CUBLAS */,         \
                                 GATE_LAYOUT>;

// DOC:
// 声明使用GPU时表示特性的参数
#define DECLARE_GPU_SPECS(T)                           \
  template struct TensorZero<GPUDevice, T>;            \
  template struct TensorUnalignedZero<GPUDevice, T>;   \
  template struct TensorCopy<GPUDevice, T>;            \
  template struct TensorCopyUnaligned<GPUDevice, T>;   \
  template struct TensorCopyToUnaligned<GPUDevice, T>; \
  template struct TensorAdd<GPUDevice, T>;             \
                                                       \
  DECLARE_GPU_FBPROP(T, ICFO);

DECLARE_GPU_SPECS(Eigen::half);
DECLARE_GPU_SPECS(float);
#undef DECLARE_GPU_SPECS
#undef DECLARE_GPU_FBPROP
}  // end namespace functor
}  // end namespace tensorflow
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
