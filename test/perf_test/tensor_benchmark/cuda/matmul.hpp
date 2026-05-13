#pragma once
#include "moke/common.hpp"
#include "moke/dtype.hpp"
#include "moke/runtime.hpp"

namespace moke::ops {
template <class dtype>
void matmul_baseline(const dtype *a, const dtype *b, dtype *d, int M, int N, int K);

template <class dtype>
void matmul_indexing_tensor(const dtype *a, const dtype *b, dtype *d, int M, int N, int K);

template <class dtype>
void matmul_indexing_mdspan(const dtype *a, const dtype *b, dtype *d, int M, int N, int K);
} // namespace moke::ops
