#pragma once
#include <moke/common.hpp>
#include <moke/mokeutils.hpp>
#include <moke/runtime.hpp>
#include <moke/type_traits.hpp>
#include <gtest/gtest.h>

#define ASSERT_IN(VALUE, LOWER, UPPER) \
    do {                               \
        ASSERT_GE(VALUE, LOWER);       \
        ASSERT_LE(VALUE, UPPER);       \
    } while (0)

#define EXPECT_IN(VALUE, LOWER, UPPER) \
    do {                               \
        EXPECT_GE(VALUE, LOWER);       \
        EXPECT_LE(VALUE, UPPER);       \
    } while (0)

namespace moke::test {
namespace traits {
    template <class T> struct GTestTypeList;

    template <template <class...> class T, class... Us>
    struct GTestTypeList<T<Us...>> : std::type_identity<::testing::Types<Us...>> {};
} // namespace traits

template <class T>
using GTestTypeList = typename traits::GTestTypeList<T>::type;

template <class... Lists>
using GTestProduction = GTestTypeList<moke::product_t<Lists...>>;
} // namespace moke::test

using memory_spaces = moke::type_tuple<moke::host_memory_t, moke::device_memory_t>;

inline size_t operator""_Ki(unsigned long long i) { return i << 10; }

inline size_t operator""_Mi(unsigned long long i) { return i << 20; }

inline size_t operator""_Gi(unsigned long long i) { return i << 30; }

inline size_t operator""_Ti(unsigned long long i) { return i << 40; }

#if defined MOKE_PLATFORM_CUDA
#include <cuda_bf16.h>
#include <cuda_fp16.h>

using half_t = __half;
using bfloat16_t = __nv_bfloat16;
#elif defined MOKE_PLATFORM_HIP
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

using half_t = __half;
using bfloat16_t = __hip_bfloat16;
#endif // MOKE_PLATFORM
