#pragma once
#include <moke/common.hpp>
#include <moke/dtype.hpp>
#include <moke/mokeutils.hpp>
#include <moke/runtime.hpp>
#include <moke/meta.hpp>
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
