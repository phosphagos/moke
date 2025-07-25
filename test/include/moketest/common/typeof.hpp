#pragma once
#include <typeinfo>
#include <cstdint>

namespace moke {
template <class T> consteval const char *type_of() { return typeid(T).name(); }
template <> consteval const char *type_of<double>() { return "fp64"; }
template <> consteval const char *type_of<float>() { return "fp32"; }
template <> consteval const char *type_of<int64_t>() { return "int64"; }
template <> consteval const char *type_of<int32_t>() { return "int32"; }
template <> consteval const char *type_of<int16_t>() { return "int16"; }
template <> consteval const char *type_of<int8_t>() { return "int8"; }
} // namespace moke
