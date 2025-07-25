#pragma once
#include <cstdio>
#include <format>

namespace moke {
template <class... Args>
auto print(const std::format_string<Args...> &fmt, Args &&...args) {
    return std::printf("%s", std::format(fmt, std::forward<Args>(args)...).c_str());
}

template <class... Args>
auto println(const std::format_string<Args...> &fmt, Args &&...args) {
    return std::printf("%s\n", std::format(fmt, std::forward<Args>(args)...).c_str());
}

template <class... Args>
auto print(FILE *out, const std::format_string<Args...> &fmt, Args &&...args) {
    return std::fprintf(out, "%s", std::format(fmt, std::forward<Args>(args)...).c_str());
}

template <class... Args>
auto println(FILE *out, const std::format_string<Args...> &fmt, Args &&...args) {
    return std::fprintf(out, "%s\n", std::format(fmt, std::forward<Args>(args)...).c_str());
}

template <class T> consteval const char *type_of() { return typeid(T).name(); }
template <> consteval const char *type_of<double>() { return "fp64"; }
template <> consteval const char *type_of<float>() { return "fp32"; }
template <> consteval const char *type_of<int64_t>() { return "int64"; }
template <> consteval const char *type_of<int32_t>() { return "int32"; }
template <> consteval const char *type_of<int16_t>() { return "int16"; }
template <> consteval const char *type_of<int8_t>() { return "int8"; }

enum TimeUnit {
    TIME_SEC = 1,
    TIME_MSEC = 1'000,
    TIME_USEC = 1'000'000,
    TIME_NSEC = 1'000'000'000,
};

template <TimeUnit SRC, TimeUnit DEST>
double TimeConvert(double src) {
    if constexpr (DEST == SRC) {
        return src;
    } else if constexpr (DEST > SRC) {
        return src * int64_t(DEST / SRC);
    } else {
        return src / int64_t(SRC / DEST);
    }
}
} // namespace moke
