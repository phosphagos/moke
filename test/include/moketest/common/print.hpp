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
} // namespace moke
