#pragma once
#include "moke/utils/functional/compare.hpp"
#include "moke/utils/functional/fill.hpp"
#include "moke/utils/vector.hpp"

namespace moke {
template <class Container, class Functor>
decltype(auto) operator|(Container &&container, Functor &&func) {
    return func(std::forward<Container>(container));
}
} // namespace moke
