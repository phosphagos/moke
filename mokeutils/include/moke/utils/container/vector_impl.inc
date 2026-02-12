#pragma once
#include "moke/utils/vector.hpp"

namespace moke {
template <class T, class MS>
auto vector<T, MS>::operator=(vector &&other) noexcept -> vector & {
    this->clear();
    m_data = other.m_data;
    m_length = other.m_length;
    other.m_data = nullptr;
    other.m_length = 0;
    return *this;
}

template <class T, class MS>
void vector<T, MS>::reset(size_t n) {
    if (m_length == n) { return; }
    if (m_data) { this->deallocate(m_data); }
    m_data = this->allocate(n);
    m_length = n;
}

template <class T, class MS>
void vector<T, MS>::clear() {
    if (m_data) {
        this->deallocate(m_data);
        m_data = nullptr;
        m_length = 0;
    }
}
} // namespace moke
