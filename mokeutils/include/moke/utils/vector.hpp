#pragma once
#include "moke/common.hpp"
#include "moke/utils/memory.hpp"

namespace moke {
template <class T, class MS>
class allocator;

template <class T, class MS>
class vector : allocator<T, MS> {
private:
    T *m_data{nullptr};
    size_t m_length{0};

public:
    /// default construction, creates an empty vector
    vector() noexcept = default;

    /// creates vector with given length
    explicit vector(size_t length) : m_data{this->allocate(length)}, m_length{length} {}

    /// creates vector with given length, and copy data into it.
    /// @param pointer the address of data to be copied.
    /// @tparam OtherMem the memory space of data to be copied.
    template <class OtherMS = MS>
    vector(const T *pointer, size_t length, OtherMS other_ms = {}) : vector(length) {
        memory_copy(mem_space(), other_ms, m_data, pointer, length);
    }

    /// move construction
    vector(vector &&other) noexcept : m_data{other.m_data}, m_length{other.m_length} {
        other.m_data = nullptr, other.m_length = 0;
    }

    /// copy construction
    vector(const vector &other) : vector{other.m_data, other.m_length} {}

    /// copy construction with another memory space
    template <class other_memory_t>
    vector(const vector<T, other_memory_t> &other) : vector{other.data(), other.size(), other_memory_t{}} {}

    /// destruction
    ~vector() { clear(); }

    /// move assignment
    vector &operator=(vector &&other) noexcept;

    /// copy assignment
    vector &operator=(const vector &other) {
        reset(other.size());
        memory_copy(mem_space(), m_data, other.m_data, m_length);
        return *this;
    }

    /// copy assignment with another memory space
    template <class Container>
    vector &operator=(const Container &other) {
        reset(other.size());
        memory_copy(mem_space(), other.mem_space(), m_data, other.data(), m_length);
        return *this;
    }

    /// get the memory space of this vector
    using allocator<T, MS>::mem_space;

    /// get data pointer of this vector
    T *data() { return m_data; }

    /// get data pointer of this vector
    const T *data() const { return m_data; }

    /// get length of this vector
    size_t size() const { return m_length; }

    /// get size of this vector in bytes
    size_t bytes() const { return m_length * sizeof(T); }

    /// check if this vector is empty
    bool empty() const { return m_length == 0; }

    operator T *() { return m_data; }

    operator const T *() const { return m_data; }

    T *begin() { return data(); }

    const T *begin() const { return data(); }

    T *end() { return data() + size(); }

    const T *end() const { return data() + size(); }

    /// clear data and release the memory
    void clear();

    /// clear and reset this vector with given size
    void reset(size_t size);
};
} // namespace moke

#include "moke/utils/container/allocator.hpp"
#include "moke/utils/container/vector_impl.inc"
namespace moke {
template <class T>
using device_vector = vector<T, device_memory_t>;

template <class T>
using host_vector = vector<T, host_memory_t>;
} // namespace moke
