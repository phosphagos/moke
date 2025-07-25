#include "moke/common/handle.hpp"
#include "moke/common/runtime.hpp"

namespace moke {
Handle::Handle(Handle &&other) noexcept
        : m_stream{other.m_stream} { other.m_stream = nullptr; }

Handle &Handle::operator=(Handle &&other) noexcept {
    this->destroy_stream();
    this->m_stream = other.m_stream;
    other.m_stream = nullptr;
    return *this;
}

void Handle::sync() const { CheckStatus(cudaStreamSynchronize(m_stream)); }

void Handle::async_memcpy(void *dest, const void *source, size_t size) {
    CheckStatus(cudaMemcpyAsync(dest, source, size, cudaMemcpyDefault, m_stream));
}

void Handle::async_malloc(void **ptr, size_t size) { CheckStatus(cudaMallocAsync(ptr, size, m_stream)); }

void Handle::async_free(void *ptr) { CheckStatus(cudaFreeAsync(ptr, m_stream)); }

void Handle::create_stream() { CheckStatus(cudaStreamCreate(&m_stream)); }

void Handle::destroy_stream() {
    CheckStatus(cudaStreamDestroy(m_stream));
    m_stream = nullptr;
}
} // namespace moke
