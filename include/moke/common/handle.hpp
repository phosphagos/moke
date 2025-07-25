#pragma once
#include "moke/common/runtime.hpp"

namespace moke {
struct NewAsyncHandle {};

class Handle {
private:
    NativeStream m_stream{nullptr};

public:
    Handle() noexcept = default;

    explicit Handle(NewAsyncHandle) { create_stream(); }

    ~Handle() { destroy_stream(); }

    Handle(const Handle &) = delete;

    Handle &operator=(const Handle &) = delete;

    Handle(Handle &&other) noexcept;

    Handle &operator=(Handle &&other) noexcept;

    operator NativeStream() const noexcept { return m_stream; }

    void sync() const;

    void async_memcpy(void *dest, const void *source, size_t size);

    void async_malloc(void **ptr, size_t size);

    void async_free(void *ptr);

private:
    void create_stream();

    void destroy_stream();
};

class AsyncAllocator {
private:
    Handle &m_handle;

public:
    AsyncAllocator(Handle &handle) noexcept : m_handle{handle} {}

    void allocate(void **ptr, size_t size) { return m_handle.async_malloc(ptr, size); }

    void deallocate(void *ptr) { return m_handle.async_free(ptr); }
};
} // namespace moke
