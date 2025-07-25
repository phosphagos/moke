#include "vector_add.hpp"
#include <cstdio>

using namespace moke;

static int TestSuitIndex = 0;

template <class T>
void ExecuteVectorAdd(size_t length) {
    println("========= Test Suit {} =========", TestSuitIndex++);
    println("Basic Info:");
    println("    data type: {}", type_of<T>());
    println("       length: {}", length);
    VectorAddExecutor<T>{length}.execute();
}

int main() {
    ExecuteVectorAdd<float>(1 << 20);
    ExecuteVectorAdd<float>(64 << 20);
    ExecuteVectorAdd<int32_t>(1 << 20);
    ExecuteVectorAdd<int32_t>(64 << 20);
}
