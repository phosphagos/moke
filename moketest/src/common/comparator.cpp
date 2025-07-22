#include "moketest/comparator.hpp"
#include "moketest/common.hpp"

namespace moke {
void AccuracyResult::print(FILE *out) const {
    println(out, "[ACCU] accuracy comparison {}:", bool(*this) ? "PASS" : "FAILED");
    if (max_error == 0) {
        println(out, "[ACCU]     result exactly the same.");
    } else if (max_error <= threshold) {
        println(out, "[ACCU]     error not reaches threshold.");
    }

    println(out, "[ACCU]     threshold: {}", threshold);
    println(out, "[ACCU]     max error: {}", max_error);
    println(out, "[ACCU]      at index: {}", index);
}
} // namespace moke
