#include "moketest/comparator.hpp"
#include "moketest/common.hpp"

namespace moke {
void AccuracyResult::print(FILE *out) const {
    moke::print(out, "Accuracy Comparison: {}", bool(*this) ? "PASS" : "FAILED");
    if (max_error == 0) {
        println(out, " (result exactly the same).");
    } else if (max_error <= threshold) {
        println(out, " (error not reaches threshold).");
    } else {
        println(out, "");
    }

    println(out, "    threshold: {}", threshold);
    println(out, "    max error: {}", max_error);
    println(out, "     at index: {}", index);
}
} // namespace moke
