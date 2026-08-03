#include <stdexcept>

namespace picosgl::utility {
    int even_div(int a, int b, bool allow_replicate = false) {
        if (allow_replicate && b > a) {
            if (b % a != 0) throw std::runtime_error("b must be divisible by a when allow_replicate=true");
            else return 1;
        }
        else return a / b;
    }

    int ceil_div(int a, int b) {
        return (a + b - 1) / b;
    }

    int align_ceil(int a, int b) {
        return ceil_div(a, b) * b;
    }

    int align_down(int a, int b) {
        return (a / b) * b;
    }
}