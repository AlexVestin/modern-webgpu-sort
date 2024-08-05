#pragma once

#include <inttypes.h>
#include <cstddef>

extern "C" {
namespace kurbo {
struct Path {
    uint8_t* types;
    size_t n_elements;
    float* points;
    size_t n_points;
    uint32_t error;
};

struct StrokeStyle {
    float tolerance = 0.03f;
    float strokeWidth = 0.0f;
    uint32_t lineJoin = 0u;
    uint32_t lineCap = 0u;
    const double* dashes;
    size_t dashSize = 0u;
};

Path kurbo_stroke(const float* points, uint32_t n_points, const uint8_t* types, uint32_t n_elements,
                       StrokeStyle style);

uint32_t test_call();

void free_arrays(Path);
}  // namespace kurbo
}
