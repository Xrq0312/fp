#pragma once

#include <cstdint>
#include <vector>

namespace pugpu {

struct Timings {
    float upload_ms = 0;
    float reliability_ms = 0;
    float edge_build_ms = 0;
    float sort_ms = 0;
    float download_ms = 0;
    float union_find_ms = 0;
    float output_ms = 0;
    float gpu_total_ms = 0;
};

struct Result {
    std::vector<float> phase;
    Timings timings;
    std::uint32_t valid_pixels = 0;
    std::uint32_t edges = 0;
};

// phase is wrapped to [-pi, pi]. mask is optional; nonzero means valid.
Result unwrap(const float* phase, const std::uint8_t* mask, int width, int height);

}  // namespace pugpu

