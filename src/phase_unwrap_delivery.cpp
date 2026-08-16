#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cwctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace fs = std::filesystem;

constexpr float kPi = 3.14159265358979323846f;
constexpr float kTwoPi = 2.0f * kPi;

struct PhaseAnchorROI {
    cv::Rect rect;
    int k_order = 0;
};

struct Config {
    fs::path config_path;
    fs::path phase_path;
    fs::path raw_dir;
    fs::path out_dir = "phase_unwrap_task/results";
    std::wstring prefix = L"phase";
    bool batch = false;
    fs::path batch_root = "PMD_DATA";
    bool invert_phase = false;
    float raw_phase_offset = 0.0f;
    bool decode_raw_phase = false;
    int raw_phase_start = 1;
    std::wstring raw_phase_series;
    float coherence_threshold = 0.45f;
    float modulation_abs_floor = 4.0f;
    float modulation_min_weak_ratio_for_otsu = 0.20f;
    float modulation_otsu_cap_pct = 25.0f;
    int modulation_erode_size = 0;
    bool use_modulation_mask = true;
    bool absolute_k_enabled = false;
    std::vector<PhaseAnchorROI> absolute_k_rois;
    std::optional<cv::Rect> integration_roi;
    float min_amplitude = 4.0f;
    float good_amplitude = 20.0f;
    float residual_scale = 0.35f;
    float confidence_threshold = 0.25f;
    float max_residue_density = 0.10f;
    int coherence_radius = 4;
    int residue_density_radius = 3;
    int residual_radius = 61;
    int background_radius = 241;
    int background_iterations = 4;
    int relief_smooth_radius = 2;
    int detail_overlay_threshold = 18;
    float background_outlier_scale = 1.25f;
    float phase_enhance_gain = 2.5f;
    float phase_display_low_pct = 0.05f;
    float phase_display_high_pct = 99.5f;
    float competitor_absolute_low_pct = 0.5f;
    float competitor_absolute_high_pct = 99.5f;
    float trend_fit_phase_threshold = 0.6f;
};

struct Edge {
    int a = 0;
    int b = 0;
    float reliability = 0.0f;
};

struct UnionFind {
    std::vector<int> parent;
    std::vector<int> size;
    std::vector<int> period_to_parent;

    explicit UnionFind(int n = 0) : parent(n, -1), size(n, 1), period_to_parent(n, 0) {}

    void activate(int i) {
        parent[i] = i;
        size[i] = 1;
        period_to_parent[i] = 0;
    }

    int find(int x) {
        int p = parent[x];
        if (p == x) {
            return x;
        }
        int root = find(p);
        period_to_parent[x] += period_to_parent[p];
        parent[x] = root;
        return root;
    }

    int periodToRoot(int x) {
        find(x);
        return period_to_parent[x];
    }

    void unite(int p, int q, int q_minus_p_periods) {
        int rp = find(p);
        int p_to_rp = period_to_parent[p];
        int rq = find(q);
        int q_to_rq = period_to_parent[q];
        if (rp == rq) {
            return;
        }

        if (size[rp] < size[rq]) {
            parent[rp] = rq;
            period_to_parent[rp] = q_to_rq - p_to_rp - q_minus_p_periods;
            size[rq] += size[rp];
        } else {
            parent[rq] = rp;
            period_to_parent[rq] = q_minus_p_periods + p_to_rp - q_to_rq;
            size[rp] += size[rq];
        }
    }
};

std::string narrowAscii(const std::wstring& text) {
    std::string out;
    out.reserve(text.size());
    for (wchar_t ch : text) {
        out.push_back(ch < 128 ? static_cast<char>(ch) : '_');
    }
    return out;
}

std::string narrowAscii(const fs::path& path) {
    return narrowAscii(path.wstring());
}

std::wstring upperAscii(std::wstring text) {
    for (wchar_t& ch : text) {
        ch = static_cast<wchar_t>(std::towupper(ch));
    }
    return text;
}

bool endsWith(const std::wstring& text, const std::wstring& suffix) {
    return text.size() >= suffix.size() &&
        text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

float wrapPi(float x) {
    x = std::fmod(x + kPi, kTwoPi);
    if (x < 0.0f) {
        x += kTwoPi;
    }
    return x - kPi;
}

float smoothstep(float edge0, float edge1, float x) {
    if (edge1 <= edge0) {
        return x >= edge1 ? 1.0f : 0.0f;
    }
    float t = std::clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

std::optional<std::vector<uchar>> readFileBytes(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        return std::nullopt;
    }
    input.seekg(0, std::ios::end);
    const auto size = input.tellg();
    input.seekg(0, std::ios::beg);
    std::vector<uchar> bytes(static_cast<size_t>(size));
    input.read(reinterpret_cast<char*>(bytes.data()), size);
    return bytes;
}

bool writeFileBytes(const fs::path& path, const std::vector<uchar>& bytes) {
    fs::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        return false;
    }
    output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    return true;
}

cv::Mat loadGray(const fs::path& path) {
    auto bytes = readFileBytes(path);
    if (!bytes) {
        throw std::runtime_error("failed to read image: " + narrowAscii(path));
    }
    cv::Mat image = cv::imdecode(*bytes, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        throw std::runtime_error("failed to decode image: " + narrowAscii(path));
    }
    return image;
}

void saveImage(const fs::path& path, const cv::Mat& image) {
    std::vector<uchar> bytes;
    const std::string ext = path.extension().empty() ? ".png" : path.extension().string();
    if (!cv::imencode(ext, image, bytes)) {
        throw std::runtime_error("failed to encode image: " + narrowAscii(path));
    }
    if (!writeFileBytes(path, bytes)) {
        throw std::runtime_error("failed to write image: " + narrowAscii(path));
    }
}

fs::path findIndexedImage(const fs::path& dir, int index, const std::wstring& series = L"") {
    const std::wstring wanted = std::to_wstring(index);
    const std::wstring series_upper = upperAscii(series);
    const std::wstring wanted_compact = series_upper + wanted;
    const std::wstring wanted_suffix = L"." + series_upper + wanted;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const std::wstring stem = entry.path().stem().wstring();
        if (stem == wanted) {
            return entry.path();
        }
        if (!series_upper.empty()) {
            const std::wstring stem_upper = upperAscii(stem);
            if (stem_upper == wanted_compact || endsWith(stem_upper, wanted_suffix)) {
                return entry.path();
            }
        }
    }
    return {};
}

float otsuFloat(const cv::Mat1f& values, const cv::Mat1b& valid = cv::Mat1b()) {
    constexpr int bins = 256;
    double min_value = 0.0;
    double max_value = 0.0;
    if (valid.empty()) {
        cv::minMaxLoc(values, &min_value, &max_value);
    } else {
        cv::minMaxLoc(values, &min_value, &max_value, nullptr, nullptr, valid);
    }
    if (max_value <= min_value) {
        return static_cast<float>(min_value);
    }

    std::vector<double> hist(bins, 0.0);
    const double scale = (bins - 1) / (max_value - min_value);
    for (int y = 0; y < values.rows; ++y) {
        const float* row = values.ptr<float>(y);
        const uchar* mask_row = valid.empty() ? nullptr : valid.ptr<uchar>(y);
        for (int x = 0; x < values.cols; ++x) {
            if (mask_row && mask_row[x] == 0) {
                continue;
            }
            int bin = static_cast<int>((row[x] - min_value) * scale + 0.5);
            bin = std::clamp(bin, 0, bins - 1);
            hist[bin] += 1.0;
        }
    }

    double total = std::accumulate(hist.begin(), hist.end(), 0.0);
    if (total <= 0.0) {
        return static_cast<float>(min_value);
    }

    double weighted_total = 0.0;
    for (int i = 0; i < bins; ++i) {
        weighted_total += hist[i] * i;
    }

    double weight0 = 0.0;
    double sum0 = 0.0;
    double best_score = -1.0;
    int best_bin = 0;
    for (int i = 0; i < bins; ++i) {
        weight0 += hist[i];
        sum0 += hist[i] * i;
        double weight1 = total - weight0;
        if (weight0 <= 0.0 || weight1 <= 0.0) {
            continue;
        }
        double mean0 = sum0 / weight0;
        double mean1 = (weighted_total - sum0) / weight1;
        double score = weight0 * weight1 * (mean0 - mean1) * (mean0 - mean1);
        if (score > best_score) {
            best_score = score;
            best_bin = i;
        }
    }
    return static_cast<float>(min_value + (max_value - min_value) * best_bin / (bins - 1));
}

cv::Mat1f grayToPhase(const cv::Mat& gray, bool invert_phase) {
    cv::Mat1f phase(gray.rows, gray.cols);
    const float sign = invert_phase ? -1.0f : 1.0f;
    for (int y = 0; y < gray.rows; ++y) {
        const uchar* src = gray.ptr<uchar>(y);
        float* dst = phase.ptr<float>(y);
        for (int x = 0; x < gray.cols; ++x) {
            dst[x] = sign * (static_cast<float>(src[x]) * (kTwoPi / 255.0f) - kPi);
        }
    }
    return phase;
}

struct DecodedPhase {
    cv::Mat1f phase;
    cv::Mat1f modulation;
    std::array<cv::Mat1f, 4> frames;
};

struct PixelQuality {
    cv::Mat1f amplitude;
    cv::Mat1f contrast;
    cv::Mat1f model_residual;
    cv::Mat1f coherence;
    cv::Mat1f residue_density;
    cv::Mat1f confidence;
    cv::Mat1b valid;
};

std::optional<std::array<cv::Mat1f, 4>> loadFourStepFrames(
    const fs::path& raw_dir,
    int start_index,
    cv::Size expected_size,
    const std::wstring& series = L"") {
    if (raw_dir.empty() || !fs::exists(raw_dir)) {
        return std::nullopt;
    }

    std::array<cv::Mat1f, 4> images;
    for (int i = 0; i < 4; ++i) {
        fs::path path = findIndexedImage(raw_dir, start_index + i, series);
        if (path.empty()) {
            return std::nullopt;
        }
        cv::Mat gray = loadGray(path);
        if (i == 0 && expected_size.empty()) {
            expected_size = gray.size();
        }
        if (gray.size() != expected_size) {
            return std::nullopt;
        }
        gray.convertTo(images[i], CV_32F);
    }
    return images;
}

std::optional<DecodedPhase> decodeFourStepPhase(
    const fs::path& raw_dir,
    int start_index,
    cv::Size expected_size,
    bool invert_phase,
    float phase_offset,
    const std::wstring& series = L"") {
    auto images = loadFourStepFrames(raw_dir, start_index, expected_size, series);
    if (!images) {
        return std::nullopt;
    }

    expected_size = (*images)[0].size();
    cv::Mat1f phase(expected_size, 0.0f);
    cv::Mat1f modulation(expected_size, 0.0f);
    const float sign = invert_phase ? -1.0f : 1.0f;
    for (int y = 0; y < expected_size.height; ++y) {
        const float* i1 = (*images)[0].ptr<float>(y);
        const float* i2 = (*images)[1].ptr<float>(y);
        const float* i3 = (*images)[2].ptr<float>(y);
        const float* i4 = (*images)[3].ptr<float>(y);
        float* ph = phase.ptr<float>(y);
        float* mod = modulation.ptr<float>(y);
        for (int x = 0; x < expected_size.width; ++x) {
            float c = (i1[x] - i3[x]) * 0.5f;
            float s = (i2[x] - i4[x]) * 0.5f;
            ph[x] = sign * std::atan2(s, c);
            ph[x] = wrapPi(ph[x] + phase_offset);
            mod[x] = 0.5f * std::sqrt(c * c + s * s);
        }
    }
    return DecodedPhase{phase, modulation, *images};
}

cv::Mat1f computeCircularCoherence(const cv::Mat1f& phase, int radius) {
    cv::Mat1f cos_phase(phase.rows, phase.cols);
    cv::Mat1f sin_phase(phase.rows, phase.cols);
    for (int y = 0; y < phase.rows; ++y) {
        const float* src = phase.ptr<float>(y);
        float* c = cos_phase.ptr<float>(y);
        float* s = sin_phase.ptr<float>(y);
        for (int x = 0; x < phase.cols; ++x) {
            c[x] = std::cos(src[x]);
            s[x] = std::sin(src[x]);
        }
    }

    cv::Mat1f mean_cos;
    cv::Mat1f mean_sin;
    const int ksize = radius * 2 + 1;
    cv::boxFilter(cos_phase, mean_cos, CV_32F, cv::Size(ksize, ksize), cv::Point(-1, -1), true, cv::BORDER_REPLICATE);
    cv::boxFilter(sin_phase, mean_sin, CV_32F, cv::Size(ksize, ksize), cv::Point(-1, -1), true, cv::BORDER_REPLICATE);

    cv::Mat1f coherence(phase.rows, phase.cols);
    for (int y = 0; y < phase.rows; ++y) {
        const float* c = mean_cos.ptr<float>(y);
        const float* s = mean_sin.ptr<float>(y);
        float* dst = coherence.ptr<float>(y);
        for (int x = 0; x < phase.cols; ++x) {
            dst[x] = std::sqrt(c[x] * c[x] + s[x] * s[x]);
        }
    }
    return coherence;
}

cv::Mat1f computeResidueDensity(const cv::Mat1f& phase, int radius) {
    cv::Mat1f residue(phase.rows, phase.cols, 0.0f);
    for (int y = 0; y + 1 < phase.rows; ++y) {
        for (int x = 0; x + 1 < phase.cols; ++x) {
            float d1 = wrapPi(phase(y, x + 1) - phase(y, x));
            float d2 = wrapPi(phase(y + 1, x + 1) - phase(y, x + 1));
            float d3 = wrapPi(phase(y + 1, x) - phase(y + 1, x + 1));
            float d4 = wrapPi(phase(y, x) - phase(y + 1, x));
            float closure = d1 + d2 + d3 + d4;
            if (std::abs(closure) > kPi) {
                residue(y, x) = 1.0f;
            }
        }
    }

    cv::Mat1f density;
    const int ksize = radius * 2 + 1;
    cv::boxFilter(residue, density, CV_32F, cv::Size(ksize, ksize), cv::Point(-1, -1), true, cv::BORDER_REPLICATE);
    return density;
}

cv::Mat1f computeReliability(const cv::Mat1f& phase, const cv::Mat1b& valid);
float percentile(std::vector<float> samples, double pct);

cv::Mat1b buildModulationMask(const std::array<cv::Mat1f, 4>& frames, const Config& cfg) {
    cv::Size size = frames[0].size();
    cv::Mat1f amplitude(size, 0.0f);
    std::vector<float> amplitude_samples;
    amplitude_samples.reserve(static_cast<size_t>(size.width) * static_cast<size_t>(size.height));
    int weak_count = 0;
    int total_count = 0;

    for (int y = 0; y < size.height; ++y) {
        const float* i1 = frames[0].ptr<float>(y);
        const float* i2 = frames[1].ptr<float>(y);
        const float* i3 = frames[2].ptr<float>(y);
        const float* i4 = frames[3].ptr<float>(y);
        float* amp = amplitude.ptr<float>(y);
        for (int x = 0; x < size.width; ++x) {
            float c = (i1[x] - i3[x]) * 0.5f;
            float s = (i2[x] - i4[x]) * 0.5f;
            float b = std::sqrt(c * c + s * s);

            amp[x] = b;
            amplitude_samples.push_back(b);
            if (b < cfg.modulation_abs_floor) {
                ++weak_count;
            }
            ++total_count;
        }
    }

    float threshold = cfg.modulation_abs_floor;
    double weak_ratio = total_count > 0 ? static_cast<double>(weak_count) / static_cast<double>(total_count) : 0.0;
    if (weak_ratio >= static_cast<double>(cfg.modulation_min_weak_ratio_for_otsu)) {
        float t_otsu = otsuFloat(amplitude);
        threshold = std::max(t_otsu, cfg.modulation_abs_floor);

        double cap_ratio = std::clamp(static_cast<double>(cfg.modulation_otsu_cap_pct) / 100.0, 0.0, 1.0);
        if (weak_ratio < cap_ratio) {
            float t_cap = percentile(amplitude_samples, cfg.modulation_otsu_cap_pct);
            threshold = std::max(std::min(threshold, t_cap), cfg.modulation_abs_floor);
        }
    }

    cv::Mat1b mask(size, uchar(0));
    for (int y = 0; y < size.height; ++y) {
        const float* amp = amplitude.ptr<float>(y);
        uchar* dst = mask.ptr<uchar>(y);
        for (int x = 0; x < size.width; ++x) {
            dst[x] = amp[x] >= threshold ? uchar(255) : uchar(0);
        }
    }

    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(11, 11)));
    if (cfg.modulation_erode_size > 0) {
        int erode_size = std::max(1, cfg.modulation_erode_size);
        cv::erode(mask, mask, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(erode_size, erode_size)));
    }
    const double foreground_ratio = static_cast<double>(cv::countNonZero(mask)) /
        static_cast<double>(size.width * size.height);
    std::cout << "  segment: modulation Otsu threshold=" << threshold
              << ", foreground=" << foreground_ratio * 100.0 << "%\n";
    return mask;
}

PixelQuality buildPixelQualityFromFourStepRaw(
    const std::array<cv::Mat1f, 4>& frames,
    const cv::Mat1f& phase,
    const Config& cfg) {

    PixelQuality quality;
    cv::Size size = phase.size();
    quality.amplitude = cv::Mat1f(size, 0.0f);
    quality.contrast = cv::Mat1f(size, 0.0f);
    quality.model_residual = cv::Mat1f(size, 0.0f);
    quality.coherence = computeCircularCoherence(phase, cfg.coherence_radius);
    quality.residue_density = computeResidueDensity(phase, cfg.residue_density_radius);
    quality.confidence = cv::Mat1f(size, 0.0f);
    quality.valid = cv::Mat1b(size, uchar(0));

    constexpr float eps = 1e-6f;
    for (int y = 0; y < phase.rows; ++y) {
        const float* i1 = frames[0].ptr<float>(y);
        const float* i2 = frames[1].ptr<float>(y);
        const float* i3 = frames[2].ptr<float>(y);
        const float* i4 = frames[3].ptr<float>(y);
        const float* coh = quality.coherence.ptr<float>(y);
        const float* res_density = quality.residue_density.ptr<float>(y);
        float* amp = quality.amplitude.ptr<float>(y);
        float* contrast = quality.contrast.ptr<float>(y);
        float* model = quality.model_residual.ptr<float>(y);
        float* conf = quality.confidence.ptr<float>(y);
        uchar* valid = quality.valid.ptr<uchar>(y);
        for (int x = 0; x < phase.cols; ++x) {
            float a = (i1[x] + i2[x] + i3[x] + i4[x]) * 0.25f;
            float c = i1[x] - i3[x];
            float s = i2[x] - i4[x];
            float b = std::sqrt(c * c + s * s);
            float r1 = i1[x] - (a + c);
            float r2 = i2[x] - (a + s);
            float r3 = i3[x] - (a - c);
            float r4 = i4[x] - (a - s);
            float normalized_residual = std::sqrt((r1 * r1 + r2 * r2 + r3 * r3 + r4 * r4) * 0.25f) / std::max(b, eps);

            float contrast_value = b / std::max(a, eps);

            float amplitude_score = smoothstep(cfg.min_amplitude, cfg.good_amplitude, b);
            float residual_score = 1.0f / (1.0f + normalized_residual / std::max(cfg.residual_scale, eps));
            float coherence_score = std::clamp((coh[x] - cfg.coherence_threshold) / std::max(1.0f - cfg.coherence_threshold, eps), 0.0f, 1.0f);
            float residue_score = std::clamp(1.0f - res_density[x] / std::max(cfg.max_residue_density, eps), 0.0f, 1.0f);

            amp[x] = b;
            contrast[x] = contrast_value;
            model[x] = normalized_residual;
            conf[x] =
                amplitude_score *
                residual_score *
                coherence_score *
                residue_score;
            valid[x] = conf[x] > cfg.confidence_threshold ? uchar(255) : uchar(0);
        }
    }
    return quality;
}

PixelQuality buildPixelQualityFromPhaseOnly(const cv::Mat1f& phase, const Config& cfg) {
    PixelQuality quality;
    quality.coherence = computeCircularCoherence(phase, cfg.coherence_radius);
    quality.residue_density = computeResidueDensity(phase, cfg.residue_density_radius);
    quality.confidence = cv::Mat1f(phase.rows, phase.cols, 0.0f);
    quality.valid = cv::Mat1b(phase.rows, phase.cols, uchar(0));

    cv::Mat1b all_valid(phase.rows, phase.cols, uchar(255));
    cv::Mat1f reliability = computeReliability(phase, all_valid);
    constexpr float max_reliability = 16.0f * kPi * kPi;
    constexpr float eps = 1e-6f;
    for (int y = 0; y < phase.rows; ++y) {
        const float* coh = quality.coherence.ptr<float>(y);
        const float* res_density = quality.residue_density.ptr<float>(y);
        const float* rel = reliability.ptr<float>(y);
        float* conf = quality.confidence.ptr<float>(y);
        uchar* valid = quality.valid.ptr<uchar>(y);
        for (int x = 0; x < phase.cols; ++x) {
            float coherence_score = std::clamp((coh[x] - cfg.coherence_threshold) / std::max(1.0f - cfg.coherence_threshold, eps), 0.0f, 1.0f);
            float residue_score = std::clamp(1.0f - res_density[x] / std::max(cfg.max_residue_density, eps), 0.0f, 1.0f);
            float reliability_score = std::clamp(rel[x] / max_reliability, 0.0f, 1.0f);
            conf[x] = coherence_score * residue_score * reliability_score;
            valid[x] = conf[x] > cfg.confidence_threshold ? uchar(255) : uchar(0);
        }
    }
    return quality;
}

cv::Mat1f computeReliability(const cv::Mat1f& phase, const cv::Mat1b& valid) {
    cv::Mat1f reliability(phase.rows, phase.cols, 0.0f);
    constexpr float max_reliability = 16.0f * kPi * kPi;

    for (int y = 1; y < phase.rows - 1; ++y) {
        const uchar* vm = valid.ptr<uchar>(y);
        for (int x = 1; x < phase.cols - 1; ++x) {
            if (vm[x] == 0) {
                continue;
            }

            auto good = [&](int yy, int xx) { return valid(yy, xx) != 0; };
            if (!good(y, x - 1) || !good(y, x + 1) || !good(y - 1, x) || !good(y + 1, x) ||
                !good(y - 1, x - 1) || !good(y + 1, x + 1) || !good(y - 1, x + 1) || !good(y + 1, x - 1)) {
                continue;
            }

            float c = phase(y, x);
            float h = wrapPi(phase(y, x - 1) - 2.0f * c + phase(y, x + 1));
            float v = wrapPi(phase(y - 1, x) - 2.0f * c + phase(y + 1, x));
            float d1 = wrapPi(phase(y - 1, x - 1) - 2.0f * c + phase(y + 1, x + 1));
            float d2 = wrapPi(phase(y - 1, x + 1) - 2.0f * c + phase(y + 1, x - 1));
            float error = h * h + v * v + d1 * d1 + d2 * d2;
            reliability(y, x) = std::max(0.0f, max_reliability - error);
        }
    }
    return reliability;
}

std::vector<Edge> buildEdges(const cv::Mat1f& reliability, const cv::Mat1b& valid, const cv::Mat1f* confidence = nullptr) {
    std::vector<Edge> edges;
    edges.reserve(static_cast<size_t>(valid.rows) * static_cast<size_t>(valid.cols));
    auto edgeWeight = [&](int y0, int x0, int y1, int x1) {
        float weight = 0.5f * (reliability(y0, x0) + reliability(y1, x1));
        if (confidence) {
            weight *= (*confidence)(y0, x0) * (*confidence)(y1, x1);
        } else {
            weight *= 2.0f;
        }
        return weight;
    };

    for (int y = 0; y < valid.rows; ++y) {
        const uchar* vm = valid.ptr<uchar>(y);
        for (int x = 0; x < valid.cols; ++x) {
            if (vm[x] == 0) {
                continue;
            }
            int index = y * valid.cols + x;
            if (x + 1 < valid.cols && valid(y, x + 1) != 0) {
                edges.push_back({index, index + 1, edgeWeight(y, x, y, x + 1)});
            }
            if (y + 1 < valid.rows && valid(y + 1, x) != 0) {
                edges.push_back({index, index + valid.cols, edgeWeight(y, x, y + 1, x)});
            }
        }
    }

    std::sort(edges.begin(), edges.end(), [](const Edge& lhs, const Edge& rhs) {
        return lhs.reliability > rhs.reliability;
    });
    return edges;
}

cv::Mat1f unwrapQualityGuided(const cv::Mat1f& phase, const cv::Mat1b& valid, const cv::Mat1f& reliability, const cv::Mat1f* confidence = nullptr) {
    const int width = phase.cols;
    const int height = phase.rows;
    const int n = width * height;

    UnionFind uf(n);
    for (int y = 0; y < height; ++y) {
        const uchar* vm = valid.ptr<uchar>(y);
        for (int x = 0; x < width; ++x) {
            if (vm[x] != 0) {
                uf.activate(y * width + x);
            }
        }
    }

    std::vector<Edge> edges = buildEdges(reliability, valid, confidence);
    for (const Edge& edge : edges) {
        int py = edge.a / width;
        int px = edge.a - py * width;
        int qy = edge.b / width;
        int qx = edge.b - qy * width;
        float p = phase(py, px);
        float q = phase(qy, qx);
        float wrapped_delta = wrapPi(q - p);
        int q_minus_p_periods = static_cast<int>(std::lround((p + wrapped_delta - q) / kTwoPi));
        uf.unite(edge.a, edge.b, q_minus_p_periods);
    }

    cv::Mat1f unwrapped(height, width, std::numeric_limits<float>::quiet_NaN());
    for (int y = 0; y < height; ++y) {
        const uchar* vm = valid.ptr<uchar>(y);
        float* dst = unwrapped.ptr<float>(y);
        for (int x = 0; x < width; ++x) {
            if (vm[x] == 0) {
                continue;
            }
            int index = y * width + x;
            int periods = uf.periodToRoot(index);
            dst[x] = phase(y, x) + kTwoPi * static_cast<float>(periods);
        }
    }
    return unwrapped;
}

struct AbsoluteKResult {
    cv::Mat1f unwrapped;
    cv::Mat1f carrier;
    cv::Mat1f residual;
    cv::Mat1f k_order;
};

std::optional<AbsoluteKResult> unwrapWithAbsoluteKPlane(
    const cv::Mat1f& wrapped,
    const cv::Mat1f& integrated_unwrapped,
    const cv::Mat1b& valid,
    const std::vector<PhaseAnchorROI>& anchors,
    float phase_fit_threshold) {

    if (anchors.empty()) {
        return std::nullopt;
    }
    const double cx = (wrapped.cols - 1) * 0.5;
    const double cy = (wrapped.rows - 1) * 0.5;
    const double coordinate_scale = static_cast<double>(std::max(wrapped.cols, wrapped.rows));
    const double phase_limit = std::clamp(static_cast<double>(phase_fit_threshold), 0.0, 1.0) * kPi;
    cv::Matx33d lhs = cv::Matx33d::zeros();
    cv::Vec3d rhs(0.0, 0.0, 0.0);
    long long sample_count = 0;

    const cv::Rect image_rect(0, 0, wrapped.cols, wrapped.rows);
    for (const PhaseAnchorROI& anchor : anchors) {
        const cv::Rect roi = anchor.rect & image_rect;
        for (int y = roi.y; y < roi.y + roi.height; ++y) {
            const float* phase_row = wrapped.ptr<float>(y);
            const uchar* valid_row = valid.ptr<uchar>(y);
            for (int x = roi.x; x < roi.x + roi.width; ++x) {
                const double phi = phase_row[x];
                if (valid_row[x] == 0 || !std::isfinite(phi) || std::abs(phi) >= phase_limit) {
                    continue;
                }
                const double xn = (static_cast<double>(x) - cx) / coordinate_scale;
                const double yn = (static_cast<double>(y) - cy) / coordinate_scale;
                const cv::Vec3d basis(xn, yn, 1.0);
                const double absolute_phase = phi + static_cast<double>(anchor.k_order) * kTwoPi;
                lhs += basis * basis.t();
                rhs += basis * absolute_phase;
                ++sample_count;
            }
        }
    }
    if (sample_count < 3) {
        throw std::runtime_error("absolute-K plane has fewer than three stable anchor samples");
    }

    cv::Vec3d coeff;
    if (!cv::solve(lhs, rhs, coeff, cv::DECOMP_SVD)) {
        throw std::runtime_error("failed to fit absolute-K carrier plane");
    }
    std::cout << "  absolute K: normalized SVD carrier, anchors=" << anchors.size()
              << ", samples=" << sample_count
              << ", coeff=[" << coeff[0] << ", " << coeff[1] << ", " << coeff[2] << "]\n";

    const float nan = std::numeric_limits<float>::quiet_NaN();
    AbsoluteKResult result{
        cv::Mat1f(wrapped.size(), nan),
        cv::Mat1f(wrapped.size(), nan),
        cv::Mat1f(wrapped.size(), nan),
        cv::Mat1f(wrapped.size(), nan)};
    long long integrated_count = 0;
    long long fallback_count = 0;
    for (int y = 0; y < wrapped.rows; ++y) {
        const float* phase_row = wrapped.ptr<float>(y);
        const float* integrated_row = integrated_unwrapped.ptr<float>(y);
        const uchar* valid_row = valid.ptr<uchar>(y);
        float* unwrapped_row = result.unwrapped.ptr<float>(y);
        float* carrier_row = result.carrier.ptr<float>(y);
        float* residual_row = result.residual.ptr<float>(y);
        float* k_row = result.k_order.ptr<float>(y);
        const double yn = (static_cast<double>(y) - cy) / coordinate_scale;
        for (int x = 0; x < wrapped.cols; ++x) {
            // The confidence mask controls carrier fitting, not output coverage.
            // For rejected samples, retain the original wrapped phase and use
            // the fitted carrier to infer K before subtracting the carrier.
            if (!std::isfinite(phase_row[x])) {
                continue;
            }
            const double xn = (static_cast<double>(x) - cx) / coordinate_scale;
            const double carrier = coeff[0] * xn + coeff[1] * yn + coeff[2];
            double absolute_phase = 0.0;
            if (valid_row[x] != 0 && std::isfinite(integrated_row[x])) {
                const int component_offset = static_cast<int>(
                    std::lround((carrier - integrated_row[x]) / kTwoPi));
                absolute_phase = integrated_row[x] + static_cast<double>(component_offset) * kTwoPi;
                ++integrated_count;
            } else {
                const int inferred_k = static_cast<int>(
                    std::lround((carrier - phase_row[x]) / kTwoPi));
                absolute_phase = phase_row[x] + static_cast<double>(inferred_k) * kTwoPi;
                ++fallback_count;
            }
            const int k = static_cast<int>(std::lround((absolute_phase - phase_row[x]) / kTwoPi));
            carrier_row[x] = static_cast<float>(carrier);
            k_row[x] = static_cast<float>(k);
            unwrapped_row[x] = static_cast<float>(absolute_phase);
            residual_row[x] = static_cast<float>(absolute_phase - carrier);
        }
    }
    std::cout << "  absolute K merge: integrated=" << integrated_count
              << ", wrapped-phase fallback=" << fallback_count << "\n";
    return result;
}

std::vector<float> collectValid(const cv::Mat1f& values, const cv::Mat1b& valid, bool absolute_value = false) {
    std::vector<float> samples;
    samples.reserve(static_cast<size_t>(cv::countNonZero(valid)));
    for (int y = 0; y < values.rows; ++y) {
        const float* row = values.ptr<float>(y);
        const uchar* vm = valid.ptr<uchar>(y);
        for (int x = 0; x < values.cols; ++x) {
            if (vm[x] == 0 || !std::isfinite(row[x])) {
                continue;
            }
            samples.push_back(absolute_value ? std::abs(row[x]) : row[x]);
        }
    }
    return samples;
}

float percentile(std::vector<float> samples, double pct) {
    if (samples.empty()) {
        return 0.0f;
    }
    std::sort(samples.begin(), samples.end());
    double pos = (pct / 100.0) * static_cast<double>(samples.size() - 1);
    size_t lo = static_cast<size_t>(std::floor(pos));
    size_t hi = static_cast<size_t>(std::ceil(pos));
    double t = pos - static_cast<double>(lo);
    return static_cast<float>(samples[lo] * (1.0 - t) + samples[hi] * t);
}

cv::Mat1b normalizeVisual(const cv::Mat1f& values, const cv::Mat1b& valid, double low_pct = 1.0, double high_pct = 99.0) {
    auto samples = collectValid(values, valid);
    float lo = percentile(samples, low_pct);
    float hi = percentile(samples, high_pct);
    if (hi <= lo) {
        hi = lo + 1.0f;
    }

    cv::Mat1b out(values.rows, values.cols, uchar(0));
    for (int y = 0; y < values.rows; ++y) {
        const float* src = values.ptr<float>(y);
        const uchar* vm = valid.ptr<uchar>(y);
        uchar* dst = out.ptr<uchar>(y);
        for (int x = 0; x < values.cols; ++x) {
            if (vm[x] == 0 || !std::isfinite(src[x])) {
                continue;
            }
            float v = std::clamp((src[x] - lo) / (hi - lo), 0.0f, 1.0f);
            dst[x] = static_cast<uchar>(std::lround(v * 255.0f));
        }
    }
    return out;
}

cv::Mat1b wrappedPhaseVisual(const cv::Mat1f& phase) {
    cv::Mat1b out(phase.rows, phase.cols, uchar(0));
    constexpr float scale = 255.0f / kTwoPi;
    for (int y = 0; y < phase.rows; ++y) {
        const float* src = phase.ptr<float>(y);
        uchar* dst = out.ptr<uchar>(y);
        for (int x = 0; x < phase.cols; ++x) {
            if (!std::isfinite(src[x])) {
                continue;
            }
            float v = (wrapPi(src[x]) + kPi) * scale;
            dst[x] = static_cast<uchar>(std::lround(std::clamp(v, 0.0f, 255.0f)));
        }
    }
    return out;
}

cv::Mat1b compositeWithBackground(const cv::Mat1b& foreground, const cv::Mat1b& background, const cv::Mat1b& valid) {
    cv::Mat1b out = background.clone();
    for (int y = 0; y < foreground.rows; ++y) {
        const uchar* fg = foreground.ptr<uchar>(y);
        const uchar* vm = valid.ptr<uchar>(y);
        uchar* dst = out.ptr<uchar>(y);
        for (int x = 0; x < foreground.cols; ++x) {
            if (vm[x] != 0) {
                dst[x] = fg[x];
            }
        }
    }
    return out;
}

cv::Mat1b overlaySignedDetailByThreshold(
    const cv::Mat1b& base,
    const cv::Mat1b& signed_detail,
    const cv::Mat1b& valid,
    int threshold) {

    cv::Mat1b out = base.clone();
    threshold = std::max(0, threshold);
    for (int y = 0; y < base.rows; ++y) {
        const uchar* detail = signed_detail.ptr<uchar>(y);
        const uchar* vm = valid.ptr<uchar>(y);
        uchar* dst = out.ptr<uchar>(y);
        for (int x = 0; x < base.cols; ++x) {
            if (vm[x] != 0 && std::abs(static_cast<int>(detail[x]) - 128) > threshold) {
                dst[x] = detail[x];
            }
        }
    }
    return out;
}

cv::Mat1f localWeightedMean(const cv::Mat1f& values, const cv::Mat1f& weights, int radius);
cv::Mat1f validToWeights(const cv::Mat1f& values, const cv::Mat1b& valid);

cv::Mat1f planeDetrended(
    const cv::Mat1f& values,
    const cv::Mat1f& wrapped_phase,
    const cv::Mat1b& valid,
    float phase_fit_threshold) {

    // Centered, normalized coordinates keep the least-squares system well
    // conditioned on multi-megapixel images. Wrapped samples near +/-pi are
    // excluded because four-step decoding is least stable at the branch cut.
    const double cx = (values.cols - 1) * 0.5;
    const double cy = (values.rows - 1) * 0.5;
    const double coordinate_scale = static_cast<double>(std::max(values.cols, values.rows));
    const double phase_limit = std::clamp(static_cast<double>(phase_fit_threshold), 0.0, 1.0) * kPi;
    cv::Matx33d lhs = cv::Matx33d::zeros();
    cv::Vec3d rhs(0.0, 0.0, 0.0);
    double n = 0.0;

    for (int y = 0; y < values.rows; ++y) {
        const float* row = values.ptr<float>(y);
        const float* phase_row = wrapped_phase.ptr<float>(y);
        const uchar* vm = valid.ptr<uchar>(y);
        for (int x = 0; x < values.cols; ++x) {
            if (vm[x] == 0 || !std::isfinite(row[x]) || !std::isfinite(phase_row[x]) ||
                std::abs(phase_row[x]) >= phase_limit) {
                continue;
            }
            const double xd = (static_cast<double>(x) - cx) / coordinate_scale;
            const double yd = (static_cast<double>(y) - cy) / coordinate_scale;
            const double zd = static_cast<double>(row[x]);
            const cv::Vec3d basis(xd, yd, 1.0);
            lhs += basis * basis.t();
            rhs += basis * zd;
            n += 1.0;
        }
    }

    if (n < 3.0) {
        return values.clone();
    }

    cv::Vec3d coeff;
    if (!cv::solve(lhs, rhs, coeff, cv::DECOMP_SVD)) {
        return values.clone();
    }
    std::cout << "  trend: normalized SVD plane, stable samples="
              << static_cast<long long>(n)
              << ", coeff=[" << coeff[0] << ", " << coeff[1] << ", " << coeff[2] << "]\n";

    float center = percentile(collectValid(values, valid), 50.0);
    cv::Mat1f out(values.rows, values.cols, std::numeric_limits<float>::quiet_NaN());
    for (int y = 0; y < values.rows; ++y) {
        const float* src = values.ptr<float>(y);
        const uchar* vm = valid.ptr<uchar>(y);
        float* dst = out.ptr<float>(y);
        for (int x = 0; x < values.cols; ++x) {
            if (vm[x] == 0 || !std::isfinite(src[x])) {
                continue;
            }
            const double xn = (static_cast<double>(x) - cx) / coordinate_scale;
            const double yn = (static_cast<double>(y) - cy) / coordinate_scale;
            double plane = coeff[0] * xn + coeff[1] * yn + coeff[2];
            dst[x] = static_cast<float>(static_cast<double>(src[x]) - plane + center);
        }
    }
    return out;
}

cv::Mat1f localMean(const cv::Mat1f& values, const cv::Mat1b& valid, int radius) {
    return localWeightedMean(values, validToWeights(values, valid), radius);
}

cv::Mat1f circularLocalResidual(const cv::Mat1f& phase, const cv::Mat1b& valid, int radius) {
    cv::Mat1f cos_phase(phase.rows, phase.cols, 0.0f);
    cv::Mat1f sin_phase(phase.rows, phase.cols, 0.0f);
    for (int y = 0; y < phase.rows; ++y) {
        const float* src = phase.ptr<float>(y);
        const uchar* vm = valid.ptr<uchar>(y);
        float* c = cos_phase.ptr<float>(y);
        float* s = sin_phase.ptr<float>(y);
        for (int x = 0; x < phase.cols; ++x) {
            if (vm[x] != 0 && std::isfinite(src[x])) {
                c[x] = std::cos(src[x]);
                s[x] = std::sin(src[x]);
            }
        }
    }

    cv::Mat1f mean_cos = localMean(cos_phase, valid, radius);
    cv::Mat1f mean_sin = localMean(sin_phase, valid, radius);
    cv::Mat1f residual(phase.rows, phase.cols, std::numeric_limits<float>::quiet_NaN());
    for (int y = 0; y < phase.rows; ++y) {
        const float* src = phase.ptr<float>(y);
        const uchar* vm = valid.ptr<uchar>(y);
        const float* mc = mean_cos.ptr<float>(y);
        const float* ms = mean_sin.ptr<float>(y);
        float* dst = residual.ptr<float>(y);
        for (int x = 0; x < phase.cols; ++x) {
            if (vm[x] == 0 || !std::isfinite(src[x])) {
                continue;
            }
            dst[x] = wrapPi(src[x] - std::atan2(ms[x], mc[x]));
        }
    }
    return residual;
}

cv::Mat1f localWeightedMean(const cv::Mat1f& values, const cv::Mat1f& weights, int radius) {
    cv::Mat1f weighted(values.rows, values.cols, 0.0f);
    for (int y = 0; y < values.rows; ++y) {
        const float* src = values.ptr<float>(y);
        const float* w = weights.ptr<float>(y);
        float* dst = weighted.ptr<float>(y);
        for (int x = 0; x < values.cols; ++x) {
            if (w[x] > 0.0f && std::isfinite(src[x])) {
                dst[x] = src[x] * w[x];
            }
        }
    }

    const int ksize = radius * 2 + 1;
    cv::Mat1f sum_values;
    cv::Mat1f sum_weights;
    cv::boxFilter(weighted, sum_values, CV_32F, cv::Size(ksize, ksize), cv::Point(-1, -1), false, cv::BORDER_REPLICATE);
    cv::boxFilter(weights, sum_weights, CV_32F, cv::Size(ksize, ksize), cv::Point(-1, -1), false, cv::BORDER_REPLICATE);

    cv::Mat1f mean(values.rows, values.cols, 0.0f);
    for (int y = 0; y < values.rows; ++y) {
        const float* sv = sum_values.ptr<float>(y);
        const float* sw = sum_weights.ptr<float>(y);
        float* dst = mean.ptr<float>(y);
        for (int x = 0; x < values.cols; ++x) {
            dst[x] = sw[x] > 1e-6f ? sv[x] / sw[x] : 0.0f;
        }
    }
    return mean;
}

cv::Mat1f subtractLocalBias(const cv::Mat1f& residual, const cv::Mat1b& valid, int radius) {
    cv::Mat1f bias = localMean(residual, valid, radius);
    cv::Mat1f out(residual.rows, residual.cols, std::numeric_limits<float>::quiet_NaN());
    for (int y = 0; y < residual.rows; ++y) {
        const float* src = residual.ptr<float>(y);
        const float* bg = bias.ptr<float>(y);
        const uchar* vm = valid.ptr<uchar>(y);
        float* dst = out.ptr<float>(y);
        for (int x = 0; x < residual.cols; ++x) {
            if (vm[x] != 0 && std::isfinite(src[x])) {
                dst[x] = src[x] - bg[x];
            }
        }
    }
    return out;
}

cv::Mat1f validToWeights(const cv::Mat1f& values, const cv::Mat1b& valid) {
    cv::Mat1f weights(values.rows, values.cols, 0.0f);
    for (int y = 0; y < values.rows; ++y) {
        const float* src = values.ptr<float>(y);
        const uchar* vm = valid.ptr<uchar>(y);
        float* dst = weights.ptr<float>(y);
        for (int x = 0; x < values.cols; ++x) {
            if (vm[x] != 0 && std::isfinite(src[x])) {
                dst[x] = 1.0f;
            }
        }
    }
    return weights;
}

cv::Mat1f robustLocalBackground(
    const cv::Mat1f& values,
    const cv::Mat1b& valid,
    int radius,
    int iterations,
    float outlier_scale) {

    cv::Mat1f base_weights = validToWeights(values, valid);
    cv::Mat1f weights = base_weights.clone();
    cv::Mat1f background = localWeightedMean(values, weights, radius);
    outlier_scale = std::max(outlier_scale, 0.1f);

    for (int iter = 0; iter < std::max(1, iterations); ++iter) {
        cv::Mat1f residual(values.rows, values.cols, std::numeric_limits<float>::quiet_NaN());
        for (int y = 0; y < values.rows; ++y) {
            const float* src = values.ptr<float>(y);
            const float* bg = background.ptr<float>(y);
            const uchar* vm = valid.ptr<uchar>(y);
            float* dst = residual.ptr<float>(y);
            for (int x = 0; x < values.cols; ++x) {
                if (vm[x] != 0 && std::isfinite(src[x]) && std::isfinite(bg[x])) {
                    dst[x] = src[x] - bg[x];
                }
            }
        }

        float scale = percentile(collectValid(residual, valid, true), 85.0) * outlier_scale;
        if (scale <= 1e-6f) {
            break;
        }

        for (int y = 0; y < values.rows; ++y) {
            const float* r = residual.ptr<float>(y);
            const float* bw = base_weights.ptr<float>(y);
            float* w = weights.ptr<float>(y);
            for (int x = 0; x < values.cols; ++x) {
                if (bw[x] <= 0.0f || !std::isfinite(r[x])) {
                    w[x] = 0.0f;
                    continue;
                }
                float t = std::abs(r[x]) / scale;
                w[x] = bw[x] / (1.0f + t * t);
            }
        }
        background = localWeightedMean(values, weights, radius);
    }
    return background;
}

cv::Mat1b deltaVisual(const cv::Mat1f& delta, const cv::Mat1b& valid, bool signed_output) {
    float scale = percentile(collectValid(delta, valid, true), 99.8);
    if (scale <= 1e-6f) {
        scale = 1.0f;
    }
    cv::Mat1b out(delta.rows, delta.cols, uchar(0));
    for (int y = 0; y < delta.rows; ++y) {
        const float* src = delta.ptr<float>(y);
        const uchar* vm = valid.ptr<uchar>(y);
        uchar* dst = out.ptr<uchar>(y);
        for (int x = 0; x < delta.cols; ++x) {
            if (vm[x] == 0 || !std::isfinite(src[x])) {
                continue;
            }
            float v = signed_output
                ? std::clamp(127.5f + src[x] * (127.5f / scale), 0.0f, 255.0f)
                : std::clamp(std::abs(src[x]) / scale, 0.0f, 1.0f) * 255.0f;
            dst[x] = static_cast<uchar>(std::lround(v));
        }
    }
    return out;
}

void processOne(const fs::path& phase_path, const fs::path& raw_dir, const fs::path& out_dir, const std::wstring& prefix, const Config& cfg) {
    std::wcout << L"[phase_unwrap] " << (cfg.decode_raw_phase ? raw_dir.wstring() : phase_path.wstring()) << L"\n";
    cv::Mat gray;
    cv::Mat1f phase;
    cv::Mat1f modulation;
    std::optional<std::array<cv::Mat1f, 4>> raw_frames;
    if (cfg.decode_raw_phase) {
        auto decoded = decodeFourStepPhase(
            raw_dir,
            cfg.raw_phase_start,
            cv::Size(),
            cfg.invert_phase,
            cfg.raw_phase_offset,
            cfg.raw_phase_series);
        if (!decoded) {
            throw std::runtime_error("failed to decode four-step RAW phase: " + narrowAscii(raw_dir));
        }
        phase = decoded->phase;
        modulation = decoded->modulation;
        raw_frames = decoded->frames;
        std::cout << "  phase: four-step RAW decode (no phase-image fallback)\n";
    } else {
        gray = loadGray(phase_path);
        phase = grayToPhase(gray, cfg.invert_phase);
        std::cout << "  phase: encoded phase image\n";
    }

    fs::create_directories(out_dir);
    fs::path sample_dir = out_dir / prefix;
    fs::create_directories(sample_dir);
    if (!cfg.config_path.empty()) {
        auto config_bytes = readFileBytes(cfg.config_path);
        if (config_bytes) {
            writeFileBytes(sample_dir / L"run_config.cfg", *config_bytes);
        }
    }

    if (!raw_frames) {
        raw_frames = loadFourStepFrames(raw_dir, cfg.raw_phase_start, phase.size(), cfg.raw_phase_series);
    }

    std::optional<cv::Mat1b> modulation_mask;
    if (raw_frames && cfg.use_modulation_mask) {
        modulation_mask = buildModulationMask(*raw_frames, cfg);
    }
    PixelQuality quality;
    if (raw_frames) {
        quality = buildPixelQualityFromFourStepRaw(*raw_frames, phase, cfg);
        if (modulation.empty()) {
            modulation = quality.amplitude;
        }
        std::cout << "  mask: multi-source RAW confidence\n";
    } else {
        quality = buildPixelQualityFromPhaseOnly(phase, cfg);
        std::cout << "  mask: phase-only confidence\n";
    }
    cv::Mat1b valid = modulation_mask ? modulation_mask->clone() : quality.valid;
    if (cfg.integration_roi) {
        cv::Mat1b roi_mask(phase.size(), uchar(0));
        const cv::Rect clipped = *cfg.integration_roi & cv::Rect(0, 0, phase.cols, phase.rows);
        if (clipped.empty()) {
            throw std::runtime_error("integration_roi does not intersect the phase image");
        }
        roi_mask(clipped).setTo(uchar(255));
        cv::bitwise_and(valid, roi_mask, valid);
        std::cout << "  integration ROI: x=" << clipped.x << ".." << clipped.x + clipped.width - 1
                  << ", y=" << clipped.y << ".." << clipped.y + clipped.height - 1 << "\n";
    }
    const double valid_ratio = static_cast<double>(cv::countNonZero(valid)) /
        static_cast<double>(valid.rows * valid.cols);
    std::cout << "  integration mask: "
              << (modulation_mask ? "Otsu modulation" : "per-pixel quality confidence")
              << ", valid=" << valid_ratio * 100.0 << "%\n";

    cv::Mat1f reliability = computeReliability(phase, valid);
    cv::Mat1f unwrapped = unwrapQualityGuided(phase, valid, reliability, &quality.confidence);
    std::cout << "  unwrap: reliability-guided union-find\n";
    std::optional<AbsoluteKResult> absolute_k;
    cv::Mat1b output_valid = valid.clone();
    if (cfg.absolute_k_enabled) {
        absolute_k = unwrapWithAbsoluteKPlane(
            phase,
            unwrapped,
            valid,
            cfg.absolute_k_rois,
            cfg.trend_fit_phase_threshold);
        if (!absolute_k) {
            throw std::runtime_error("absolute_k_enabled requires at least one absolute_k_roi");
        }
        unwrapped = absolute_k->unwrapped;
        output_valid = cv::Mat1b(phase.size(), uchar(255));
        std::cout << "  unwrap: absolute K anchored by carrier plane\n";
        std::cout << "  output coverage: full phase image; rejected integration samples use wrapped phase + inferred K\n";
    }
    cv::Mat1f delta = circularLocalResidual(phase, output_valid, cfg.residual_radius);
    cv::Mat1f display_delta = subtractLocalBias(delta, output_valid, std::max(181, cfg.residual_radius * 3));
    cv::Mat1f phase_background = absolute_k
        ? absolute_k->carrier
        : robustLocalBackground(
            unwrapped,
            output_valid,
            cfg.background_radius,
            cfg.background_iterations,
            cfg.background_outlier_scale);
    cv::Mat1f phase_residual = absolute_k ? absolute_k->residual : unwrapped - phase_background;
    cv::Mat1f phase_relief = cfg.relief_smooth_radius > 0
        ? localMean(phase_residual, output_valid, cfg.relief_smooth_radius)
        : phase_residual;
    cv::Mat1f phase_unwrapped_detrended = planeDetrended(
        unwrapped,
        phase,
        output_valid,
        cfg.trend_fit_phase_threshold);

    cv::Mat1b phase_unwrapped_visual = normalizeVisual(phase_unwrapped_detrended, output_valid);
    cv::Mat1b phase_wrapped_visual = wrappedPhaseVisual(phase);
    cv::Mat1b defect_delta_raw_visual = deltaVisual(delta, output_valid, true);
    cv::Mat1b defect_delta_visual = deltaVisual(display_delta, output_valid, true);
    cv::Mat1b phase_unwrapped_with_background = compositeWithBackground(phase_unwrapped_visual, phase_wrapped_visual, output_valid);

    saveImage(sample_dir / L"defect_delta.png", defect_delta_visual);
    saveImage(sample_dir / L"defect_delta_raw.png", defect_delta_raw_visual);
    saveImage(sample_dir / L"phase_relief.png", deltaVisual(phase_relief, output_valid, true));
    saveImage(sample_dir / L"phase_unwrapped.png", phase_unwrapped_visual);
    saveImage(sample_dir / L"phase_unwrapped_with_background.png", phase_unwrapped_with_background);
    saveImage(sample_dir / L"phase_wrapped.png", phase_wrapped_visual);
    if (absolute_k) {
        saveImage(sample_dir / L"phase_absolute_k.png", normalizeVisual(absolute_k->k_order, output_valid));
        saveImage(sample_dir / L"phase_absolute_k_removed.png", deltaVisual(absolute_k->residual, output_valid, true));
        saveImage(sample_dir / L"phase_absolute_k.tiff", absolute_k->k_order);
        saveImage(sample_dir / L"phase_absolute_k_removed.tiff", absolute_k->residual);
    }
    saveImage(sample_dir / L"phase_unwrapped_with_background_detail.png", overlaySignedDetailByThreshold(
        phase_unwrapped_with_background,
        defect_delta_visual,
        output_valid,
        cfg.detail_overlay_threshold));
}

std::string trimAscii(const std::string& text) {
    size_t first = 0;
    while (first < text.size() && std::isspace(static_cast<unsigned char>(text[first]))) {
        ++first;
    }
    size_t last = text.size();
    while (last > first && std::isspace(static_cast<unsigned char>(text[last - 1]))) {
        --last;
    }
    return text.substr(first, last - first);
}

std::string normalizeConfigKey(std::string key) {
    key = trimAscii(key);
    for (char& ch : key) {
        if (ch == '-') {
            ch = '_';
        } else {
            ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        }
    }
    return key;
}

std::wstring utf8ToWide(const std::string& text) {
    if (text.empty()) {
        return {};
    }
    int length = MultiByteToWideChar(
        CP_UTF8,
        MB_ERR_INVALID_CHARS,
        text.data(),
        static_cast<int>(text.size()),
        nullptr,
        0);
    if (length <= 0) {
        throw std::runtime_error("invalid UTF-8 text in config");
    }

    std::wstring out(static_cast<size_t>(length), L'\0');
    int written = MultiByteToWideChar(
        CP_UTF8,
        MB_ERR_INVALID_CHARS,
        text.data(),
        static_cast<int>(text.size()),
        out.data(),
        length);
    if (written != length) {
        throw std::runtime_error("failed to convert UTF-8 config text");
    }
    return out;
}

bool parseBoolValue(const std::string& text) {
    std::string value = normalizeConfigKey(text);
    if (value == "1" || value == "true" || value == "yes" || value == "on") {
        return true;
    }
    if (value == "0" || value == "false" || value == "no" || value == "off") {
        return false;
    }
    throw std::runtime_error("invalid boolean value: " + text);
}

fs::path parsePathValue(const std::string& text) {
    return fs::path(utf8ToWide(trimAscii(text)));
}

std::wstring parseWideValue(const std::string& text) {
    return utf8ToWide(trimAscii(text));
}

PhaseAnchorROI parsePhaseAnchorROI(std::string text) {
    std::replace(text.begin(), text.end(), ',', ' ');
    std::istringstream input(text);
    int x = 0;
    int y = 0;
    int width = 0;
    int height = 0;
    int k = 0;
    if (!(input >> x >> y >> width >> height >> k) || width <= 0 || height <= 0) {
        throw std::runtime_error("absolute_k_roi must be x,y,width,height,k");
    }
    return PhaseAnchorROI{cv::Rect(x, y, width, height), k};
}

cv::Rect parseRect(std::string text, const char* key) {
    std::replace(text.begin(), text.end(), ',', ' ');
    std::istringstream input(text);
    int x = 0;
    int y = 0;
    int width = 0;
    int height = 0;
    if (!(input >> x >> y >> width >> height) || width <= 0 || height <= 0) {
        throw std::runtime_error(std::string(key) + " must be x,y,width,height");
    }
    return cv::Rect(x, y, width, height);
}

void applyConfigValue(Config& cfg, const std::string& key, const std::string& value) {
    if (key == "phase" || key == "phase_path") {
        cfg.phase_path = parsePathValue(value);
    } else if (key == "raw_dir") {
        cfg.raw_dir = parsePathValue(value);
    } else if (key == "out_dir") {
        cfg.out_dir = parsePathValue(value);
    } else if (key == "prefix") {
        cfg.prefix = parseWideValue(value);
    } else if (key == "batch") {
        cfg.batch = parseBoolValue(value);
    } else if (key == "batch_root") {
        cfg.batch_root = parsePathValue(value);
    } else if (key == "invert_phase") {
        cfg.invert_phase = parseBoolValue(value);
    } else if (key == "raw_phase_offset") {
        cfg.raw_phase_offset = std::stof(value);
    } else if (key == "decode_raw_phase") {
        cfg.decode_raw_phase = parseBoolValue(value);
    } else if (key == "raw_phase_start") {
        cfg.raw_phase_start = std::stoi(value);
    } else if (key == "raw_phase_series") {
        cfg.raw_phase_series = parseWideValue(value);
    } else if (key == "coherence_threshold") {
        cfg.coherence_threshold = std::stof(value);
    } else if (key == "modulation_abs_floor") {
        cfg.modulation_abs_floor = std::stof(value);
    } else if (key == "modulation_min_weak_ratio_for_otsu") {
        cfg.modulation_min_weak_ratio_for_otsu = std::stof(value);
    } else if (key == "modulation_otsu_cap_pct") {
        cfg.modulation_otsu_cap_pct = std::stof(value);
    } else if (key == "modulation_erode_size") {
        cfg.modulation_erode_size = std::stoi(value);
    } else if (key == "use_modulation_mask") {
        cfg.use_modulation_mask = parseBoolValue(value);
    } else if (key == "absolute_k_enabled") {
        cfg.absolute_k_enabled = parseBoolValue(value);
    } else if (key == "absolute_k_roi") {
        cfg.absolute_k_rois.push_back(parsePhaseAnchorROI(value));
    } else if (key == "integration_roi") {
        cfg.integration_roi = parseRect(value, "integration_roi");
    } else if (key == "min_amplitude") {
        cfg.min_amplitude = std::stof(value);
    } else if (key == "good_amplitude") {
        cfg.good_amplitude = std::stof(value);
    } else if (key == "residual_scale") {
        cfg.residual_scale = std::stof(value);
    } else if (key == "confidence_threshold") {
        cfg.confidence_threshold = std::stof(value);
    } else if (key == "max_residue_density") {
        cfg.max_residue_density = std::stof(value);
    } else if (key == "coherence_radius") {
        cfg.coherence_radius = std::stoi(value);
    } else if (key == "residue_density_radius") {
        cfg.residue_density_radius = std::stoi(value);
    } else if (key == "residual_radius") {
        cfg.residual_radius = std::stoi(value);
    } else if (key == "background_radius") {
        cfg.background_radius = std::stoi(value);
    } else if (key == "background_iterations") {
        cfg.background_iterations = std::stoi(value);
    } else if (key == "relief_smooth_radius") {
        cfg.relief_smooth_radius = std::stoi(value);
    } else if (key == "detail_overlay_threshold") {
        cfg.detail_overlay_threshold = std::stoi(value);
    } else if (key == "background_outlier_scale") {
        cfg.background_outlier_scale = std::stof(value);
    } else if (key == "phase_enhance_gain") {
        cfg.phase_enhance_gain = std::stof(value);
    } else if (key == "phase_display_low_pct") {
        cfg.phase_display_low_pct = std::stof(value);
    } else if (key == "phase_display_high_pct") {
        cfg.phase_display_high_pct = std::stof(value);
    } else if (key == "competitor_absolute_low_pct") {
        cfg.competitor_absolute_low_pct = std::stof(value);
    } else if (key == "competitor_absolute_high_pct") {
        cfg.competitor_absolute_high_pct = std::stof(value);
    } else if (key == "trend_fit_phase_threshold") {
        cfg.trend_fit_phase_threshold = std::stof(value);
    } else {
        throw std::runtime_error("unknown config key: " + key);
    }
}

Config readConfigFile(const fs::path& path) {
    Config cfg;
    cfg.config_path = path;
    auto bytes = readFileBytes(path);
    if (!bytes) {
        throw std::runtime_error("failed to read config: " + narrowAscii(path));
    }

    std::string text(bytes->begin(), bytes->end());
    if (text.size() >= 3 &&
        static_cast<unsigned char>(text[0]) == 0xEF &&
        static_cast<unsigned char>(text[1]) == 0xBB &&
        static_cast<unsigned char>(text[2]) == 0xBF) {
        text.erase(0, 3);
    }

    size_t line_start = 0;
    int line_number = 1;
    while (line_start <= text.size()) {
        size_t line_end = text.find('\n', line_start);
        if (line_end == std::string::npos) {
            line_end = text.size();
        }

        std::string line = text.substr(line_start, line_end - line_start);
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        size_t comment = line.find('#');
        if (comment != std::string::npos) {
            line = line.substr(0, comment);
        }
        line = trimAscii(line);
        if (!line.empty()) {
            size_t eq = line.find('=');
            if (eq == std::string::npos) {
                throw std::runtime_error("invalid config line " + std::to_string(line_number) + ": expected key=value");
            }
            std::string key = normalizeConfigKey(line.substr(0, eq));
            std::string value = trimAscii(line.substr(eq + 1));
            applyConfigValue(cfg, key, value);
        }

        if (line_end == text.size()) {
            break;
        }
        line_start = line_end + 1;
        ++line_number;
    }
    return cfg;
}

int wmain(int argc, wchar_t** argv) {
    try {
        if (argc != 2) {
            std::cerr << "error: expected one config file path\n";
            return 2;
        }

        Config cfg = readConfigFile(argv[1]);
        if (cfg.batch) {
            int index = 1;
            for (const auto& entry : fs::directory_iterator(cfg.batch_root)) {
                if (!entry.is_directory()) {
                    continue;
                }
                fs::path phase = entry.path() / L"PHASEX.bmp";
                if (fs::exists(phase)) {
                    processOne(phase, entry.path() / L"RAW", cfg.out_dir, L"data" + std::to_wstring(index) + L"_phasex", cfg);
                }
                int roi_index = 1;
                for (const auto& file : fs::directory_iterator(entry.path())) {
                    if (file.is_regular_file() && file.path().filename().wstring().find(L"ROI") != std::wstring::npos &&
                        file.path().extension() == L".bmp") {
                        processOne(
                            file.path(),
                            {},
                            cfg.out_dir,
                            L"data" + std::to_wstring(index) + L"_roi" + std::to_wstring(roi_index),
                            cfg);
                        ++roi_index;
                    }
                }
                ++index;
            }
            return 0;
        }

        if (cfg.decode_raw_phase && cfg.raw_dir.empty()) {
            std::cerr << "error: raw_dir is required when decode_raw_phase=true\n";
            return 2;
        }
        if (!cfg.decode_raw_phase && cfg.phase_path.empty()) {
            std::cerr << "error: phase_path is required when decode_raw_phase=false\n";
            return 2;
        }
        processOne(cfg.phase_path, cfg.raw_dir, cfg.out_dir, cfg.prefix, cfg);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
