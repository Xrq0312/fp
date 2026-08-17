#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

struct PhaseROI {
    cv::Rect rect;
    int k_order;
};

constexpr double kPhaseFitThreshold = 0.6;
struct QualityEdge {
    uint32_t encoded = 0;
    float score = 0.0f;
};

struct UnionFind {
    std::vector<int> parent;
    std::vector<int> size;
    std::vector<int> period_to_parent;

    explicit UnionFind(int count) : parent(count), size(count, 1), period_to_parent(count, 0) {
        for (int i = 0; i < count; ++i) {
            parent[i] = i;
        }
    }

    int find(int value) {
        const int p = parent[value];
        if (p == value) {
            return value;
        }
        const int root = find(p);
        period_to_parent[value] += period_to_parent[p];
        parent[value] = root;
        return root;
    }

    int periodToRoot(int value) {
        find(value);
        return period_to_parent[value];
    }

    void unite(int p, int q, int q_minus_p_periods) {
        const int root_p = find(p);
        const int p_to_root = period_to_parent[p];
        const int root_q = find(q);
        const int q_to_root = period_to_parent[q];
        if (root_p == root_q) {
            return;
        }
        if (size[root_p] < size[root_q]) {
            parent[root_p] = root_q;
            period_to_parent[root_p] = q_to_root - p_to_root - q_minus_p_periods;
            size[root_q] += size[root_p];
        } else {
            parent[root_q] = root_p;
            period_to_parent[root_q] = q_minus_p_periods + p_to_root - q_to_root;
            size[root_p] += size[root_q];
        }
    }
};

double wrapPi(double value) {
    return std::atan2(std::sin(value), std::cos(value));
}

cv::Mat unwrapQualityGuidedROI(
    const cv::Mat& wrapped,
    const cv::Rect& requested_roi,
    cv::Mat1b& integration_mask) {
    CV_Assert(wrapped.type() == CV_64FC1);
    const cv::Rect roi = requested_roi & cv::Rect(0, 0, wrapped.cols, wrapped.rows);
    cv::Mat1f phase;
    wrapped(roi).convertTo(phase, CV_32FC1);
    cv::Mat1f reliability(phase.size(), 0.0f);
    constexpr float max_reliability = 16.0f * static_cast<float>(CV_PI * CV_PI);

    for (int y = 1; y < phase.rows - 1; ++y) {
        for (int x = 1; x < phase.cols - 1; ++x) {
            const float center = phase(y, x);
            const double h = wrapPi(phase(y, x - 1) - 2.0 * center + phase(y, x + 1));
            const double v = wrapPi(phase(y - 1, x) - 2.0 * center + phase(y + 1, x));
            const double d1 = wrapPi(phase(y - 1, x - 1) - 2.0 * center + phase(y + 1, x + 1));
            const double d2 = wrapPi(phase(y - 1, x + 1) - 2.0 * center + phase(y + 1, x - 1));
            const double error = h * h + v * v + d1 * d1 + d2 * d2;
            reliability(y, x) = static_cast<float>(std::max(0.0, max_reliability - error));
        }
    }

    cv::Mat1f cos_phase;
    cv::Mat1f sin_phase;
    cv::polarToCart(cv::Mat1f(phase.size(), 1.0f), phase, cos_phase, sin_phase);
    cv::boxFilter(cos_phase, cos_phase, CV_32F, cv::Size(9, 9), cv::Point(-1, -1), true, cv::BORDER_REPLICATE);
    cv::boxFilter(sin_phase, sin_phase, CV_32F, cv::Size(9, 9), cv::Point(-1, -1), true, cv::BORDER_REPLICATE);

    cv::Mat1f residue(phase.size(), 0.0f);
    for (int y = 0; y + 1 < phase.rows; ++y) {
        for (int x = 0; x + 1 < phase.cols; ++x) {
            const double closure =
                wrapPi(phase(y, x + 1) - phase(y, x)) +
                wrapPi(phase(y + 1, x + 1) - phase(y, x + 1)) +
                wrapPi(phase(y + 1, x) - phase(y + 1, x + 1)) +
                wrapPi(phase(y, x) - phase(y + 1, x));
            if (std::abs(closure) > CV_PI) {
                residue(y, x) = 1.0f;
            }
        }
    }
    cv::Mat1f residue_density;
    cv::boxFilter(residue, residue_density, CV_32F, cv::Size(7, 7), cv::Point(-1, -1), true, cv::BORDER_REPLICATE);

    cv::Mat1b valid(phase.size(), uchar(0));
    for (int y = 0; y < phase.rows; ++y) {
        const float* mean_cos = cos_phase.ptr<float>(y);
        const float* mean_sin = sin_phase.ptr<float>(y);
        const float* density = residue_density.ptr<float>(y);
        const float* rel = reliability.ptr<float>(y);
        uchar* mask = valid.ptr<uchar>(y);
        for (int x = 0; x < phase.cols; ++x) {
            const float coherence = std::sqrt(mean_cos[x] * mean_cos[x] + mean_sin[x] * mean_sin[x]);
            const float coherence_score = std::clamp((coherence - 0.45f) / 0.55f, 0.0f, 1.0f);
            const float residue_score = std::clamp(1.0f - density[x] / 0.10f, 0.0f, 1.0f);
            const float reliability_score = std::clamp(rel[x] / max_reliability, 0.0f, 1.0f);
            mask[x] = coherence_score * residue_score * reliability_score > 0.25f ? uchar(255) : uchar(0);
        }
    }

    const int width = phase.cols;
    const int height = phase.rows;
    const int pixel_count = width * height;
    std::vector<QualityEdge> edges;
    edges.reserve(static_cast<size_t>(cv::countNonZero(valid)) * 2);
    auto add_edge = [&](int index, bool vertical, float score) {
        edges.push_back({static_cast<uint32_t>(index * 2 + (vertical ? 1 : 0)), score});
    };
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const int index = y * width + x;
            if (valid(y, x) == 0) {
                continue;
            }
            if (x + 1 < width && valid(y, x + 1) != 0) {
                add_edge(index, false, 0.5f * (reliability(y, x) + reliability(y, x + 1)));
            }
            if (y + 1 < height && valid(y + 1, x) != 0) {
                add_edge(index, true, 0.5f * (reliability(y, x) + reliability(y + 1, x)));
            }
        }
    }

    std::sort(edges.begin(), edges.end(), [](const QualityEdge& left, const QualityEdge& right) {
        return left.score > right.score;
    });

    UnionFind groups(pixel_count);
    for (const QualityEdge& edge : edges) {
            const uint32_t encoded = edge.encoded;
            const bool vertical = (encoded & 1U) != 0;
            const int p_index = static_cast<int>(encoded >> 1U);
            const int q_index = p_index + (vertical ? width : 1);
            const int py = p_index / width;
            const int px = p_index - py * width;
            const int qy = q_index / width;
            const int qx = q_index - qy * width;
            const double p = phase(py, px);
            const double q = phase(qy, qx);
            const double delta = wrapPi(q - p);
            const int periods = static_cast<int>(std::lround((p + delta - q) / (2.0 * CV_PI)));
            groups.unite(p_index, q_index, periods);
    }

    std::unordered_map<int, std::pair<long long, long long>> component_alignment;
    component_alignment.reserve(65536);
    for (int y = 0; y < height; ++y) {
        const uchar* valid_row = valid.ptr<uchar>(y);
        for (int x = 0; x < width; ++x) {
            if (valid_row[x] == 0) {
                continue;
            }
            const int index = y * width + x;
            const int root = groups.find(index);
            const int periods = groups.periodToRoot(index);
            auto& stats = component_alignment[root];
            stats.first -= periods;
            ++stats.second;
        }
    }

    cv::Mat continuous = wrapped.clone();
    for (int y = 0; y < height; ++y) {
        const float* source = phase.ptr<float>(y);
        const uchar* valid_row = valid.ptr<uchar>(y);
        double* target = continuous.ptr<double>(y + roi.y) + roi.x;
        for (int x = 0; x < width; ++x) {
            if (valid_row[x] == 0) {
                continue;
            }
            const int index = y * width + x;
            const int root = groups.find(index);
            const int periods = groups.periodToRoot(index);
            const auto stats = component_alignment.at(root);
            const int alignment = static_cast<int>(std::lround(
                static_cast<double>(stats.first) / static_cast<double>(stats.second)));
            target[x] = source[x] + (periods + alignment) * 2.0 * CV_PI;
        }
    }


    long long simple_jumps = 0;
    long long continuous_jumps = 0;
    for (int y = 0; y < height; ++y) {
        const uchar* valid_row = valid.ptr<uchar>(y);
        const double* simple_row = wrapped.ptr<double>(y + roi.y) + roi.x;
        const double* continuous_row = continuous.ptr<double>(y + roi.y) + roi.x;
        for (int x = 0; x < width; ++x) {
            if (valid_row[x] == 0) {
                continue;
            }
            if (x + 1 < width && valid_row[x + 1] != 0) {
                simple_jumps += std::abs(simple_row[x + 1] - simple_row[x]) > CV_PI;
                continuous_jumps += std::abs(continuous_row[x + 1] - continuous_row[x]) > CV_PI;
            }
            if (y + 1 < height && valid(y + 1, x) != 0) {
                simple_jumps += std::abs(wrapped.at<double>(y + roi.y + 1, x + roi.x) - simple_row[x]) > CV_PI;
                continuous_jumps += std::abs(continuous.at<double>(y + roi.y + 1, x + roi.x) - continuous_row[x]) > CV_PI;
            }
        }
    }
    if (continuous_jumps >= simple_jumps) {
        wrapped.copyTo(continuous);
        continuous_jumps = simple_jumps;
    }

    integration_mask = cv::Mat1b(wrapped.size(), uchar(0));
    valid.copyTo(integration_mask(roi));
    std::cout << "quality unwrap ROI=" << roi.x << "," << roi.y << ","
              << roi.width << "," << roi.height
              << ", valid=" << cv::countNonZero(valid)
              << ", components=" << component_alignment.size()
              << ", sorted edges=" << edges.size()
              << ", valid jumps=" << simple_jumps << " -> " << continuous_jumps << "\n";
    return continuous;
}

std::vector<PhaseROI> roisForAxis(const std::string& axis, int set_index) {
    if (axis == "x" || axis == "X") {
        if (set_index == 1) {
            return {
                {cv::Rect(920, 2420, 160, 160), -2},
                {cv::Rect(2120, 2420, 160, 160), -1},
                {cv::Rect(3320, 2420, 160, 160), 0},
                {cv::Rect(4520, 2420, 160, 160), 1},
                {cv::Rect(5720, 2420, 160, 160), 2},
                {cv::Rect(6920, 2420, 160, 160), 3},
            };
        }
        return {
            {cv::Rect(1120, 2420, 160, 160), -2},
            {cv::Rect(2120, 2420, 160, 160), -1},
            {cv::Rect(3120, 2420, 160, 160), 0},
            {cv::Rect(4120, 2420, 160, 160), 1},
            {cv::Rect(5520, 2420, 160, 160), 2},
            {cv::Rect(6720, 2420, 160, 160), 3},
        };
    }
    if (axis == "y" || axis == "Y") {
        if (set_index == 1) {
            return {
                {cv::Rect(3920, 420, 160, 160), -1},
                {cv::Rect(3920, 1420, 160, 160), 0},
                {cv::Rect(3920, 2620, 160, 160), 1},
                {cv::Rect(3920, 3820, 160, 160), 2},
            };
        }
        return {
            {cv::Rect(3920, 920, 160, 160), 0},
            {cv::Rect(3920, 1520, 160, 160), 1},
            {cv::Rect(3920, 2720, 160, 160), 2},
            {cv::Rect(3920, 3920, 160, 160), 3},
        };
    }
    throw std::runtime_error("axis must be x or y");
}

bool fitCarrierPlane(
    const cv::Mat& phase64,
    const std::vector<PhaseROI>& rois,
    cv::Vec3d& coefficients,
    int& sample_count) {
    const int width = phase64.cols;
    const int height = phase64.rows;
    const double cx = width / 2.0;
    const double cy = height / 2.0;
    const double inv_scale = 1.0 / static_cast<double>(std::max(width, height));
    int capacity = 0;
    for (const auto& roi : rois) {
        capacity += (roi.rect & cv::Rect(0, 0, width, height)).area();
    }
    cv::Mat design(capacity, 3, CV_64FC1);
    cv::Mat target(capacity, 1, CV_64FC1);
    sample_count = 0;
    for (const auto& roi : rois) {
        const cv::Rect clipped = roi.rect & cv::Rect(0, 0, width, height);
        for (int y = clipped.y; y < clipped.y + clipped.height; ++y) {
            const double yn = (static_cast<double>(y) - cy) * inv_scale;
            const double* phase_row = phase64.ptr<double>(y);
            for (int x = clipped.x; x < clipped.x + clipped.width; ++x) {
                const double wrapped = phase_row[x];
                if (std::abs(wrapped) >= kPhaseFitThreshold * CV_PI) {
                    continue;
                }
                design.at<double>(sample_count, 0) = (static_cast<double>(x) - cx) * inv_scale;
                design.at<double>(sample_count, 1) = yn;
                design.at<double>(sample_count, 2) = 1.0;
                target.at<double>(sample_count, 0) = wrapped + roi.k_order * 2.0 * CV_PI;
                ++sample_count;
            }
        }
    }
    if (sample_count < 3) {
        return false;
    }
    cv::Mat solved;
    if (!cv::solve(
            design.rowRange(0, sample_count),
            target.rowRange(0, sample_count),
            solved,
            cv::DECOMP_SVD)) {
        return false;
    }
    coefficients = cv::Vec3d(
        solved.at<double>(0, 0),
        solved.at<double>(1, 0),
        solved.at<double>(2, 0));
    return true;
}

struct PeriodicCalibration {
    std::array<double, 8> harmonic{};
    double peak_to_peak = 0.0;
    double minimum_derivative = 1.0;
    bool applied = false;
};

double periodicError(const std::array<double, 8>& harmonic, double phase) {
    double value = 0.0;
    for (int order = 1; order <= 4; ++order) {
        value += harmonic[static_cast<size_t>((order - 1) * 2)] * std::sin(order * phase);
        value += harmonic[static_cast<size_t>((order - 1) * 2 + 1)] * std::cos(order * phase);
    }
    return value;
}

PeriodicCalibration fitPeriodicCalibration(const cv::Mat& phase64, const cv::Vec3d& plane) {
    constexpr int feature_count = 14;
    constexpr int sample_step = 4;
    constexpr double huber_delta = 0.08;
    const int width = phase64.cols;
    const int height = phase64.rows;
    const double cx = width / 2.0;
    const double cy = height / 2.0;
    const double inv_scale = 1.0 / static_cast<double>(std::max(width, height));
    const cv::Rect calibration_roi = cv::Rect(1000, 1200, 5801, 2701) &
        cv::Rect(0, 0, width, height);
    std::array<double, feature_count> coefficients{};

    for (int iteration = 0; iteration < 5; ++iteration) {
        cv::Mat1d lhs(feature_count, feature_count, 0.0);
        cv::Mat1d rhs(feature_count, 1, 0.0);
        for (int y = calibration_roi.y; y < calibration_roi.y + calibration_roi.height; y += sample_step) {
            const double yn = (static_cast<double>(y) - cy) * inv_scale;
            const double* phase_row = phase64.ptr<double>(y);
            for (int x = calibration_roi.x; x < calibration_roi.x + calibration_roi.width; x += sample_step) {
                const double xn = (static_cast<double>(x) - cx) * inv_scale;
                const double phi = phase_row[x];
                const double carrier = plane[0] * xn + plane[1] * yn + plane[2];
                const double response = wrapPi(phi - carrier);
                if (std::abs(response) >= 2.8) {
                    continue;
                }
                std::array<double, feature_count> feature{
                    1.0, xn, yn, xn * xn, xn * yn, yn * yn,
                    std::sin(phi), std::cos(phi),
                    std::sin(2.0 * phi), std::cos(2.0 * phi),
                    std::sin(3.0 * phi), std::cos(3.0 * phi),
                    std::sin(4.0 * phi), std::cos(4.0 * phi)};
                double prediction = 0.0;
                for (int i = 0; i < feature_count; ++i) {
                    prediction += coefficients[static_cast<size_t>(i)] * feature[static_cast<size_t>(i)];
                }
                const double error = response - prediction;
                const double weight = iteration == 0
                    ? 1.0
                    : std::min(1.0, huber_delta / std::max(std::abs(error), 1e-12));
                for (int i = 0; i < feature_count; ++i) {
                    rhs(i, 0) += weight * feature[static_cast<size_t>(i)] * response;
                    for (int j = 0; j < feature_count; ++j) {
                        lhs(i, j) += weight * feature[static_cast<size_t>(i)] * feature[static_cast<size_t>(j)];
                    }
                }
            }
        }
        cv::Mat1d solved;
        if (!cv::solve(lhs, rhs, solved, cv::DECOMP_SVD)) {
            return {};
        }
        for (int i = 0; i < feature_count; ++i) {
            coefficients[static_cast<size_t>(i)] = solved(i, 0);
        }
    }

    PeriodicCalibration calibration;
    for (int i = 0; i < 8; ++i) {
        calibration.harmonic[static_cast<size_t>(i)] = coefficients[static_cast<size_t>(i + 6)];
    }
    double minimum = std::numeric_limits<double>::infinity();
    double maximum = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < 4096; ++i) {
        const double phase = -CV_PI + 2.0 * CV_PI * i / 4096.0;
        const double error = periodicError(calibration.harmonic, phase);
        minimum = std::min(minimum, error);
        maximum = std::max(maximum, error);
        double error_derivative = 0.0;
        for (int order = 1; order <= 4; ++order) {
            const double sine = calibration.harmonic[static_cast<size_t>((order - 1) * 2)];
            const double cosine = calibration.harmonic[static_cast<size_t>((order - 1) * 2 + 1)];
            error_derivative += order * sine * std::cos(order * phase);
            error_derivative -= order * cosine * std::sin(order * phase);
        }
        calibration.minimum_derivative = std::min(calibration.minimum_derivative, 1.0 - error_derivative);
    }
    calibration.peak_to_peak = maximum - minimum;
    calibration.applied = calibration.peak_to_peak >= 0.05 && calibration.minimum_derivative >= 0.25;
    return calibration;
}

bool unwrapPhaseWithROIPlane(
    const cv::Mat& phase_input,
    const std::vector<PhaseROI>& rois,
    cv::Mat& unwrapped,
    cv::Mat& residual,
    cv::Mat& wrapped_residual,
    cv::Mat& uncalibrated_residual,
    cv::Mat1b& integration_mask,
    cv::Mat& corrected_phase,
    cv::Mat1f& periodic_correction_lut,
    cv::Mat& k_order_image,
    cv::Vec3d& plane_coeffs) {

    if (phase_input.empty() || rois.empty()) {
        return false;
    }

    cv::Mat phase64;
    if (phase_input.type() == CV_8UC1) {
        phase_input.convertTo(phase64, CV_64FC1, 2.0 * CV_PI / 255.0, -CV_PI);
    } else {
        phase_input.convertTo(phase64, CV_64FC1);
    }

    const int width = phase64.cols;
    const int height = phase64.rows;
    const double cx = width / 2.0;
    const double cy = height / 2.0;
    const double inv_scale = 1.0 / static_cast<double>(std::max(width, height));

    int row_index = 0;
    if (!fitCarrierPlane(phase64, rois, plane_coeffs, row_index)) {
        return false;
    }

    uncalibrated_residual.create(height, width, CV_64FC1);
    cv::parallel_for_(cv::Range(0, height), [&](const cv::Range& range) {
        for (int y = range.start; y < range.end; ++y) {
            const double yn = (static_cast<double>(y) - cy) * inv_scale;
            const double* phase_row = phase64.ptr<double>(y);
            double* output_row = uncalibrated_residual.ptr<double>(y);
            for (int x = 0; x < width; ++x) {
                const double xn = (static_cast<double>(x) - cx) * inv_scale;
                const double carrier = plane_coeffs[0] * xn + plane_coeffs[1] * yn + plane_coeffs[2];
                output_row[x] = wrapPi(phase_row[x] - carrier);
            }
        }
    });

    // Keep the measured phase unchanged.  A periodic response curve cannot be
    // identified from this object measurement alone: the fitted curve also
    // absorbs real, spatially non-linear phase.  Applying it once per 2*pi
    // period creates carrier-aligned bands in the residual image.
    periodic_correction_lut = cv::Mat1f(1, 256, 0.0f);
    corrected_phase = phase64.clone();
    std::cout << "periodic calibration: disabled; original measured phase preserved\n";

    unwrapped.create(height, width, CV_64FC1);
    residual.create(height, width, CV_64FC1);
    k_order_image.create(height, width, CV_32SC1);
    cv::parallel_for_(cv::Range(0, height), [&](const cv::Range& range) {
        for (int y = range.start; y < range.end; ++y) {
            const double yn = (static_cast<double>(y) - cy) * inv_scale;
            const double* phase_row = phase64.ptr<double>(y);
            double* unwrap_row = unwrapped.ptr<double>(y);
            double* residual_row = residual.ptr<double>(y);
            int* k_row = k_order_image.ptr<int>(y);
            for (int x = 0; x < width; ++x) {
                const double xn = (static_cast<double>(x) - cx) * inv_scale;
                const double plane = plane_coeffs[0] * xn + plane_coeffs[1] * yn + plane_coeffs[2];
                const double wrapped = phase_row[x];
                const int k = static_cast<int>(std::floor((plane - wrapped) / (2.0 * CV_PI) + 0.5));
                const double absolute = wrapped + k * 2.0 * CV_PI;
                k_row[x] = k;
                unwrap_row[x] = absolute;
                residual_row[x] = absolute - plane;
            }
        }
    });

    // The plane-guided result above is the original simple method: it keeps
    // every measured detail, but wraps the plane residual back to [-pi, pi].
    // Unwrap that residual by changing integer 2*pi orders only.  No mask,
    // filtering, interpolation, or smoothing is applied.
    wrapped_residual = residual.clone();
    const cv::Rect integration_roi(600, 700, 7001, 4001);
    cv::Mat continuous_residual = unwrapQualityGuidedROI(wrapped_residual, integration_roi, integration_mask);

    cv::parallel_for_(cv::Range(0, height), [&](const cv::Range& range) {
        for (int y = range.start; y < range.end; ++y) {
            const double yn = (static_cast<double>(y) - cy) * inv_scale;
            const double* phase_row = phase64.ptr<double>(y);
            const double* continuous_row = continuous_residual.ptr<double>(y);
            double* unwrap_row = unwrapped.ptr<double>(y);
            double* residual_row = residual.ptr<double>(y);
            int* k_row = k_order_image.ptr<int>(y);
            for (int x = 0; x < width; ++x) {
                const double xn = (static_cast<double>(x) - cx) * inv_scale;
                const double plane = plane_coeffs[0] * xn + plane_coeffs[1] * yn + plane_coeffs[2];
                const bool integrated = integration_mask(y, x) != 0;
                const double continuous = integrated
                    ? continuous_row[x]
                    : wrapped_residual.at<double>(y, x);
                const double absolute = plane + continuous;
                residual_row[x] = continuous;
                unwrap_row[x] = absolute;
                k_row[x] = static_cast<int>(std::lround((absolute - phase_row[x]) / (2.0 * CV_PI)));
            }
        }
    });

    std::cout << "fit samples=" << row_index << "\n"
              << "Phi(xn,yn)=" << plane_coeffs[0] << "*xn + "
              << plane_coeffs[1] << "*yn + " << plane_coeffs[2] << "\n"
              << "continuous residual: quality-guided integer K inside ROI, component-aligned to plane\n";
    return true;
}

void saveFloatTiff(const fs::path& path, const cv::Mat& source) {
    cv::Mat out;
    source.convertTo(out, CV_32FC1);
    if (!cv::imwrite(path.string(), out)) {
        throw std::runtime_error("failed to write " + path.string());
    }
}

void saveNormalized(const fs::path& path, const cv::Mat& source) {
    cv::Mat out;
    cv::normalize(source, out, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    if (!cv::imwrite(path.string(), out)) {
        throw std::runtime_error("failed to write " + path.string());
    }
}

void saveQuantizationAwarePreview(
    const fs::path& path,
    const cv::Mat& source,
    double gain,
    double bias) {
    CV_Assert(source.type() == CV_64FC1);
    cv::Mat1b preview(source.size());
    const double amplified_input_quantum = std::abs(gain) * (2.0 * CV_PI / 255.0);
    cv::parallel_for_(cv::Range(0, source.rows), [&](const cv::Range& range) {
        for (int y = range.start; y < range.end; ++y) {
            const double* source_row = source.ptr<double>(y);
            uchar* preview_row = preview.ptr<uchar>(y);
            for (int x = 0; x < source.cols; ++x) {
                // The source phase is an 8-bit BMP.  At gain=-200 one source
                // code becomes a 4.93-level display step, which otherwise
                // forms false contour lines.  Deterministic zero-mean dither
                // represents that unavoidable quantization uncertainty in the
                // preview only; the float TIFF remains untouched.
                uint32_t hash = static_cast<uint32_t>(x) * 0x9e3779b9U ^
                    static_cast<uint32_t>(y) * 0x85ebca6bU;
                hash ^= hash >> 16U;
                hash *= 0x7feb352dU;
                hash ^= hash >> 15U;
                const double unit = static_cast<double>(hash & 0xffffU) / 65535.0 - 0.5;
                const double value = source_row[x] * gain + bias + unit * amplified_input_quantum;
                preview_row[x] = cv::saturate_cast<uchar>(value);
            }
        }
    });
    if (!cv::imwrite(path.string(), preview)) {
        throw std::runtime_error("failed to write " + path.string());
    }
}

void savePeriodicCorrectionCurve(const fs::path& path, const cv::Mat1f& lut) {
    constexpr int width = 768;
    constexpr int height = 320;
    constexpr int margin = 36;
    cv::Mat3b canvas(height, width, cv::Vec3b(255, 255, 255));
    double max_abs = 0.0;
    for (int code = 0; code < lut.cols; ++code) {
        max_abs = std::max(max_abs, std::abs(static_cast<double>(lut(0, code))));
    }
    max_abs = std::max(max_abs, 0.01);
    const int center_y = height / 2;
    cv::line(canvas, cv::Point(margin, center_y), cv::Point(width - margin, center_y), cv::Scalar(180, 180, 180), 1);
    cv::line(canvas, cv::Point(margin, margin), cv::Point(margin, height - margin), cv::Scalar(180, 180, 180), 1);
    std::vector<cv::Point> curve;
    curve.reserve(static_cast<size_t>(lut.cols));
    for (int code = 0; code < lut.cols; ++code) {
        const int px = margin + code * (width - 2 * margin) / std::max(1, lut.cols - 1);
        const int py = center_y - static_cast<int>(std::lround(lut(0, code) * (height / 2 - margin) / max_abs));
        curve.emplace_back(px, py);
    }
    cv::polylines(canvas, curve, false, cv::Scalar(220, 70, 20), 2, cv::LINE_AA);
    cv::putText(canvas, "phase code 0..255", cv::Point(width / 2 - 90, height - 8),
                cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(60, 60, 60), 1, cv::LINE_AA);
    cv::putText(canvas, "periodic correction (rad)", cv::Point(44, 24),
                cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(60, 60, 60), 1, cv::LINE_AA);
    if (!cv::imwrite(path.string(), canvas)) {
        throw std::runtime_error("failed to write " + path.string());
    }
}

int main(int argc, char** argv) {
    try {
        if (argc != 4 && argc != 5) {
            std::cerr << "Usage: manual_roi_plane <phase.bmp> <output_dir> <x|y> [set1|set2]\n";
            return 2;
        }
        const fs::path input = argv[1];
        const fs::path output = argv[2];
        const std::string axis = argv[3];
        int set_index = 2;
        if (argc == 5) {
            const std::string set_arg = argv[4];
            if (set_arg == "set1" || set_arg == "1") {
                set_index = 1;
            } else if (set_arg != "set2" && set_arg != "2") {
                throw std::runtime_error("set must be set1 or set2");
            }
        }
        const auto rois = roisForAxis(axis, set_index);

        cv::Mat phase = cv::imread(input.string(), cv::IMREAD_GRAYSCALE);
        if (phase.empty()) {
            throw std::runtime_error("failed to read " + input.string());
        }
        fs::create_directories(output);

        cv::Mat unwrapped;
        cv::Mat residual;
        cv::Mat wrapped_residual;
        cv::Mat uncalibrated_residual;
        cv::Mat1b integration_mask;
        cv::Mat corrected_phase;
        cv::Mat1f periodic_correction_lut;
        cv::Mat k_orders;
        cv::Vec3d coeffs;
        if (!unwrapPhaseWithROIPlane(
                phase,
                rois,
                unwrapped,
                residual,
                wrapped_residual,
                uncalibrated_residual,
                integration_mask,
                corrected_phase,
                periodic_correction_lut,
                k_orders,
                coeffs)) {
            throw std::runtime_error("ROI plane fitting failed");
        }

        saveFloatTiff(output / "unwrappedPhase.tiff", unwrapped);
        saveFloatTiff(output / "residualPhase.tiff", residual);
        saveFloatTiff(output / "residualPhase_simple_wrapped.tiff", wrapped_residual);
        saveFloatTiff(output / "residualPhase_uncalibrated.tiff", uncalibrated_residual);
        saveFloatTiff(output / "phase_corrected.tiff", corrected_phase);
        saveFloatTiff(output / "periodic_correction_lut.tiff", periodic_correction_lut);
        saveFloatTiff(output / "kOrder.tiff", k_orders);
        saveNormalized(output / "wrapped.png", phase);
        saveNormalized(output / "unwrappedPhase.png", unwrapped);
        saveNormalized(output / "residualPhase.png", residual);
        saveNormalized(output / "residualPhase_simple_wrapped.png", wrapped_residual);
        saveNormalized(output / "phase_corrected.png", corrected_phase);
        saveNormalized(output / "periodic_correction_lut.png", periodic_correction_lut);
        savePeriodicCorrectionCurve(output / "periodic_correction_curve.png", periodic_correction_lut);
        cv::imwrite((output / "integration_mask.png").string(), integration_mask);

        cv::Mat residual_kb;
        residual.convertTo(residual_kb, CV_8UC1, -200.0, 128.0);
        cv::imwrite((output / "residualPhase_k-200_b128_quantized.png").string(), residual_kb);
        saveQuantizationAwarePreview(output / "residualPhase_k-200_b128.png", residual, -200.0, 128.0);

        cv::Mat simple_residual_kb;
        wrapped_residual.convertTo(simple_residual_kb, CV_8UC1, -200.0, 128.0);
        cv::imwrite((output / "residualPhase_simple_detail_k-200_b128_quantized.png").string(), simple_residual_kb);
        saveQuantizationAwarePreview(
            output / "residualPhase_simple_detail_k-200_b128.png",
            wrapped_residual,
            -200.0,
            128.0);

        cv::Mat uncalibrated_kb;
        uncalibrated_residual.convertTo(uncalibrated_kb, CV_8UC1, -200.0, 128.0);
        cv::imwrite((output / "residualPhase_uncalibrated_k-200_b128_quantized.png").string(), uncalibrated_kb);
        saveQuantizationAwarePreview(
            output / "residualPhase_uncalibrated_k-200_b128.png",
            uncalibrated_residual,
            -200.0,
            128.0);

        cv::Mat roi_visual;
        cv::cvtColor(phase, roi_visual, cv::COLOR_GRAY2BGR);
        for (const auto& roi : rois) {
            cv::rectangle(roi_visual, roi.rect, cv::Scalar(0, 0, 255), 3);
            cv::putText(
                roi_visual,
                "K=" + std::to_string(roi.k_order),
                roi.rect.tl() - cv::Point(0, 8),
                cv::FONT_HERSHEY_SIMPLEX,
                0.8,
                cv::Scalar(0, 255, 0),
                2);
        }
        cv::imwrite((output / "roi_anchors.png").string(), roi_visual);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << "\n";
        return 1;
    }
}
