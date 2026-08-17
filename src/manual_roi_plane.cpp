#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
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

bool unwrapPhaseWithROIPlane(
    const cv::Mat& phase_input,
    const std::vector<PhaseROI>& rois,
    cv::Mat& unwrapped,
    cv::Mat& residual,
    cv::Mat& wrapped_residual,
    cv::Mat1b& integration_mask,
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

    int total_points = 0;
    for (const auto& roi : rois) {
        total_points += (roi.rect & cv::Rect(0, 0, width, height)).area();
    }
    cv::Mat design(total_points, 3, CV_64FC1);
    cv::Mat target(total_points, 1, CV_64FC1);

    int row_index = 0;
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
                design.at<double>(row_index, 0) = (static_cast<double>(x) - cx) * inv_scale;
                design.at<double>(row_index, 1) = yn;
                design.at<double>(row_index, 2) = 1.0;
                target.at<double>(row_index, 0) = wrapped + roi.k_order * 2.0 * CV_PI;
                ++row_index;
            }
        }
    }
    if (row_index < 3) {
        return false;
    }
    design = design.rowRange(0, row_index);
    target = target.rowRange(0, row_index);

    cv::Mat coeffs;
    if (!cv::solve(design, target, coeffs, cv::DECOMP_SVD)) {
        return false;
    }
    plane_coeffs = cv::Vec3d(
        coeffs.at<double>(0, 0),
        coeffs.at<double>(1, 0),
        coeffs.at<double>(2, 0));

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
        cv::Mat1b integration_mask;
        cv::Mat k_orders;
        cv::Vec3d coeffs;
        if (!unwrapPhaseWithROIPlane(
                phase, rois, unwrapped, residual, wrapped_residual, integration_mask, k_orders, coeffs)) {
            throw std::runtime_error("ROI plane fitting failed");
        }

        saveFloatTiff(output / "unwrappedPhase.tiff", unwrapped);
        saveFloatTiff(output / "residualPhase.tiff", residual);
        saveFloatTiff(output / "residualPhase_simple_wrapped.tiff", wrapped_residual);
        saveFloatTiff(output / "kOrder.tiff", k_orders);
        saveNormalized(output / "wrapped.png", phase);
        saveNormalized(output / "unwrappedPhase.png", unwrapped);
        saveNormalized(output / "residualPhase.png", residual);
        saveNormalized(output / "residualPhase_simple_wrapped.png", wrapped_residual);
        cv::imwrite((output / "integration_mask.png").string(), integration_mask);

        cv::Mat residual_kb;
        residual.convertTo(residual_kb, CV_8UC1, -200.0, 128.0);
        cv::imwrite((output / "residualPhase_k-200_b128.png").string(), residual_kb);

        cv::Mat simple_residual_kb;
        wrapped_residual.convertTo(simple_residual_kb, CV_8UC1, -200.0, 128.0);
        cv::imwrite((output / "residualPhase_simple_detail_k-200_b128.png").string(), simple_residual_kb);

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
