#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

struct PhaseROI {
    cv::Rect rect;
    int k_order;
};

constexpr double kPhaseFitThreshold = 0.6;

std::vector<PhaseROI> roisForAxis(const std::string& axis) {
    if (axis == "x" || axis == "X") {
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

    std::cout << "fit samples=" << row_index << "\n"
              << "Phi(xn,yn)=" << plane_coeffs[0] << "*xn + "
              << plane_coeffs[1] << "*yn + " << plane_coeffs[2] << "\n";
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
        if (argc != 4) {
            std::cerr << "Usage: manual_roi_plane <phase.bmp> <output_dir> <x|y>\n";
            return 2;
        }
        const fs::path input = argv[1];
        const fs::path output = argv[2];
        const std::string axis = argv[3];
        const auto rois = roisForAxis(axis);

        cv::Mat phase = cv::imread(input.string(), cv::IMREAD_GRAYSCALE);
        if (phase.empty()) {
            throw std::runtime_error("failed to read " + input.string());
        }
        fs::create_directories(output);

        cv::Mat unwrapped;
        cv::Mat residual;
        cv::Mat k_orders;
        cv::Vec3d coeffs;
        if (!unwrapPhaseWithROIPlane(phase, rois, unwrapped, residual, k_orders, coeffs)) {
            throw std::runtime_error("ROI plane fitting failed");
        }

        saveFloatTiff(output / "unwrappedPhase.tiff", unwrapped);
        saveFloatTiff(output / "residualPhase.tiff", residual);
        saveFloatTiff(output / "kOrder.tiff", k_orders);
        saveNormalized(output / "wrapped.png", phase);
        saveNormalized(output / "unwrappedPhase.png", unwrapped);
        saveNormalized(output / "residualPhase.png", residual);

        cv::Mat residual_kb;
        residual.convertTo(residual_kb, CV_8UC1, -200.0, 128.0);
        cv::imwrite((output / "residualPhase_k-200_b128.png").string(), residual_kb);

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
