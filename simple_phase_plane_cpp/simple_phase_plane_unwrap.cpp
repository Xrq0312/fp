#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

constexpr double kTwoPi = 2.0 * CV_PI;
// 拟合时避开包裹相位接近 +/-pi 的跳变区域。
constexpr double kFitPhaseLimit = 0.6 * CV_PI;
// RAW 输入用调制度排除几乎没有条纹信号的像素。
constexpr float kModulationFloor = 4.0f;

// wrapped 始终使用弧度，范围为 [-pi, pi]。
// 单张相位图没有独立的调制度信息，因此 loadInput() 会将其视为全有效。
struct InputPhase {
    cv::Mat1d wrapped;
    cv::Mat1f modulation;
};

struct FitResult {
    // 最终趋势：a*xn + b*yn + c，坐标定义见 unwrapWithAutomaticROI()。
    cv::Vec3d plane;
    // 两张全尺寸结果在 ROI 外仍保存原始包裹相位。
    cv::Mat1d unwrapped;
    cv::Mat1d residual;
    long long sample_count = 0;
    double guide_dx = 0.0;
    double guide_dy = 0.0;
};

cv::Mat readImage(const fs::path& path, int flags) {
    // 使用 imdecode 而不是 imread，保证 Windows 下中文路径可用。
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("cannot open input image");
    }
    std::vector<uchar> bytes(
        (std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
    cv::Mat image = cv::imdecode(bytes, flags);
    if (image.empty()) {
        throw std::runtime_error("cannot decode input image");
    }
    return image;
}

void writeImage(const fs::path& path, const cv::Mat& image) {
    // 与 readImage 对称，通过内存编码支持中文输出路径。
    std::vector<uchar> bytes;
    if (!cv::imencode(path.extension().string(), image, bytes)) {
        throw std::runtime_error("cannot encode output image");
    }
    std::ofstream stream(path, std::ios::binary);
    stream.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
    if (!stream) {
        throw std::runtime_error("cannot write output image");
    }
}

fs::path findRawFrame(const fs::path& directory, int index) {
    for (const wchar_t* extension : {L".jpg", L".png", L".bmp", L".tif", L".tiff"}) {
        const fs::path candidate = directory / (std::to_wstring(index) + extension);
        if (fs::exists(candidate)) {
            return candidate;
        }
    }
    throw std::runtime_error("RAW directory must contain frames 1..4");
}

InputPhase loadInput(const fs::path& input) {
    // 文件输入：认为它是 0..255 编码的包裹相位，线性映射到 [-pi, pi]。
    if (!fs::is_directory(input)) {
        cv::Mat gray = readImage(input, cv::IMREAD_GRAYSCALE);
        InputPhase result;
        gray.convertTo(result.wrapped, CV_64F, kTwoPi / 255.0, -CV_PI);
        result.modulation = cv::Mat1f(gray.size(), 255.0f);
        return result;
    }

    // 目录输入：自动寻找 1..4 四张相移原图。
    std::array<cv::Mat1f, 4> frame;
    for (int i = 0; i < 4; ++i) {
        cv::Mat gray = readImage(findRawFrame(input, i + 1), cv::IMREAD_GRAYSCALE);
        gray.convertTo(frame[static_cast<size_t>(i)], CV_32F);
        if (i > 0 && frame[static_cast<size_t>(i)].size() != frame[0].size()) {
            throw std::runtime_error("RAW frames have different sizes");
        }
    }

    InputPhase result;
    result.wrapped.create(frame[0].size());
    result.modulation.create(frame[0].size());
    for (int y = 0; y < frame[0].rows; ++y) {
        const float* i1 = frame[0].ptr<float>(y);
        const float* i2 = frame[1].ptr<float>(y);
        const float* i3 = frame[2].ptr<float>(y);
        const float* i4 = frame[3].ptr<float>(y);
        double* phase = result.wrapped.ptr<double>(y);
        float* modulation = result.modulation.ptr<float>(y);
        for (int x = 0; x < frame[0].cols; ++x) {
            // 四步相移正交分量。相位约定与当前 data2 数据保持一致。
            const double c = 0.5 * (i1[x] - i3[x]);
            const double s = 0.5 * (i2[x] - i4[x]);
            // atan2(s,c) 取反并加 pi/2，再显式包裹回 [-pi, pi]。
            phase[x] = std::atan2(std::sin(-std::atan2(s, c) + CV_PI / 2.0),
                                  std::cos(-std::atan2(s, c) + CV_PI / 2.0));
            modulation[x] = static_cast<float>(0.5 * std::sqrt(c * c + s * s)); // 条纹调制度。
        }
    }
    return result;
}

double median(std::vector<double>& values) {
    // 载频估计使用中位数，降低物体边缘和局部缺陷的影响。
    if (values.empty()) {
        throw std::runtime_error("not enough valid phase differences in ROI");
    }
    const size_t middle = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + middle, values.end());
    return values[middle];
}

double wrapPi(double value) {
    // atan2(sin,cos) 对正负输入都能稳定地映射到 [-pi, pi]。
    return std::atan2(std::sin(value), std::cos(value));
}

FitResult unwrapWithAutomaticROI(const InputPhase& input, const cv::Rect& requested_roi) {
    // ROI 使用 OpenCV 半开区间语义：[x0,x1) x [y0,y1)。超出图像的部分会被裁掉。
    const cv::Rect image_rect(0, 0, input.wrapped.cols, input.wrapped.rows);
    const cv::Rect roi = requested_roi & image_rect;
    if (roi.width < 2 || roi.height < 2) {
        throw std::runtime_error("ROI is empty after clipping to image");
    }

    // 第 1 步： ϕ = guide_dx*x+guide_dy*y+c
    // 估计一个仅用于自动赋 K 的粗略载频斜率。
    // 8 位相位图相邻像素经常具有相同灰度，所以跨 32 像素求包裹相位差，
    // 然后除以基线长度，分别得到每像素的 x/y 相位增量。
    constexpr int slope_baseline = 32;
    std::vector<double> differences;
    differences.reserve(static_cast<size_t>(roi.area()));
    for (int y = roi.y; y < roi.y + roi.height; ++y) {
        const double* phase = input.wrapped.ptr<double>(y);
        const float* modulation = input.modulation.ptr<float>(y);
        for (int x = roi.x; x < roi.x + roi.width - slope_baseline; ++x) {
            if (modulation[x] >= kModulationFloor &&
                modulation[x + slope_baseline] >= kModulationFloor) {
                differences.push_back(
                    wrapPi(phase[x + slope_baseline] - phase[x]) / slope_baseline);
            }
        }
    }
    const double guide_dx = median(differences);
    differences.clear();
    for (int y = roi.y; y < roi.y + roi.height - slope_baseline; ++y) {
        const double* phase0 = input.wrapped.ptr<double>(y);
        const double* phase1 = input.wrapped.ptr<double>(y + slope_baseline);
        const float* mod0 = input.modulation.ptr<float>(y);
        const float* mod1 = input.modulation.ptr<float>(y + slope_baseline);
        for (int x = roi.x; x < roi.x + roi.width; ++x) {
            if (mod0[x] >= kModulationFloor && mod1[x] >= kModulationFloor) {
                differences.push_back(wrapPi(phase1[x] - phase0[x]) / slope_baseline);
            }
        }
    }
    const double guide_dy = median(differences);

    // 第 2 步：在圆周上估计粗略平面的常数项 c 。
    // 不能直接平均 reduced，因为它仍是包裹量 c_=c+2pik；平均单位复数后再取角度去掉周期2pik。
    double sum_sin = 0.0;
    double sum_cos = 0.0;
    long long guide_count = 0;
    for (int y = roi.y; y < roi.y + roi.height; ++y) {
        const double* phase = input.wrapped.ptr<double>(y);
        const float* modulation = input.modulation.ptr<float>(y);
        for (int x = roi.x; x < roi.x + roi.width; ++x) {
            if (modulation[x] < kModulationFloor) {
                continue;
            }
            const double reduced = phase[x] - guide_dx * x - guide_dy * y;
            sum_sin += std::sin(reduced);
            sum_cos += std::cos(reduced);
            ++guide_count;
        }
    }
    if (guide_count < 3) {
        throw std::runtime_error("not enough valid pixels in ROI");
    }
    const double guide_c = std::atan2(sum_sin, sum_cos);

    // 第 3 步：拟合最终的一次相位平面。
    // 中心化、归一化坐标可以避免大像素坐标导致方程病态：
    //   xn=(x-cx)/scale, yn=(y-cy)/scale
    //   phi_fit=a*xn+b*yn+c
    const double cx = input.wrapped.cols / 2.0;
    const double cy = input.wrapped.rows / 2.0;
    const double scale = static_cast<double>(std::max(input.wrapped.cols, input.wrapped.rows));
    // 累加 A^T*A 和 A^T*B，避免为数百万拟合点分配巨大的 N×3 矩阵。
    // 最后仍使用 SVD 求解这个三参数最小二乘问题。
    cv::Matx33d normal = cv::Matx33d::zeros();
    cv::Vec3d rhs(0.0, 0.0, 0.0);
    long long sample_count = 0;

    for (int y = roi.y; y < roi.y + roi.height; ++y) {
        const double yn = (y - cy) / scale;
        const double* phase = input.wrapped.ptr<double>(y);
        const float* modulation = input.modulation.ptr<float>(y);
        for (int x = roi.x; x < roi.x + roi.width; ++x) {
            const double wrapped = phase[x];
            if (modulation[x] < kModulationFloor || std::abs(wrapped) >= kFitPhaseLimit) { //<0.6π 远离包裹跳变
                continue;
            }
            // 这里的 guide 替代“手工框 ROI 并指定 K”的环节。
            // 用最近整数确定该样本属于哪个 2*pi 周期：
            //   K=round((guide-wrapped)/(2*pi))
            const double guide = guide_dx * x + guide_dy * y + guide_c;
            const int order = static_cast<int>(std::floor((guide - wrapped) / kTwoPi + 0.5)); // 就是k
            const double absolute = wrapped + order * kTwoPi;
            const cv::Vec3d basis((x - cx) / scale, yn, 1.0);
            for (int r = 0; r < 3; ++r) {
                rhs[r] += basis[r] * absolute;
                for (int c = 0; c < 3; ++c) {
                    normal(r, c) += basis[r] * basis[c];
                }
            }
            ++sample_count;
        }
    }
    if (sample_count < 3) {
        throw std::runtime_error("not enough automatically selected fit samples");
    }
    // svd 求平面。
    cv::Vec3d plane;
    if (!cv::solve(normal, rhs, plane, cv::DECOMP_SVD)) {
        throw std::runtime_error("plane fit failed");
    }

    // 先复制原包裹相位，因此下面只覆盖 ROI；ROI 外不会被解包裹或置零。
    FitResult result;
    result.plane = plane;
    result.unwrapped = input.wrapped.clone();
    result.residual = input.wrapped.clone();
    result.sample_count = sample_count;
    result.guide_dx = guide_dx;
    result.guide_dy = guide_dy;

    for (int y = roi.y; y < roi.y + roi.height; ++y) {
        const double yn = (y - cy) / scale;
        const double* phase = input.wrapped.ptr<double>(y);
        double* unwrapped = result.unwrapped.ptr<double>(y);
        double* residual = result.residual.ptr<double>(y);
        for (int x = roi.x; x < roi.x + roi.width; ++x) {
            const double trend = plane[0] * (x - cx) / scale + plane[1] * yn + plane[2];
            // 第 4 步：用最终拟合平面重新计算每个 ROI 像素的 K。
            // residual 是实际解相位减去最终趋势，而不是减去前面的粗略 guide。
            const int order = static_cast<int>(std::floor((trend - phase[x]) / kTwoPi + 0.5));
            const double absolute = phase[x] + order * kTwoPi;
            unwrapped[x] = absolute;
            residual[x] = absolute - trend;
        }
    }
    return result;
}

void saveResults(const fs::path& output, const InputPhase& input,
                 const FitResult& result, const cv::Rect& requested_roi) {
    fs::create_directories(output);
    const cv::Rect roi = requested_roi & cv::Rect(0, 0, input.wrapped.cols, input.wrapped.rows);

    // 浮点 TIFF 保存弧度值；两张图都是全尺寸，ROI 外保留原包裹相位。
    cv::Mat1f unwrapped_float;
    cv::Mat1f residual_float;
    result.unwrapped.convertTo(unwrapped_float, CV_32F);
    result.residual.convertTo(residual_float, CV_32F);
    writeImage(output / "unwrapped_roi_outside_original.tiff", unwrapped_float);
    writeImage(output / "residual_roi_outside_original.tiff", residual_float);
    // 仅输出 ROI 裁剪图：gray=clip(-200*r+128)。
    cv::Mat1b residual_kb;
    result.residual(roi).convertTo(residual_kb, CV_8U, -200.0, 128.0);
    writeImage(output / "residual_roi_k-200_b128.png", residual_kb);

    // 报告中的统计量只计算 ROI 内有效调制度像素。
    double sum = 0.0;
    double sum2 = 0.0;
    long long count = 0;
    for (int y = roi.y; y < roi.y + roi.height; ++y) {
        const double* residual = result.residual.ptr<double>(y);
        const float* modulation = input.modulation.ptr<float>(y);
        for (int x = roi.x; x < roi.x + roi.width; ++x) {
            if (modulation[x] < kModulationFloor) {
                continue;
            }
            sum += residual[x];
            sum2 += residual[x] * residual[x];
            ++count;
        }
    }
    std::ofstream report(output / "report.txt");
    report << std::setprecision(12)
           << "roi_bounds_x0_y0_x1_y1=" << roi.x << ',' << roi.y << ','
           << roi.x + roi.width << ',' << roi.y + roi.height << '\n'
           << "roi_size_width_height=" << roi.width << ',' << roi.height << '\n'
           << "fit_samples=" << result.sample_count << '\n'
           << "guide_dx=" << result.guide_dx << '\n'
           << "guide_dy=" << result.guide_dy << '\n'
           << "plane_a_b_c=" << result.plane[0] << ',' << result.plane[1] << ',' << result.plane[2] << '\n'
           << "residual_mean=" << sum / count << '\n'
           << "residual_rms=" << std::sqrt(sum2 / count) << '\n';
}

int wmain(int argc, wchar_t** argv) {
    try {
        if (argc != 3 && argc != 7) {
            std::wcerr << L"Usage: simple_phase_plane_unwrap <phase-image|raw-dir> <output-dir> "
                          L"[x0 y0 x1 y1]\n";
            return 2;
        }
        const fs::path input_path = argv[1];
        const fs::path output_path = argv[2];
        const InputPhase input = loadInput(input_path);

        // 未指定坐标时使用 data2 调试阶段确定的默认 ROI。
        int x0 = 100;
        int y0 = 1200;
        int x1 = 4701;
        int y1 = 3648;
        if (argc == 7) {
            x0 = std::stoi(argv[3]);
            y0 = std::stoi(argv[4]);
            x1 = std::stoi(argv[5]);
            y1 = std::stoi(argv[6]);
        }
        const cv::Rect roi(x0, y0, x1 - x0, y1 - y0);
        const FitResult result = unwrapWithAutomaticROI(input, roi);
        saveResults(output_path, input, result, roi);
        std::cout << "fit_samples=" << result.sample_count << "\n"
                  << "plane=" << result.plane << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
