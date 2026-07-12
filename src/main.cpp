#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include <pcl/common/common.h>
#include <pcl/conversions.h>
#include <pcl/filters/random_sample.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/ply_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/region_growing.h>

namespace fs = std::filesystem;

using PointT = pcl::PointXYZ;
using CloudT = pcl::PointCloud<PointT>;
using CloudPtr = CloudT::Ptr;
using ColorPointT = pcl::PointXYZRGB;
using ColorCloudT = pcl::PointCloud<ColorPointT>;
using ColorCloudPtr = ColorCloudT::Ptr;

struct Config {
    std::vector<fs::path> inputs;
    fs::path output_dir = "corner_results";
    float min_x = -std::numeric_limits<float>::infinity();
    float max_x = std::numeric_limits<float>::infinity();
    float min_y = -std::numeric_limits<float>::infinity();
    float max_y = std::numeric_limits<float>::infinity();
    float min_z = -std::numeric_limits<float>::infinity();
    float max_z = std::numeric_limits<float>::infinity();
    float merge_plane_angle_deg = 8.0f;
    float merge_plane_distance_scale = 12.0f;
    float merge_plane_adjacency_scale = 20.0f;
    float expand_plane_distance_scale = 2.5f;
    float expand_plane_margin_scale = 20.0f;
    float min_plane_extent_scale = 12.0f;
    float min_plane_aspect_ratio = 0.08f;
    float robust_fit_keep_ratio = 0.85f;
    int robust_fit_iterations = 2;
    float strict_plane_distance_scale = 0.75f;
    float support_conflict_scale = 0.5f;
    float core_conflict_scale = 0.25f;
    float min_visible_support_core_ratio = 0.5f;
    float min_visible_support_median_ratio = 0.10f;
    float min_visible_density_median_ratio = 0.05f;
    std::vector<float> atom_cell_scales{16.0f, 32.0f, 64.0f};
    int atom_min_points = 60;
    float atom_max_planarity_error = 0.035f;
    float atom_max_rms_scale = 1.0f;
    float atom_merge_angle_deg = 10.0f;
    float atom_merge_distance_scale = 4.0f;
    bool region_growing_plane_mode = false;
    int normal_k = 24;
    int region_neighbors = 30;
    float region_smoothness_deg = 3.0f;
    float region_curvature_threshold = 0.05f;
    int region_min_cluster_size = 200;
    int max_planes = 80;
    int min_plane_points = 80;
    float max_planarity_error = 0.06f;

    float normal_parallel_cos = 0.975f;
    float intersection_merge_scale = 4.5f;
    float corner_support_radius_scale = 7.0f;
    float corner_edge_radius_scale = 3.5f;
    std::vector<float> corner_edge_multi_scales{2.5f, 4.5f, 7.0f};
    int corner_edge_sample_limit = 40000;
    float max_corner_edge_score = 0.0f;
    float max_corner_edge_growth_ratio = 0.0f;
    int min_corner_support = 8;
    int min_plane_local_support = 3;
    int min_corner_planes = 1;
    int boundary_min_plane_supports = 1;
    float hull_min_edge_scale = 10.0f;
    float hull_corner_cos = 0.8f;
    float hull_max_plane_fraction = 0.4f;
    int resolution_sample_cap = 5000;
    bool legacy_resolution_estimator = true;
    bool verbose = true;
    int load_limit = 0;
    float voxel_leaf = 0.0f;
    float voxel_leaf_scale = 0.0f;
    int sor_mean_k = 0;
    float sor_stddev = 1.0f;
    float ror_radius_scale = 0.0f;
    int ror_min_neighbors = 0;
    float input_scale = 1.0f;
    bool preserve_input_points = false;
};

struct Bounds {
    PointT min_pt;
    PointT max_pt;
};

struct PlanePatch;
static Bounds compute_bounds(const CloudPtr& cloud);
static void plane_basis(const PlanePatch& patch, Eigen::Vector3f& axis_u, Eigen::Vector3f& axis_v);

struct PreparedCloud {
    CloudPtr points{new CloudT};
    CloudPtr full_points{new CloudT};
    float model_resolution = 0.0f;
};

struct PlanePatch {
    Eigen::Vector3f normal{0.0f, 0.0f, 1.0f};
    float d = 0.0f;
    CloudPtr cloud{new CloudT};
    CloudPtr support_cloud;
    CloudPtr output_cloud;
    Eigen::Vector3f centroid{0.0f, 0.0f, 0.0f};
    float planarity_error = 0.0f;
};

struct CornerCandidate {
    Eigen::Vector3f position{0.0f, 0.0f, 0.0f};
    std::vector<int> plane_ids;
    int support = 0;
};

struct ProjectedPoint {
    float x = 0.0f;
    float y = 0.0f;
    int index = 0;
};

struct CornerDetectionResult {
    std::vector<PlanePatch> plane_patches;
    std::vector<CornerCandidate> corners;
};

struct PlaneEnvelope {
    PlanePatch seed;
    Eigen::Vector3f axis_u{1.0f, 0.0f, 0.0f};
    Eigen::Vector3f axis_v{0.0f, 1.0f, 0.0f};
    float min_u = 0.0f;
    float max_u = 0.0f;
    float min_v = 0.0f;
    float max_v = 0.0f;
    CloudPtr assigned{new CloudT};
};

struct PlaneShapeMetrics {
    float extent_u = 0.0f;
    float extent_v = 0.0f;
    float short_extent = 0.0f;
    float long_extent = 0.0f;
    float aspect_ratio = 0.0f;
};

struct CellKey {
    int x = 0;
    int y = 0;
    int z = 0;

    bool operator==(const CellKey& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct CellKeyHash {
    std::size_t operator()(const CellKey& key) const {
        std::size_t h = static_cast<std::size_t>(key.x) * 73856093u;
        h ^= static_cast<std::size_t>(key.y) * 19349663u;
        h ^= static_cast<std::size_t>(key.z) * 83492791u;
        return h;
    }
};

struct PlaneAtom {
    CellKey key;
    PlanePatch patch;
    float rms_residual = 0.0f;
};

struct DisjointSet {
    std::vector<int> parent;
    std::vector<int> rank;

    explicit DisjointSet(std::size_t n) : parent(n), rank(n, 0) {
        std::iota(parent.begin(), parent.end(), 0);
    }

    int find(int x) {
        if (parent[static_cast<std::size_t>(x)] != x) parent[static_cast<std::size_t>(x)] = find(parent[static_cast<std::size_t>(x)]);
        return parent[static_cast<std::size_t>(x)];
    }

    void unite(int a, int b) {
        int ra = find(a);
        int rb = find(b);
        if (ra == rb) return;
        if (rank[static_cast<std::size_t>(ra)] < rank[static_cast<std::size_t>(rb)]) std::swap(ra, rb);
        parent[static_cast<std::size_t>(rb)] = ra;
        if (rank[static_cast<std::size_t>(ra)] == rank[static_cast<std::size_t>(rb)]) ++rank[static_cast<std::size_t>(ra)];
    }
};

static std::string lower_copy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

static bool has_flag(const std::vector<std::string>& args, const std::string& flag) {
    return std::find(args.begin(), args.end(), flag) != args.end();
}

static std::optional<std::string> value_after(const std::vector<std::string>& args, const std::string& key) {
    for (std::size_t i = 0; i + 1 < args.size(); ++i) {
        if (args[i] == key) return args[i + 1];
    }
    return std::nullopt;
}

static std::vector<float> parse_float_list(const std::string& text) {
    std::vector<float> values;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) values.push_back(std::stof(item));
    }
    return values;
}

static Config parse_args(int argc, char** argv) {
    Config cfg;
    std::vector<std::string> args(argv + 1, argv + argc);

    if (has_flag(args, "--help") || has_flag(args, "-h")) {
        std::cout
            << "Usage: pointcloud_corner_compare [inputs...] [options]\n"
            << "Common options: --output-dir, --input-scale, --min/max-x/y/z, --atom-cell-scales,\n"
            << "  --region-growing-plane-mode, --merge-plane-*, --expand-plane-*, --min-corner-planes,\n"
            << "  --corner-edge-*, --sor-*, --ror-*, --voxel-leaf-scale.\n"
            << "  See README.md for recommended parameter sets.\n";
        std::exit(0);
    }

    if (auto v = value_after(args, "--output-dir")) cfg.output_dir = *v;
    if (auto v = value_after(args, "--min-x")) cfg.min_x = std::stof(*v);
    if (auto v = value_after(args, "--max-x")) cfg.max_x = std::stof(*v);
    if (auto v = value_after(args, "--min-y")) cfg.min_y = std::stof(*v);
    if (auto v = value_after(args, "--max-y")) cfg.max_y = std::stof(*v);
    if (auto v = value_after(args, "--min-z")) cfg.min_z = std::stof(*v);
    if (auto v = value_after(args, "--max-z")) cfg.max_z = std::stof(*v);
    if (auto v = value_after(args, "--merge-plane-angle-deg")) cfg.merge_plane_angle_deg = std::stof(*v);
    if (auto v = value_after(args, "--merge-plane-distance-scale")) cfg.merge_plane_distance_scale = std::stof(*v);
    if (auto v = value_after(args, "--merge-plane-adjacency-scale")) cfg.merge_plane_adjacency_scale = std::stof(*v);
    if (auto v = value_after(args, "--expand-plane-distance-scale")) cfg.expand_plane_distance_scale = std::stof(*v);
    if (auto v = value_after(args, "--expand-plane-margin-scale")) cfg.expand_plane_margin_scale = std::stof(*v);
    if (auto v = value_after(args, "--min-plane-extent-scale")) cfg.min_plane_extent_scale = std::stof(*v);
    if (auto v = value_after(args, "--min-plane-aspect-ratio")) cfg.min_plane_aspect_ratio = std::stof(*v);
    if (auto v = value_after(args, "--robust-fit-keep-ratio")) cfg.robust_fit_keep_ratio = std::stof(*v);
    if (auto v = value_after(args, "--robust-fit-iterations")) cfg.robust_fit_iterations = std::stoi(*v);
    if (auto v = value_after(args, "--strict-plane-distance-scale")) cfg.strict_plane_distance_scale = std::stof(*v);
    if (auto v = value_after(args, "--support-conflict-scale")) cfg.support_conflict_scale = std::stof(*v);
    if (auto v = value_after(args, "--core-conflict-scale")) cfg.core_conflict_scale = std::stof(*v);
    if (auto v = value_after(args, "--min-visible-support-core-ratio")) cfg.min_visible_support_core_ratio = std::stof(*v);
    if (auto v = value_after(args, "--min-visible-support-median-ratio")) cfg.min_visible_support_median_ratio = std::stof(*v);
    if (auto v = value_after(args, "--min-visible-density-median-ratio")) cfg.min_visible_density_median_ratio = std::stof(*v);
    if (auto v = value_after(args, "--atom-cell-scales")) {
        auto values = parse_float_list(*v);
        if (!values.empty()) cfg.atom_cell_scales = values;
    }
    if (auto v = value_after(args, "--atom-min-points")) cfg.atom_min_points = std::stoi(*v);
    if (auto v = value_after(args, "--atom-max-planarity-error")) cfg.atom_max_planarity_error = std::stof(*v);
    if (auto v = value_after(args, "--atom-max-rms-scale")) cfg.atom_max_rms_scale = std::stof(*v);
    if (auto v = value_after(args, "--atom-merge-angle-deg")) cfg.atom_merge_angle_deg = std::stof(*v);
    if (auto v = value_after(args, "--atom-merge-distance-scale")) cfg.atom_merge_distance_scale = std::stof(*v);
    cfg.region_growing_plane_mode = has_flag(args, "--region-growing-plane-mode");
    if (auto v = value_after(args, "--normal-k")) cfg.normal_k = std::stoi(*v);
    if (auto v = value_after(args, "--region-neighbors")) cfg.region_neighbors = std::stoi(*v);
    if (auto v = value_after(args, "--region-smoothness-deg")) cfg.region_smoothness_deg = std::stof(*v);
    if (auto v = value_after(args, "--region-curvature-threshold")) cfg.region_curvature_threshold = std::stof(*v);
    if (auto v = value_after(args, "--region-min-cluster-size")) cfg.region_min_cluster_size = std::stoi(*v);
    if (auto v = value_after(args, "--max-planes")) cfg.max_planes = std::stoi(*v);
    if (auto v = value_after(args, "--min-plane-points")) cfg.min_plane_points = std::stoi(*v);
    if (auto v = value_after(args, "--max-planarity-error")) cfg.max_planarity_error = std::stof(*v);
    if (auto v = value_after(args, "--normal-parallel-cos")) cfg.normal_parallel_cos = std::stof(*v);
    if (auto v = value_after(args, "--intersection-merge-scale")) cfg.intersection_merge_scale = std::stof(*v);
    if (auto v = value_after(args, "--corner-support-radius-scale")) cfg.corner_support_radius_scale = std::stof(*v);
    if (auto v = value_after(args, "--corner-edge-radius-scale")) cfg.corner_edge_radius_scale = std::stof(*v);
    if (auto v = value_after(args, "--corner-edge-multi-scales")) {
        auto values = parse_float_list(*v);
        if (values.size() >= 3) cfg.corner_edge_multi_scales = {values[0], values[1], values[2]};
    }
    if (auto v = value_after(args, "--corner-edge-sample-limit")) cfg.corner_edge_sample_limit = std::stoi(*v);
    if (auto v = value_after(args, "--max-corner-edge-score")) cfg.max_corner_edge_score = std::stof(*v);
    if (auto v = value_after(args, "--max-corner-edge-growth-ratio")) cfg.max_corner_edge_growth_ratio = std::stof(*v);
    if (auto v = value_after(args, "--min-corner-support")) cfg.min_corner_support = std::stoi(*v);
    if (auto v = value_after(args, "--min-plane-local-support")) cfg.min_plane_local_support = std::stoi(*v);
    if (auto v = value_after(args, "--min-corner-planes")) cfg.min_corner_planes = std::stoi(*v);
    if (auto v = value_after(args, "--boundary-min-plane-supports")) cfg.boundary_min_plane_supports = std::stoi(*v);
    if (auto v = value_after(args, "--hull-min-edge-scale")) cfg.hull_min_edge_scale = std::stof(*v);
    if (auto v = value_after(args, "--hull-corner-cos")) cfg.hull_corner_cos = std::stof(*v);
    if (auto v = value_after(args, "--hull-max-plane-fraction")) cfg.hull_max_plane_fraction = std::stof(*v);
    if (auto v = value_after(args, "--resolution-sample-cap")) cfg.resolution_sample_cap = std::stoi(*v);
    if (auto v = value_after(args, "--load-limit")) cfg.load_limit = std::stoi(*v);
    if (auto v = value_after(args, "--voxel-leaf")) cfg.voxel_leaf = std::stof(*v);
    if (auto v = value_after(args, "--voxel-leaf-scale")) cfg.voxel_leaf_scale = std::stof(*v);
    if (auto v = value_after(args, "--sor-mean-k")) cfg.sor_mean_k = std::stoi(*v);
    if (auto v = value_after(args, "--sor-stddev")) cfg.sor_stddev = std::stof(*v);
    if (auto v = value_after(args, "--ror-radius-scale")) cfg.ror_radius_scale = std::stof(*v);
    if (auto v = value_after(args, "--ror-min-neighbors")) cfg.ror_min_neighbors = std::stoi(*v);
    if (auto v = value_after(args, "--input-scale")) cfg.input_scale = std::stof(*v);
    cfg.preserve_input_points = has_flag(args, "--preserve-input-points");
    cfg.legacy_resolution_estimator = !has_flag(args, "--point-spacing-resolution-estimator");
    if (has_flag(args, "--legacy-resolution-estimator")) cfg.legacy_resolution_estimator = true;
    cfg.verbose = !has_flag(args, "--quiet");
    const std::set<std::string> valued = {
        "--output-dir",
        "--min-x", "--max-x", "--min-y", "--max-y", "--min-z", "--max-z",
        "--merge-plane-angle-deg", "--merge-plane-distance-scale", "--merge-plane-adjacency-scale",
        "--expand-plane-distance-scale", "--expand-plane-margin-scale",
        "--min-plane-extent-scale", "--min-plane-aspect-ratio",
        "--robust-fit-keep-ratio", "--robust-fit-iterations", "--strict-plane-distance-scale", "--support-conflict-scale", "--core-conflict-scale",
        "--min-visible-support-core-ratio", "--min-visible-support-median-ratio", "--min-visible-density-median-ratio",
        "--atom-cell-scales", "--atom-min-points", "--atom-max-planarity-error",
        "--atom-max-rms-scale", "--atom-merge-angle-deg", "--atom-merge-distance-scale",
        "--normal-k", "--region-neighbors", "--region-smoothness-deg",
        "--region-curvature-threshold", "--region-min-cluster-size",
        "--max-planes", "--min-plane-points", "--max-planarity-error",
        "--normal-parallel-cos", "--intersection-merge-scale", "--corner-support-radius-scale",
        "--corner-edge-radius-scale", "--corner-edge-multi-scales", "--corner-edge-sample-limit", "--max-corner-edge-score", "--max-corner-edge-growth-ratio",
        "--min-corner-support", "--min-plane-local-support", "--min-corner-planes", "--boundary-min-plane-supports",
        "--hull-min-edge-scale", "--hull-corner-cos", "--hull-max-plane-fraction",
        "--resolution-sample-cap",
        "--load-limit", "--voxel-leaf", "--voxel-leaf-scale",
        "--sor-mean-k", "--sor-stddev", "--ror-radius-scale", "--ror-min-neighbors",
        "--input-scale"
    };

    for (std::size_t i = 0; i < args.size(); ++i) {
        if (valued.count(args[i])) {
            ++i;
            continue;
        }
        if (!args[i].empty() && args[i].front() != '-') cfg.inputs.emplace_back(args[i]);
    }
    return cfg;
}

static std::vector<fs::path> discover_inputs(const Config& cfg) {
    if (!cfg.inputs.empty()) return cfg.inputs;

    std::vector<fs::path> result;
    const auto output_dir_abs = fs::weakly_canonical(cfg.output_dir);
    for (const auto& entry : fs::recursive_directory_iterator(fs::current_path())) {
        if (!entry.is_regular_file()) continue;
        if (lower_copy(entry.path().extension().string()) != ".ply") continue;
        const auto abs = fs::weakly_canonical(entry.path());
        if (abs.string().find(output_dir_abs.string()) == 0) continue;
        result.push_back(entry.path());
    }
    std::sort(result.begin(), result.end());
    return result;
}

static CloudPtr load_xyz_cloud(const fs::path& path, float input_scale) {
    pcl::PCLPointCloud2 blob;
    if (pcl::io::loadPLYFile(path.string(), blob) != 0) {
        throw std::runtime_error("Failed to load PLY: " + path.string());
    }
    CloudPtr cloud(new CloudT);
    pcl::fromPCLPointCloud2(blob, *cloud);
    if (input_scale != 1.0f) {
        for (auto& p : cloud->points) {
            p.x *= input_scale;
            p.y *= input_scale;
            p.z *= input_scale;
        }
    }
    return cloud;
}

static Bounds compute_bounds(const CloudPtr& cloud) {
    Bounds bounds;
    pcl::getMinMax3D(*cloud, bounds.min_pt, bounds.max_pt);
    return bounds;
}

static CloudPtr sample_cloud_for_resolution(const CloudPtr& cloud, int sample_cap) {
    CloudPtr sample(new CloudT);
    if (cloud->size() <= static_cast<std::size_t>(sample_cap)) {
        sample.reset(new CloudT(*cloud));
    } else {
        pcl::RandomSample<PointT> sampler;
        sampler.setInputCloud(cloud);
        sampler.setSample(static_cast<unsigned int>(sample_cap));
        sampler.setSeed(42);
        sampler.filter(*sample);
    }
    return sample;
}

static float robust_mean_nearest_distance(std::vector<float> distances) {
    distances.erase(
        std::remove_if(distances.begin(), distances.end(), [](float d) {
            return !std::isfinite(d) || d <= 1e-8f;
        }),
        distances.end());
    if (distances.empty()) return 0.0f;

    std::sort(distances.begin(), distances.end());
    const std::size_t trim = distances.size() >= 20 ? distances.size() / 20 : 0;
    const std::size_t first = trim;
    const std::size_t last = distances.size() - trim;
    double sum = 0.0;
    for (std::size_t i = first; i < last; ++i) sum += distances[i];
    return static_cast<float>(sum / static_cast<double>(std::max<std::size_t>(last - first, 1)));
}

static float estimate_model_resolution(const CloudPtr& cloud, const Config& cfg) {
    if (cloud->size() < 2) return 0.0f;
    const int sample_cap = std::max(cfg.resolution_sample_cap, 2);
    CloudPtr sample = sample_cloud_for_resolution(cloud, sample_cap);

    pcl::KdTreeFLANN<PointT> tree;
    tree.setInputCloud(cfg.legacy_resolution_estimator ? sample : cloud);

    std::vector<int> indices(2);
    std::vector<float> distances(2);
    if (cfg.legacy_resolution_estimator) {
        double sum = 0.0;
        int used = 0;
        for (std::size_t i = 0; i < sample->size(); ++i) {
            if (tree.nearestKSearch(sample->points[i], 2, indices, distances) == 2) {
                sum += std::sqrt(distances[1]);
                ++used;
            }
        }
        return used > 0 ? static_cast<float>(sum / used) : 0.0f;
    }

    std::vector<float> nearest_distances;
    nearest_distances.reserve(sample->size());
    for (std::size_t i = 0; i < sample->size(); ++i) {
        if (tree.nearestKSearch(sample->points[i], 2, indices, distances) == 2) {
            nearest_distances.push_back(std::sqrt(distances[1]));
        }
    }
    return robust_mean_nearest_distance(std::move(nearest_distances));
}

static CloudPtr random_limit_cloud(const CloudPtr& cloud, int limit) {
    if (limit <= 0 || cloud->size() <= static_cast<std::size_t>(limit)) return cloud;
    CloudPtr filtered(new CloudT);
    pcl::RandomSample<PointT> sampler;
    sampler.setInputCloud(cloud);
    sampler.setSample(static_cast<unsigned int>(limit));
    sampler.setSeed(42);
    sampler.filter(*filtered);
    return filtered;
}

static CloudPtr voxel_downsample_cloud(const CloudPtr& cloud, float leaf_size) {
    if (leaf_size <= 0.0f || cloud->empty()) return cloud;
    CloudPtr filtered(new CloudT);
    pcl::VoxelGrid<PointT> voxel;
    voxel.setInputCloud(cloud);
    voxel.setLeafSize(leaf_size, leaf_size, leaf_size);
    voxel.filter(*filtered);
    return filtered;
}

static CloudPtr remove_statistical_outliers(const CloudPtr& cloud, int mean_k, float stddev) {
    if (mean_k <= 1 || cloud->size() <= static_cast<std::size_t>(mean_k)) return cloud;
    CloudPtr filtered(new CloudT);
    pcl::StatisticalOutlierRemoval<PointT> sor;
    sor.setInputCloud(cloud);
    sor.setMeanK(mean_k);
    sor.setStddevMulThresh(std::max(stddev, 0.01f));
    sor.filter(*filtered);
    return filtered;
}

static CloudPtr remove_radius_outliers(const CloudPtr& cloud, float radius, int min_neighbors) {
    if (radius <= 0.0f || min_neighbors <= 0 || cloud->empty()) return cloud;
    CloudPtr filtered(new CloudT);
    pcl::RadiusOutlierRemoval<PointT> ror;
    ror.setInputCloud(cloud);
    ror.setRadiusSearch(radius);
    ror.setMinNeighborsInRadius(min_neighbors);
    ror.filter(*filtered);
    return filtered;
}

static CloudPtr apply_preprocessing_filters(const CloudPtr& cloud, const Config& cfg, float reference_resolution) {
    CloudPtr filtered = random_limit_cloud(cloud, cfg.load_limit);
    const float scaled_leaf = cfg.voxel_leaf_scale > 0.0f
        ? cfg.voxel_leaf_scale * reference_resolution
        : 0.0f;
    const float leaf = cfg.voxel_leaf > 0.0f ? cfg.voxel_leaf : scaled_leaf;
    filtered = voxel_downsample_cloud(filtered, leaf);
    filtered = remove_statistical_outliers(filtered, cfg.sor_mean_k, cfg.sor_stddev);
    const float ror_radius = cfg.ror_radius_scale > 0.0f
        ? cfg.ror_radius_scale * reference_resolution
        : 0.0f;
    filtered = remove_radius_outliers(filtered, ror_radius, cfg.ror_min_neighbors);
    return filtered;
}

static PreparedCloud prepare_cloud(const CloudPtr& input_cloud, const Config& cfg) {
    CloudPtr cropped(new CloudT);

    const bool has_crop =
        std::isfinite(cfg.min_x) || std::isfinite(cfg.max_x) ||
        std::isfinite(cfg.min_y) || std::isfinite(cfg.max_y) ||
        std::isfinite(cfg.min_z) || std::isfinite(cfg.max_z);
    if (has_crop) {
        cropped->reserve(input_cloud->size());
        for (const auto& p : input_cloud->points) {
            if (p.x < cfg.min_x || p.x > cfg.max_x) continue;
            if (p.y < cfg.min_y || p.y > cfg.max_y) continue;
            if (p.z < cfg.min_z || p.z > cfg.max_z) continue;
            cropped->push_back(p);
        }
        cropped->width = static_cast<std::uint32_t>(cropped->size());
        cropped->height = 1;
        cropped->is_dense = input_cloud->is_dense;
    } else {
        cropped.reset(new CloudT(*input_cloud));
    }

    PreparedCloud prepared;
    const float reference_resolution = estimate_model_resolution(cropped, cfg);
    prepared.points = apply_preprocessing_filters(cropped, cfg, reference_resolution);
    prepared.full_points = cfg.preserve_input_points ? cropped : prepared.points;
    prepared.model_resolution = estimate_model_resolution(prepared.points, cfg);
    if (prepared.model_resolution <= 0.0f) prepared.model_resolution = 1e-4f;
    return prepared;
}

static PlanePatch fit_plane_patch_pca(const CloudPtr& patch_cloud) {
    PlanePatch patch;
    patch.cloud = patch_cloud;

    for (const auto& p : patch_cloud->points) {
        patch.centroid += Eigen::Vector3f(p.x, p.y, p.z);
    }
    patch.centroid /= std::max(1.0f, static_cast<float>(patch_cloud->size()));

    Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
    for (const auto& p : patch_cloud->points) {
        const Eigen::Vector3f q(p.x, p.y, p.z);
        const Eigen::Vector3f centered = q - patch.centroid;
        covariance += centered * centered.transpose();
    }
    covariance /= std::max(1.0f, static_cast<float>(patch_cloud->size()));

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
    const Eigen::Vector3f evals = solver.eigenvalues();
    patch.normal = solver.eigenvectors().col(0).normalized();
    patch.d = -patch.normal.dot(patch.centroid);
    patch.planarity_error = evals.x() / std::max(evals.sum(), 1e-8f);
    return patch;
}

static CloudPtr trimmed_by_plane_residual(const PlanePatch& patch, float keep_ratio) {
    if (patch.cloud->size() < 4) return patch.cloud;
    keep_ratio = std::clamp(keep_ratio, 0.1f, 1.0f);
    std::vector<float> residuals;
    residuals.reserve(patch.cloud->size());
    for (const auto& p : patch.cloud->points) {
        const Eigen::Vector3f q(p.x, p.y, p.z);
        residuals.push_back(std::abs(patch.normal.dot(q) + patch.d));
    }
    std::vector<float> sorted = residuals;
    std::sort(sorted.begin(), sorted.end());
    const std::size_t keep_count = std::max<std::size_t>(3, static_cast<std::size_t>(std::ceil(keep_ratio * static_cast<float>(sorted.size()))));
    const float threshold = sorted[std::min(keep_count - 1, sorted.size() - 1)];

    CloudPtr trimmed(new CloudT);
    trimmed->reserve(keep_count);
    for (std::size_t i = 0; i < patch.cloud->size(); ++i) {
        if (residuals[i] <= threshold) trimmed->push_back(patch.cloud->points[i]);
    }
    trimmed->width = static_cast<std::uint32_t>(trimmed->size());
    trimmed->height = 1;
    trimmed->is_dense = patch.cloud->is_dense;
    return trimmed;
}

static PlanePatch fit_plane_patch(const CloudPtr& patch_cloud, const Config& cfg) {
    CloudPtr current(new CloudT(*patch_cloud));
    PlanePatch fitted = fit_plane_patch_pca(current);
    for (int i = 0; i < std::max(cfg.robust_fit_iterations, 0); ++i) {
        CloudPtr trimmed = trimmed_by_plane_residual(fitted, cfg.robust_fit_keep_ratio);
        if (trimmed->size() < 3 || trimmed->size() == current->size()) break;
        current = trimmed;
        fitted = fit_plane_patch_pca(current);
    }
    fitted.cloud = patch_cloud;
    return fitted;
}

static CloudPtr inliers_near_plane(const PlanePatch& plane, const CloudPtr& cloud, float distance_threshold) {
    CloudPtr inliers(new CloudT);
    inliers->reserve(cloud->size());
    for (const auto& p : cloud->points) {
        const Eigen::Vector3f q(p.x, p.y, p.z);
        if (std::abs(plane.normal.dot(q) + plane.d) <= distance_threshold) inliers->push_back(p);
    }
    inliers->width = static_cast<std::uint32_t>(inliers->size());
    inliers->height = 1;
    inliers->is_dense = cloud->is_dense;
    return inliers;
}

static PlanePatch fit_strict_plane_patch(const CloudPtr& candidate_cloud, float resolution, const Config& cfg) {
    PlanePatch loose_fit = fit_plane_patch(candidate_cloud, cfg);
    const float strict_distance = std::max(cfg.strict_plane_distance_scale * resolution, 1e-4f);
    CloudPtr strict_cloud = inliers_near_plane(loose_fit, candidate_cloud, strict_distance);
    if (strict_cloud->size() < static_cast<std::size_t>(std::max(cfg.min_plane_points, 3))) {
        loose_fit.support_cloud = candidate_cloud;
        loose_fit.output_cloud = candidate_cloud;
        return loose_fit;
    }
    PlanePatch strict_fit = fit_plane_patch(strict_cloud, cfg);
    strict_fit.support_cloud = candidate_cloud;
    strict_fit.output_cloud = candidate_cloud;
    return strict_fit;
}

static bool planes_can_merge(const PlanePatch& a, const PlanePatch& b, float normal_cos_threshold, float distance_threshold) {
    const float normal_dot = a.normal.dot(b.normal);
    if (std::abs(normal_dot) < normal_cos_threshold) return false;

    const float b_d_aligned = normal_dot >= 0.0f ? b.d : -b.d;
    if (std::abs(a.d - b_d_aligned) > distance_threshold) return false;

    const float centroid_distance = std::abs(a.normal.dot(b.centroid) + a.d);
    return centroid_distance <= distance_threshold;
}

struct ProjectedRange2D {
    float min_u = std::numeric_limits<float>::max();
    float max_u = std::numeric_limits<float>::lowest();
    float min_v = std::numeric_limits<float>::max();
    float max_v = std::numeric_limits<float>::lowest();
};

static ProjectedRange2D projected_range_on_patch_basis(
    const CloudPtr& cloud,
    const PlanePatch& basis_patch,
    const Eigen::Vector3f& axis_u,
    const Eigen::Vector3f& axis_v) {
    ProjectedRange2D range;
    for (const auto& p : cloud->points) {
        const Eigen::Vector3f q(p.x, p.y, p.z);
        const Eigen::Vector3f centered = q - basis_patch.centroid;
        const float u = centered.dot(axis_u);
        const float v = centered.dot(axis_v);
        range.min_u = std::min(range.min_u, u);
        range.max_u = std::max(range.max_u, u);
        range.min_v = std::min(range.min_v, v);
        range.max_v = std::max(range.max_v, v);
    }
    return range;
}

static bool projected_ranges_touch(const ProjectedRange2D& a, const ProjectedRange2D& b, float margin) {
    if (a.max_u + margin < b.min_u || b.max_u + margin < a.min_u) return false;
    if (a.max_v + margin < b.min_v || b.max_v + margin < a.min_v) return false;
    return true;
}

static bool patches_are_projected_neighbors(const PlanePatch& a, const PlanePatch& b, float margin) {
    if (margin <= 0.0f) return true;
    if (a.cloud->empty() || b.cloud->empty()) return false;

    Eigen::Vector3f axis_u;
    Eigen::Vector3f axis_v;
    plane_basis(a, axis_u, axis_v);

    const ProjectedRange2D range_a = projected_range_on_patch_basis(a.cloud, a, axis_u, axis_v);
    const ProjectedRange2D range_b = projected_range_on_patch_basis(b.cloud, a, axis_u, axis_v);
    return projected_ranges_touch(range_a, range_b, margin);
}

static CloudPtr visible_cloud_for_patch(const PlanePatch& patch) {
    return patch.support_cloud && !patch.support_cloud->empty() ? patch.support_cloud : patch.cloud;
}

static CloudPtr output_cloud_for_patch(const PlanePatch& patch) {
    return patch.output_cloud && !patch.output_cloud->empty() ? patch.output_cloud : visible_cloud_for_patch(patch);
}

static std::vector<PlanePatch> merge_coplanar_patches(std::vector<PlanePatch> patches, float resolution, const Config& cfg) {
    const float angle_rad = cfg.merge_plane_angle_deg / 180.0f * 3.14159265358979323846f;
    const float normal_cos_threshold = std::cos(std::max(angle_rad, 0.0f));
    const float distance_threshold = std::max(cfg.merge_plane_distance_scale * resolution, 1e-4f);
    const float adjacency_margin = cfg.merge_plane_adjacency_scale > 0.0f
        ? std::max(cfg.merge_plane_adjacency_scale * resolution, distance_threshold)
        : 0.0f;

    std::vector<PlanePatch> merged;
    for (const auto& patch : patches) {
        bool absorbed = false;
        for (auto& group : merged) {
            if (!planes_can_merge(group, patch, normal_cos_threshold, distance_threshold)) continue;
            if (!patches_are_projected_neighbors(group, patch, adjacency_margin)) continue;
            CloudPtr merged_support(new CloudT);
            *merged_support += *visible_cloud_for_patch(group);
            *merged_support += *visible_cloud_for_patch(patch);
            CloudPtr merged_output(new CloudT);
            *merged_output += *output_cloud_for_patch(group);
            *merged_output += *output_cloud_for_patch(patch);
            *group.cloud += *patch.cloud;
            group = fit_plane_patch(group.cloud, cfg);
            group.support_cloud = merged_support;
            group.output_cloud = merged_output;
            absorbed = true;
            break;
        }
        if (!absorbed) merged.push_back(patch);
    }

    std::sort(merged.begin(), merged.end(), [](const PlanePatch& a, const PlanePatch& b) {
        return a.cloud->size() > b.cloud->size();
    });
    if (merged.size() > static_cast<std::size_t>(cfg.max_planes)) merged.resize(static_cast<std::size_t>(cfg.max_planes));
    return merged;
}

static float rms_plane_residual(const PlanePatch& patch) {
    if (patch.cloud->empty()) return std::numeric_limits<float>::max();
    double sum_sq = 0.0;
    for (const auto& p : patch.cloud->points) {
        const Eigen::Vector3f q(p.x, p.y, p.z);
        const float distance = patch.normal.dot(q) + patch.d;
        sum_sq += static_cast<double>(distance) * static_cast<double>(distance);
    }
    return static_cast<float>(std::sqrt(sum_sq / static_cast<double>(patch.cloud->size())));
}

static CellKey point_cell_key(const PointT& p, const Bounds& bounds, float cell_size) {
    return {
        static_cast<int>(std::floor((p.x - bounds.min_pt.x) / cell_size)),
        static_cast<int>(std::floor((p.y - bounds.min_pt.y) / cell_size)),
        static_cast<int>(std::floor((p.z - bounds.min_pt.z) / cell_size))
    };
}

static bool atoms_are_compatible(const PlaneAtom& a, const PlaneAtom& b, float normal_cos_threshold, float distance_threshold) {
    return planes_can_merge(a.patch, b.patch, normal_cos_threshold, distance_threshold);
}

static std::vector<PlanePatch> extract_plane_patches_atoms_one_scale(const CloudPtr& cloud, const Bounds& bounds, float resolution, float cell_scale, const Config& cfg) {
    const float cell_size = std::max(cell_scale * resolution, 1e-4f);
    std::unordered_map<CellKey, std::vector<int>, CellKeyHash> cells;
    cells.reserve(cloud->size() / static_cast<std::size_t>(std::max(cfg.atom_min_points, 1)) + 1);

    for (std::size_t i = 0; i < cloud->size(); ++i) {
        cells[point_cell_key(cloud->points[i], bounds, cell_size)].push_back(static_cast<int>(i));
    }

    std::vector<PlaneAtom> atoms;
    std::unordered_map<CellKey, int, CellKeyHash> atom_by_key;
    const float max_rms = std::max(cfg.atom_max_rms_scale * resolution, 1e-4f);
    for (const auto& [key, indices] : cells) {
        if (indices.size() < static_cast<std::size_t>(std::max(cfg.atom_min_points, 3))) continue;

        CloudPtr atom_cloud(new CloudT);
        atom_cloud->reserve(indices.size());
        for (int idx : indices) atom_cloud->push_back(cloud->points[static_cast<std::size_t>(idx)]);
        atom_cloud->width = static_cast<std::uint32_t>(atom_cloud->size());
        atom_cloud->height = 1;
        atom_cloud->is_dense = cloud->is_dense;

        PlanePatch patch = fit_plane_patch(atom_cloud, cfg);
        const float rms = rms_plane_residual(patch);
        if (patch.planarity_error > cfg.atom_max_planarity_error) continue;
        if (rms > max_rms) continue;

        const int atom_id = static_cast<int>(atoms.size());
        atoms.push_back({key, std::move(patch), rms});
        atom_by_key[key] = atom_id;
    }

    if (atoms.empty()) return {};

    DisjointSet dsu(atoms.size());
    const float normal_cos_threshold = std::cos(cfg.atom_merge_angle_deg / 180.0f * 3.14159265358979323846f);
    const float distance_threshold = std::max(cfg.atom_merge_distance_scale * resolution, 1e-4f);

    for (std::size_t i = 0; i < atoms.size(); ++i) {
        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dz = -1; dz <= 1; ++dz) {
                    if (dx == 0 && dy == 0 && dz == 0) continue;
                    const CellKey neighbor{atoms[i].key.x + dx, atoms[i].key.y + dy, atoms[i].key.z + dz};
                    auto it = atom_by_key.find(neighbor);
                    if (it == atom_by_key.end()) continue;
                    const int j = it->second;
                    if (j <= static_cast<int>(i)) continue;
                    if (atoms_are_compatible(atoms[i], atoms[static_cast<std::size_t>(j)], normal_cos_threshold, distance_threshold)) {
                        dsu.unite(static_cast<int>(i), j);
                    }
                }
            }
        }
    }

    std::unordered_map<int, CloudPtr> component_clouds;
    for (std::size_t i = 0; i < atoms.size(); ++i) {
        const int root = dsu.find(static_cast<int>(i));
        auto& component = component_clouds[root];
        if (!component) component.reset(new CloudT);
        *component += *atoms[i].patch.cloud;
    }

    std::vector<PlanePatch> patches;
    for (auto& [root, component] : component_clouds) {
        component->width = static_cast<std::uint32_t>(component->size());
        component->height = 1;
        component->is_dense = cloud->is_dense;
        if (component->size() < static_cast<std::size_t>(std::max(cfg.min_plane_points, 3))) continue;
        PlanePatch patch = fit_plane_patch(component, cfg);
        if (patch.planarity_error <= cfg.max_planarity_error) patches.push_back(std::move(patch));
    }
    return patches;
}

static std::vector<PlanePatch> extract_plane_patches_atoms(const CloudPtr& cloud, const Bounds& bounds, float resolution, const Config& cfg) {
    std::vector<PlanePatch> patches;
    for (float scale : cfg.atom_cell_scales) {
        auto scale_patches = extract_plane_patches_atoms_one_scale(cloud, bounds, resolution, scale, cfg);
        patches.insert(patches.end(), scale_patches.begin(), scale_patches.end());
    }
    return merge_coplanar_patches(std::move(patches), resolution, cfg);
}

static std::vector<PlanePatch> extract_plane_patches_region_growing(const CloudPtr& cloud, float resolution, const Config& cfg) {
    if (!cloud || cloud->size() < static_cast<std::size_t>(std::max(cfg.min_plane_points, 3))) return {};

    pcl::search::Search<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::NormalEstimation<PointT, pcl::Normal> normal_estimator;
    normal_estimator.setInputCloud(cloud);
    normal_estimator.setSearchMethod(tree);
    normal_estimator.setKSearch(std::max(cfg.normal_k, 3));
    normal_estimator.compute(*normals);

    pcl::RegionGrowing<PointT, pcl::Normal> region_growing;
    region_growing.setInputCloud(cloud);
    region_growing.setInputNormals(normals);
    region_growing.setSearchMethod(tree);
    region_growing.setMinClusterSize(std::max(cfg.region_min_cluster_size, 3));
    region_growing.setMaxClusterSize(static_cast<int>(std::min<std::size_t>(
        cloud->size(), static_cast<std::size_t>(std::numeric_limits<int>::max()))));
    region_growing.setNumberOfNeighbours(std::max(cfg.region_neighbors, 1));
    region_growing.setSmoothnessThreshold(cfg.region_smoothness_deg / 180.0f * 3.14159265358979323846f);
    region_growing.setCurvatureThreshold(std::max(cfg.region_curvature_threshold, 0.0f));

    std::vector<pcl::PointIndices> clusters;
    region_growing.extract(clusters);

    std::vector<PlanePatch> patches;
    patches.reserve(clusters.size());
    for (const auto& cluster : clusters) {
        if (cluster.indices.size() < static_cast<std::size_t>(std::max(cfg.min_plane_points, 3))) continue;

        CloudPtr cluster_cloud(new CloudT);
        cluster_cloud->reserve(cluster.indices.size());
        for (int idx : cluster.indices) {
            if (idx < 0 || static_cast<std::size_t>(idx) >= cloud->size()) continue;
            cluster_cloud->push_back(cloud->points[static_cast<std::size_t>(idx)]);
        }
        cluster_cloud->width = static_cast<std::uint32_t>(cluster_cloud->size());
        cluster_cloud->height = 1;
        cluster_cloud->is_dense = cloud->is_dense;
        if (cluster_cloud->size() < static_cast<std::size_t>(std::max(cfg.min_plane_points, 3))) continue;

        PlanePatch patch = fit_plane_patch(cluster_cloud, cfg);
        if (patch.planarity_error <= cfg.max_planarity_error) patches.push_back(std::move(patch));
    }

    std::sort(patches.begin(), patches.end(), [](const PlanePatch& a, const PlanePatch& b) {
        return a.cloud->size() > b.cloud->size();
    });
    if (patches.size() > static_cast<std::size_t>(std::max(cfg.max_planes, 1))) {
        patches.resize(static_cast<std::size_t>(std::max(cfg.max_planes, 1)));
    }
    if (cfg.verbose) {
        std::cout << "Region-growing clusters: " << clusters.size()
                  << ", planar seeds: " << patches.size() << "\n";
    }
    return merge_coplanar_patches(std::move(patches), resolution, cfg);
}

static void plane_basis(const PlanePatch& patch, Eigen::Vector3f& axis_u, Eigen::Vector3f& axis_v) {
    axis_u = patch.normal.cross(Eigen::Vector3f::UnitZ());
    if (axis_u.squaredNorm() < 1e-6f) axis_u = patch.normal.cross(Eigen::Vector3f::UnitX());
    axis_u.normalize();
    axis_v = patch.normal.cross(axis_u).normalized();
}

static PlaneEnvelope make_plane_envelope_from_cloud(const PlanePatch& seed_patch, const CloudPtr& envelope_cloud) {
    PlaneEnvelope envelope;
    envelope.seed = seed_patch;
    plane_basis(seed_patch, envelope.axis_u, envelope.axis_v);
    envelope.min_u = std::numeric_limits<float>::max();
    envelope.max_u = std::numeric_limits<float>::lowest();
    envelope.min_v = std::numeric_limits<float>::max();
    envelope.max_v = std::numeric_limits<float>::lowest();

    const CloudPtr source = envelope_cloud && !envelope_cloud->empty() ? envelope_cloud : seed_patch.cloud;
    for (const auto& p : source->points) {
        const Eigen::Vector3f q(p.x, p.y, p.z);
        const Eigen::Vector3f centered = q - seed_patch.centroid;
        const float u = centered.dot(envelope.axis_u);
        const float v = centered.dot(envelope.axis_v);
        envelope.min_u = std::min(envelope.min_u, u);
        envelope.max_u = std::max(envelope.max_u, u);
        envelope.min_v = std::min(envelope.min_v, v);
        envelope.max_v = std::max(envelope.max_v, v);
    }
    return envelope;
}

static PlaneEnvelope make_plane_envelope(const PlanePatch& seed_patch) {
    return make_plane_envelope_from_cloud(seed_patch, seed_patch.cloud);
}

static bool point_matches_envelope(const PlaneEnvelope& envelope, const Eigen::Vector3f& q, float distance_threshold, float margin, float& distance) {
    distance = std::abs(envelope.seed.normal.dot(q) + envelope.seed.d);
    if (distance > distance_threshold) return false;

    const Eigen::Vector3f centered = q - envelope.seed.centroid;
    const float u = centered.dot(envelope.axis_u);
    const float v = centered.dot(envelope.axis_v);
    if (u < envelope.min_u - margin || u > envelope.max_u + margin) return false;
    if (v < envelope.min_v - margin || v > envelope.max_v + margin) return false;
    return true;
}

static PlaneShapeMetrics compute_plane_shape_metrics(const PlanePatch& patch) {
    PlaneShapeMetrics metrics;
    if (patch.cloud->empty()) return metrics;

    Eigen::Vector3f axis_u;
    Eigen::Vector3f axis_v;
    plane_basis(patch, axis_u, axis_v);

    float min_u = std::numeric_limits<float>::max();
    float max_u = std::numeric_limits<float>::lowest();
    float min_v = std::numeric_limits<float>::max();
    float max_v = std::numeric_limits<float>::lowest();
    for (const auto& p : patch.cloud->points) {
        const Eigen::Vector3f q(p.x, p.y, p.z);
        const Eigen::Vector3f centered = q - patch.centroid;
        const float u = centered.dot(axis_u);
        const float v = centered.dot(axis_v);
        min_u = std::min(min_u, u);
        max_u = std::max(max_u, u);
        min_v = std::min(min_v, v);
        max_v = std::max(max_v, v);
    }

    metrics.extent_u = std::max(max_u - min_u, 0.0f);
    metrics.extent_v = std::max(max_v - min_v, 0.0f);
    metrics.short_extent = std::min(metrics.extent_u, metrics.extent_v);
    metrics.long_extent = std::max(metrics.extent_u, metrics.extent_v);
    metrics.aspect_ratio = metrics.long_extent > 1e-6f ? metrics.short_extent / metrics.long_extent : 0.0f;
    return metrics;
}

static bool is_valid_plane_patch_shape(const PlanePatch& patch, float resolution, const Config& cfg) {
    const PlaneShapeMetrics metrics = compute_plane_shape_metrics(patch);
    const float min_short_extent = std::max(cfg.min_plane_extent_scale * resolution, 0.0f);
    if (metrics.short_extent < min_short_extent) return false;
    if (metrics.aspect_ratio < cfg.min_plane_aspect_ratio) return false;
    return true;
}

static void finalize_plane_patches(std::vector<PlanePatch>& patches, float resolution, const Config& cfg) {
    patches = merge_coplanar_patches(std::move(patches), resolution, cfg);
    patches.erase(
        std::remove_if(patches.begin(), patches.end(), [&](const PlanePatch& patch) {
            return !is_valid_plane_patch_shape(patch, resolution, cfg);
        }),
        patches.end());
}

static void trim_support_cloud_conflicts(std::vector<PlanePatch>& patches, float resolution, const Config& cfg) {
    const float conflict_margin = std::max(cfg.support_conflict_scale * resolution, 0.0f);
    if (conflict_margin <= 0.0f) return;

    for (std::size_t i = 0; i < patches.size(); ++i) {
        CloudPtr support = visible_cloud_for_patch(patches[i]);
        if (!support || support->empty()) continue;

        CloudPtr trimmed(new CloudT);
        trimmed->reserve(support->size());
        for (const auto& p : support->points) {
            const Eigen::Vector3f q(p.x, p.y, p.z);
            const float self_distance = std::abs(patches[i].normal.dot(q) + patches[i].d);
            bool conflicted = false;
            for (std::size_t j = 0; j < patches.size(); ++j) {
                if (i == j) continue;
                if (std::abs(patches[i].normal.dot(patches[j].normal)) > cfg.normal_parallel_cos) continue;
                const float other_distance = std::abs(patches[j].normal.dot(q) + patches[j].d);
                if (other_distance + conflict_margin < self_distance) {
                    conflicted = true;
                    break;
                }
                if (std::abs(other_distance - self_distance) <= conflict_margin) {
                    conflicted = true;
                    break;
                }
            }
            if (!conflicted) trimmed->push_back(p);
        }
        trimmed->width = static_cast<std::uint32_t>(trimmed->size());
        trimmed->height = 1;
        trimmed->is_dense = support->is_dense;
        patches[i].support_cloud = trimmed;
    }
}

static void trim_core_cloud_conflicts(std::vector<PlanePatch>& patches, float resolution, const Config& cfg) {
    const float conflict_margin = std::max(cfg.core_conflict_scale * resolution, 0.0f);
    if (conflict_margin <= 0.0f) return;

    std::vector<PlanePatch> trimmed_patches;
    trimmed_patches.reserve(patches.size());
    for (std::size_t i = 0; i < patches.size(); ++i) {
        const auto& patch = patches[i];
        if (patch.cloud->empty()) continue;

        CloudPtr trimmed_core(new CloudT);
        trimmed_core->reserve(patch.cloud->size());
        for (const auto& p : patch.cloud->points) {
            const Eigen::Vector3f q(p.x, p.y, p.z);
            const float self_distance = std::abs(patch.normal.dot(q) + patch.d);
            bool conflicted = false;
            for (std::size_t j = 0; j < patches.size(); ++j) {
                if (i == j) continue;
                if (std::abs(patch.normal.dot(patches[j].normal)) > cfg.normal_parallel_cos) continue;
                const float other_distance = std::abs(patches[j].normal.dot(q) + patches[j].d);
                if (other_distance + conflict_margin < self_distance) {
                    conflicted = true;
                    break;
                }
                if (std::abs(other_distance - self_distance) <= conflict_margin) {
                    conflicted = true;
                    break;
                }
            }
            if (!conflicted) trimmed_core->push_back(p);
        }

        trimmed_core->width = static_cast<std::uint32_t>(trimmed_core->size());
        trimmed_core->height = 1;
        trimmed_core->is_dense = patch.cloud->is_dense;

        if (trimmed_core->size() < static_cast<std::size_t>(std::max(cfg.min_plane_points, 3))) {
            continue;
        }

        const std::size_t removed = patch.cloud->size() - trimmed_core->size();
        PlanePatch refit = removed > 0 ? fit_plane_patch(trimmed_core, cfg) : patch;
        if (removed > 0) {
            refit.support_cloud = patch.support_cloud;
            refit.output_cloud = patch.output_cloud;
        }
        if (!is_valid_plane_patch_shape(refit, resolution, cfg)) continue;
        trimmed_patches.push_back(std::move(refit));
    }

    patches = std::move(trimmed_patches);
}

static void filter_weak_visible_patches(std::vector<PlanePatch>& patches, const Config& cfg) {
    if (patches.empty()) return;

    std::vector<std::size_t> visible_sizes;
    visible_sizes.reserve(patches.size());
    std::vector<double> visible_densities;
    visible_densities.reserve(patches.size());
    for (const auto& patch : patches) {
        const std::size_t visible_points = visible_cloud_for_patch(patch)->size();
        const PlaneShapeMetrics metrics = compute_plane_shape_metrics(patch);
        const double projected_area = std::max<double>(
            static_cast<double>(metrics.extent_u) * static_cast<double>(metrics.extent_v), 1e-6);
        visible_sizes.push_back(visible_points);
        visible_densities.push_back(static_cast<double>(visible_points) / projected_area);
    }
    std::sort(visible_sizes.begin(), visible_sizes.end());
    std::sort(visible_densities.begin(), visible_densities.end());

    const double visible_support_median = static_cast<double>(visible_sizes[visible_sizes.size() / 2]);
    const std::size_t min_visible_support = static_cast<std::size_t>(
        std::max<double>(static_cast<double>(cfg.min_plane_points), visible_support_median * cfg.min_visible_support_median_ratio));
    const double min_visible_density =
        visible_densities[visible_densities.size() / 2] * static_cast<double>(cfg.min_visible_density_median_ratio);

    patches.erase(
        std::remove_if(patches.begin(), patches.end(), [&](const PlanePatch& patch) {
            const std::size_t core_points = patch.cloud->size();
            const std::size_t visible_points = visible_cloud_for_patch(patch)->size();
            if (core_points == 0 || visible_points == 0) return true;

            const PlaneShapeMetrics metrics = compute_plane_shape_metrics(patch);
            const double projected_area = std::max<double>(
                static_cast<double>(metrics.extent_u) * static_cast<double>(metrics.extent_v), 1e-6);
            const double visible_density = static_cast<double>(visible_points) / projected_area;
            const double visible_core_ratio = static_cast<double>(visible_points) / static_cast<double>(core_points);
            const bool weak_vs_core = visible_core_ratio < static_cast<double>(cfg.min_visible_support_core_ratio);
            const bool weak_vs_scene = visible_points < min_visible_support;
            const bool sparse_vs_scene = visible_density < min_visible_density;
            return weak_vs_scene && (weak_vs_core || sparse_vs_scene);
        }),
        patches.end());
}

static std::vector<PlanePatch> expand_patches_per_plane(const std::vector<PlanePatch>& seed_patches, const CloudPtr& full_cloud, float resolution, const Config& cfg) {
    std::vector<PlanePatch> patches;
    const float distance_threshold = std::max(cfg.expand_plane_distance_scale * resolution, 1e-4f);
    const float margin = std::max(cfg.expand_plane_margin_scale * resolution, distance_threshold);

    patches.reserve(seed_patches.size());
    for (std::size_t i = 0; i < seed_patches.size(); ++i) {
        PlaneEnvelope envelope = make_plane_envelope(seed_patches[i]);
        CloudPtr candidate(new CloudT);
        candidate->reserve(seed_patches[i].cloud->size());

        for (const auto& p : full_cloud->points) {
            const Eigen::Vector3f q(p.x, p.y, p.z);
            float distance = 0.0f;
            if (point_matches_envelope(envelope, q, distance_threshold, margin, distance)) {
                candidate->push_back(p);
            }
        }

        candidate->width = static_cast<std::uint32_t>(candidate->size());
        candidate->height = 1;
        candidate->is_dense = full_cloud->is_dense;
        if (candidate->size() >= static_cast<std::size_t>(std::max(cfg.min_plane_points, 3))) {
            patches.push_back(fit_strict_plane_patch(candidate, resolution, cfg));
        }
    }

    finalize_plane_patches(patches, resolution, cfg);
    return patches;
}

static void refresh_output_clouds_from_full_cloud(std::vector<PlanePatch>& patches, const CloudPtr& full_cloud, float resolution, const Config& cfg) {
    const float distance_threshold = std::max(cfg.expand_plane_distance_scale * resolution, 1e-4f);
    const float margin = std::max(cfg.expand_plane_margin_scale * resolution, distance_threshold);

    std::vector<PlaneEnvelope> envelopes;
    envelopes.reserve(patches.size());
    std::vector<std::size_t> patch_indices;
    patch_indices.reserve(patches.size());
    for (std::size_t i = 0; i < patches.size(); ++i) {
        const CloudPtr envelope_source = output_cloud_for_patch(patches[i]);
        if (!envelope_source || envelope_source->empty()) continue;
        envelopes.push_back(make_plane_envelope_from_cloud(patches[i], envelope_source));
        patch_indices.push_back(i);
    }
    if (envelopes.empty()) return;

    std::vector<CloudPtr> filled(patches.size());
    for (auto& cloud : filled) cloud.reset(new CloudT);

    for (const auto& p : full_cloud->points) {
        const Eigen::Vector3f q(p.x, p.y, p.z);
        std::size_t best_patch = patches.size();
        float best_distance = std::numeric_limits<float>::max();
        for (std::size_t k = 0; k < envelopes.size(); ++k) {
            float distance = 0.0f;
            if (point_matches_envelope(envelopes[k], q, distance_threshold, margin, distance) && distance < best_distance) {
                best_distance = distance;
                best_patch = patch_indices[k];
            }
        }
        if (best_patch < patches.size()) filled[best_patch]->push_back(p);
    }

    for (std::size_t i = 0; i < patches.size(); ++i) {
        auto& filled_cloud = filled[i];
        if (!filled_cloud || filled_cloud->empty()) continue;
        if (filled_cloud->size() >= static_cast<std::size_t>(std::max(cfg.min_plane_points, 3))) {
            filled_cloud->width = static_cast<std::uint32_t>(filled_cloud->size());
            filled_cloud->height = 1;
            filled_cloud->is_dense = full_cloud->is_dense;
            patches[i].output_cloud = filled_cloud;
        }
    }
}

static int nearby_support_count(pcl::KdTreeFLANN<PointT>& tree, const Eigen::Vector3f& p, float radius) {
    PointT probe;
    probe.x = p.x();
    probe.y = p.y();
    probe.z = p.z();
    std::vector<int> indices;
    std::vector<float> distances;
    return tree.radiusSearch(probe, radius, indices, distances);
}

class CornerEdgeScorer {
public:
    CornerEdgeScorer(pcl::KdTreeFLANN<PointT>& tree, CloudPtr cloud, float resolution, const Config& cfg)
        : tree_(tree), cloud_(std::move(cloud)), resolution_(resolution), cfg_(cfg) {}

    bool passes_filter(const CornerCandidate& candidate) const {
        const float edge_score = score_at_scale(candidate.position, cfg_.corner_edge_radius_scale);
        if (cfg_.max_corner_edge_score > 0.0f && edge_score > cfg_.max_corner_edge_score) return false;
        if (cfg_.max_corner_edge_growth_ratio > 0.0f) {
            const float mid = std::max(score_at_scale(candidate.position, edge_scale(1)), 1e-6f);
            const float large = score_at_scale(candidate.position, edge_scale(2));
            if (large / mid > cfg_.max_corner_edge_growth_ratio) return false;
        }
        return true;
    }

private:
    float edge_scale(std::size_t index) const {
        if (cfg_.corner_edge_multi_scales.empty()) return cfg_.corner_edge_radius_scale;
        return cfg_.corner_edge_multi_scales[std::min(index, cfg_.corner_edge_multi_scales.size() - 1)];
    }

    float score_at_scale(const Eigen::Vector3f& p, float scale) const {
        return local_covariance_score(p, std::max(scale * resolution_, 1e-4f));
    }

    float local_covariance_score(const Eigen::Vector3f& p, float radius) const {
        PointT probe;
        probe.x = p.x();
        probe.y = p.y();
        probe.z = p.z();
        std::vector<int> indices;
        std::vector<float> distances;
        const int found = tree_.radiusSearch(probe, radius, indices, distances);
        if (found < 6) return 0.0f;

        const std::size_t limit = cfg_.corner_edge_sample_limit > 0
            ? static_cast<std::size_t>(cfg_.corner_edge_sample_limit)
            : std::numeric_limits<std::size_t>::max();
        const std::size_t step = indices.size() > limit
            ? static_cast<std::size_t>(std::ceil(static_cast<double>(indices.size()) / static_cast<double>(limit)))
            : 1;

        Eigen::Vector3f mean = Eigen::Vector3f::Zero();
        std::size_t used = 0;
        for (std::size_t i = 0; i < indices.size() && used < limit; i += step) {
            const auto& q = cloud_->points[static_cast<std::size_t>(indices[i])];
            mean += Eigen::Vector3f(q.x, q.y, q.z);
            ++used;
        }
        if (used < 6) return 0.0f;
        mean /= static_cast<float>(used);

        Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
        std::size_t used_cov = 0;
        for (std::size_t i = 0; i < indices.size() && used_cov < limit; i += step) {
            const auto& q = cloud_->points[static_cast<std::size_t>(indices[i])];
            const Eigen::Vector3f centered = Eigen::Vector3f(q.x, q.y, q.z) - mean;
            covariance += centered * centered.transpose();
            ++used_cov;
        }
        covariance /= static_cast<float>(std::max<std::size_t>(used_cov, 1));

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
        if (solver.info() != Eigen::Success) return 0.0f;
        const Eigen::Vector3f eigenvalues = solver.eigenvalues().cwiseMax(0.0f);
        const float sum = eigenvalues.sum();
        return sum > 1e-12f ? eigenvalues.x() / sum : 0.0f;
    }

    pcl::KdTreeFLANN<PointT>& tree_;
    CloudPtr cloud_;
    float resolution_ = 0.0f;
    const Config& cfg_;
};

static int nearby_support_count_on_patch(const PlanePatch& patch, const Eigen::Vector3f& p, float radius) {
    const float radius_sq = radius * radius;
    int count = 0;
    for (const auto& q : patch.cloud->points) {
        const float dx = q.x - p.x();
        const float dy = q.y - p.y();
        const float dz = q.z - p.z();
        if (dx * dx + dy * dy + dz * dz <= radius_sq) ++count;
    }
    return count;
}

static bool intersection_of_three_planes(const PlanePatch& a, const PlanePatch& b, const PlanePatch& c, Eigen::Vector3f& out) {
    Eigen::Matrix3f A;
    A.row(0) = a.normal.transpose();
    A.row(1) = b.normal.transpose();
    A.row(2) = c.normal.transpose();
    if (std::abs(A.determinant()) < 1e-4f) return false;
    out = A.colPivHouseholderQr().solve(Eigen::Vector3f(-a.d, -b.d, -c.d));
    return out.allFinite();
}

static bool within_bounds(const Eigen::Vector3f& p, const Bounds& bounds, float margin) {
    return p.x() >= bounds.min_pt.x - margin && p.x() <= bounds.max_pt.x + margin &&
           p.y() >= bounds.min_pt.y - margin && p.y() <= bounds.max_pt.y + margin &&
           p.z() >= bounds.min_pt.z - margin && p.z() <= bounds.max_pt.z + margin;
}

static float cross_2d(const ProjectedPoint& o, const ProjectedPoint& a, const ProjectedPoint& b) {
    return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
}

static std::vector<int> convex_hull_indices(std::vector<ProjectedPoint> points) {
    if (points.size() < 3) {
        std::vector<int> indices;
        for (const auto& p : points) indices.push_back(p.index);
        return indices;
    }

    std::sort(points.begin(), points.end(), [](const ProjectedPoint& a, const ProjectedPoint& b) {
        if (a.x == b.x) return a.y < b.y;
        return a.x < b.x;
    });

    std::vector<ProjectedPoint> hull;
    for (const auto& p : points) {
        while (hull.size() >= 2 && cross_2d(hull[hull.size() - 2], hull.back(), p) <= 0.0f) hull.pop_back();
        hull.push_back(p);
    }
    const std::size_t lower_size = hull.size();
    for (auto it = points.rbegin() + 1; it != points.rend(); ++it) {
        while (hull.size() > lower_size && cross_2d(hull[hull.size() - 2], hull.back(), *it) <= 0.0f) hull.pop_back();
        hull.push_back(*it);
    }
    if (!hull.empty()) hull.pop_back();

    std::vector<int> indices;
    for (const auto& p : hull) indices.push_back(p.index);
    return indices;
}

static std::vector<CornerCandidate> merge_candidates(std::vector<CornerCandidate> candidates, float merge_radius) {
    std::vector<CornerCandidate> merged;
    const float merge_sq = merge_radius * merge_radius;
    for (auto& candidate : candidates) {
        bool absorbed = false;
        for (auto& group : merged) {
            if ((group.position - candidate.position).squaredNorm() <= merge_sq) {
                if (candidate.support > group.support) group.position = candidate.position;
                group.support = std::max(group.support, candidate.support);
                group.plane_ids.insert(group.plane_ids.end(), candidate.plane_ids.begin(), candidate.plane_ids.end());
                std::sort(group.plane_ids.begin(), group.plane_ids.end());
                group.plane_ids.erase(std::unique(group.plane_ids.begin(), group.plane_ids.end()), group.plane_ids.end());
                absorbed = true;
                break;
            }
        }
        if (!absorbed) merged.push_back(candidate);
    }
    std::sort(merged.begin(), merged.end(), [](const CornerCandidate& a, const CornerCandidate& b) {
        return a.support > b.support;
    });
    return merged;
}

static std::vector<CornerCandidate> extract_patch_boundary_corners(const std::vector<PlanePatch>& patches, const CloudPtr& support_cloud, pcl::KdTreeFLANN<PointT>& support_tree, float resolution, const Config& cfg) {
    std::vector<CornerCandidate> corners;
    const float min_edge = std::max(cfg.hull_min_edge_scale * resolution, 1e-4f);
    const float support_radius = std::max(cfg.corner_support_radius_scale * resolution, 1e-4f);

    for (std::size_t patch_id = 0; patch_id < patches.size(); ++patch_id) {
        const auto& patch = patches[patch_id];
        const float patch_fraction = static_cast<float>(patch.cloud->size()) / std::max(1.0f, static_cast<float>(support_cloud->size()));
        if (patch_fraction > cfg.hull_max_plane_fraction) continue;

        Eigen::Vector3f axis_u;
        Eigen::Vector3f axis_v;
        plane_basis(patch, axis_u, axis_v);

        std::vector<ProjectedPoint> projected;
        projected.reserve(patch.cloud->size());
        for (std::size_t i = 0; i < patch.cloud->size(); ++i) {
            const auto& p = patch.cloud->points[i];
            const Eigen::Vector3f q(p.x, p.y, p.z);
            const Eigen::Vector3f centered = q - patch.centroid;
            projected.push_back({centered.dot(axis_u), centered.dot(axis_v), static_cast<int>(i)});
        }

        const auto hull = convex_hull_indices(std::move(projected));
        if (hull.size() < 3) continue;

        for (std::size_t h = 0; h < hull.size(); ++h) {
            const auto& prev = patch.cloud->points[static_cast<std::size_t>(hull[(h + hull.size() - 1) % hull.size()])];
            const auto& cur = patch.cloud->points[static_cast<std::size_t>(hull[h])];
            const auto& next = patch.cloud->points[static_cast<std::size_t>(hull[(h + 1) % hull.size()])];
            const Eigen::Vector3f p_prev(prev.x, prev.y, prev.z);
            const Eigen::Vector3f p_cur(cur.x, cur.y, cur.z);
            const Eigen::Vector3f p_next(next.x, next.y, next.z);
            const Eigen::Vector3f a = p_prev - p_cur;
            const Eigen::Vector3f b = p_next - p_cur;
            const float len_a = a.norm();
            const float len_b = b.norm();
            if (len_a < min_edge || len_b < min_edge) continue;
            if (a.dot(b) / std::max(len_a * len_b, 1e-8f) > cfg.hull_corner_cos) continue;

            const int support = nearby_support_count(support_tree, p_cur, support_radius);
            if (support < cfg.min_corner_support) continue;

            int supporting_planes = 0;
            for (const auto& other_patch : patches) {
                if (nearby_support_count_on_patch(other_patch, p_cur, support_radius) >= cfg.min_plane_local_support) ++supporting_planes;
            }
            if (supporting_planes < std::max(1, cfg.boundary_min_plane_supports)) continue;

            corners.push_back({p_cur, {static_cast<int>(patch_id)}, support});
        }
    }
    return corners;
}

static std::vector<CornerCandidate> extract_corners(const CloudPtr& cloud, const std::vector<PlanePatch>& patches, const Bounds& bounds, float resolution, const Config& cfg) {
    std::vector<CornerCandidate> candidates;
    pcl::KdTreeFLANN<PointT> support_tree;
    support_tree.setInputCloud(cloud);
    const float support_radius = std::max(cfg.corner_support_radius_scale * resolution, 1e-4f);
    const float merge_radius = std::max(cfg.intersection_merge_scale * resolution, 1e-4f);
    const float bbox_margin = 3.0f * resolution;
    const float plane_distance_threshold = std::max(2.0f * resolution, 1e-4f);

    for (std::size_t i = 0; i < patches.size(); ++i) {
        for (std::size_t j = i + 1; j < patches.size(); ++j) {
            if (std::abs(patches[i].normal.dot(patches[j].normal)) > cfg.normal_parallel_cos) continue;
            for (std::size_t k = j + 1; k < patches.size(); ++k) {
                if (std::abs(patches[i].normal.dot(patches[k].normal)) > cfg.normal_parallel_cos) continue;
                if (std::abs(patches[j].normal.dot(patches[k].normal)) > cfg.normal_parallel_cos) continue;

                Eigen::Vector3f p;
                if (!intersection_of_three_planes(patches[i], patches[j], patches[k], p)) continue;
                if (!within_bounds(p, bounds, bbox_margin)) continue;
                if (std::abs(patches[i].normal.dot(p) + patches[i].d) > plane_distance_threshold) continue;
                if (std::abs(patches[j].normal.dot(p) + patches[j].d) > plane_distance_threshold) continue;
                if (std::abs(patches[k].normal.dot(p) + patches[k].d) > plane_distance_threshold) continue;

                const int support = nearby_support_count(support_tree, p, support_radius);
                if (support < cfg.min_corner_support) continue;
                if (nearby_support_count_on_patch(patches[i], p, support_radius) < cfg.min_plane_local_support) continue;
                if (nearby_support_count_on_patch(patches[j], p, support_radius) < cfg.min_plane_local_support) continue;
                if (nearby_support_count_on_patch(patches[k], p, support_radius) < cfg.min_plane_local_support) continue;
                candidates.push_back({p, {static_cast<int>(i), static_cast<int>(j), static_cast<int>(k)}, support});
            }
        }
    }

    auto boundary = extract_patch_boundary_corners(patches, cloud, support_tree, resolution, cfg);
    candidates.insert(candidates.end(), boundary.begin(), boundary.end());
    auto merged = merge_candidates(std::move(candidates), merge_radius);
    const CornerEdgeScorer edge_scorer(support_tree, cloud, resolution, cfg);
    const std::size_t min_planes = static_cast<std::size_t>(std::max(cfg.min_corner_planes, 1));
    merged.erase(
        std::remove_if(merged.begin(), merged.end(), [&](const CornerCandidate& candidate) {
            if (candidate.plane_ids.size() < min_planes) return true;
            return !edge_scorer.passes_filter(candidate);
        }),
        merged.end());
    return merged;
}

static void save_cloud(const fs::path& path, const ColorCloudPtr& cloud) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Failed to open output PLY: " + path.string());
    out << "ply\nformat ascii 1.0\ncomment Generated by pointcloud_corner_compare\n";
    out << "element vertex " << cloud->size() << "\n";
    out << "property float x\nproperty float y\nproperty float z\n";
    out << "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n";
    out.setf(std::ios::fixed, std::ios::floatfield);
    out.precision(6);
    for (const auto& p : cloud->points) {
        out << p.x << ' ' << p.y << ' ' << p.z << ' '
            << static_cast<int>(p.r) << ' '
            << static_cast<int>(p.g) << ' '
            << static_cast<int>(p.b) << '\n';
    }
}

static ColorCloudPtr make_corner_cloud(const std::vector<CornerCandidate>& corners) {
    ColorCloudPtr out(new ColorCloudT);
    for (const auto& corner : corners) {
        ColorPointT p;
        p.x = corner.position.x();
        p.y = corner.position.y();
        p.z = corner.position.z();
        p.r = 255;
        p.g = 64;
        p.b = 64;
        out->push_back(p);
    }
    out->width = static_cast<std::uint32_t>(out->size());
    out->height = 1;
    out->is_dense = true;
    return out;
}

static std::array<std::uint8_t, 3> unique_patch_color(std::size_t patch_id) {
    std::uint32_t x = static_cast<std::uint32_t>(patch_id + 1);
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return {
        static_cast<std::uint8_t>(80 + (x & 0x7fU)),
        static_cast<std::uint8_t>(80 + ((x >> 8) & 0x7fU)),
        static_cast<std::uint8_t>(80 + ((x >> 16) & 0x7fU))
    };
}

static ColorCloudPtr make_patch_cloud(const std::vector<PlanePatch>& patches) {
    ColorCloudPtr out(new ColorCloudT);
    for (std::size_t i = 0; i < patches.size(); ++i) {
        const auto color = unique_patch_color(i);
        const CloudPtr visible_cloud = output_cloud_for_patch(patches[i]);
        for (const auto& src : visible_cloud->points) {
            ColorPointT p;
            p.x = src.x;
            p.y = src.y;
            p.z = src.z;
            p.r = color[0];
            p.g = color[1];
            p.b = color[2];
            out->push_back(p);
        }
    }
    out->width = static_cast<std::uint32_t>(out->size());
    out->height = 1;
    out->is_dense = true;
    return out;
}

static void save_plane_report(const fs::path& path, const std::vector<PlanePatch>& patches) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Failed to open plane report: " + path.string());
    out << "id,core_points,visible_points,output_points,nx,ny,nz,d,cx,cy,cz,planarity_error,extent_u,extent_v,short_extent,long_extent,aspect_ratio\n";
    out.setf(std::ios::fixed, std::ios::floatfield);
    out.precision(6);
    for (std::size_t i = 0; i < patches.size(); ++i) {
        const auto& patch = patches[i];
        const PlaneShapeMetrics metrics = compute_plane_shape_metrics(patch);
        out << i << ','
            << patch.cloud->size() << ','
            << visible_cloud_for_patch(patch)->size() << ','
            << output_cloud_for_patch(patch)->size() << ','
            << patch.normal.x() << ','
            << patch.normal.y() << ','
            << patch.normal.z() << ','
            << patch.d << ','
            << patch.centroid.x() << ','
            << patch.centroid.y() << ','
            << patch.centroid.z() << ','
            << patch.planarity_error << ','
            << metrics.extent_u << ','
            << metrics.extent_v << ','
            << metrics.short_extent << ','
            << metrics.long_extent << ','
            << metrics.aspect_ratio << '\n';
    }
}

class CornerFeatureDetector {
public:
    explicit CornerFeatureDetector(Config cfg) : cfg_(std::move(cfg)) {}

    CornerDetectionResult detect(const CloudPtr& input_cloud) const {
        return detect_prepared(prepare_cloud(input_cloud, cfg_));
    }

    CornerDetectionResult detectFile(const fs::path& path) const {
        return detect_prepared(prepare_cloud(load_xyz_cloud(path, cfg_.input_scale), cfg_));
    }

private:
    CornerDetectionResult detect_prepared(PreparedCloud cloud) const {
        const Bounds bounds = compute_bounds(cloud.full_points);
        if (cfg_.verbose) {
            std::cout << "Detection points: " << cloud.points->size()
                      << ", full points: " << cloud.full_points->size()
                      << ", model resolution: " << cloud.model_resolution << "\n";
        }
        const auto seed_patches = cfg_.region_growing_plane_mode
            ? extract_plane_patches_region_growing(cloud.points, cloud.model_resolution, cfg_)
            : extract_plane_patches_atoms(cloud.points, bounds, cloud.model_resolution, cfg_);

        auto patches = expand_patches_per_plane(
            seed_patches, cloud.full_points, cloud.model_resolution, cfg_);
        trim_support_cloud_conflicts(patches, cloud.model_resolution, cfg_);
        trim_core_cloud_conflicts(patches, cloud.model_resolution, cfg_);
        trim_support_cloud_conflicts(patches, cloud.model_resolution, cfg_);
        filter_weak_visible_patches(patches, cfg_);
        refresh_output_clouds_from_full_cloud(patches, cloud.full_points, cloud.model_resolution, cfg_);

        CornerDetectionResult result;
        result.corners = extract_corners(cloud.full_points, patches, bounds, cloud.model_resolution, cfg_);
        result.plane_patches = std::move(patches);
        return result;
    }

    Config cfg_;
};

static void process_one(const fs::path& path, const Config& cfg, const CornerFeatureDetector& detector) {
    std::cout << "\n=== Processing " << path.string() << " ===\n";
    std::cout << "Detecting plane patches and corners...\n" << std::flush;
    const CornerDetectionResult result = detector.detectFile(path);
    std::cout << "Plane patches: " << result.plane_patches.size()
              << ", corners: " << result.corners.size() << "\n";

    const fs::path out_dir = cfg.output_dir / path.stem();
    fs::create_directories(out_dir);
    save_cloud(out_dir / "plane_patches.ply", make_patch_cloud(result.plane_patches));
    save_cloud(out_dir / "polyhedron_corners.ply", make_corner_cloud(result.corners));
    save_plane_report(out_dir / "plane_report.csv", result.plane_patches);
}

int main(int argc, char** argv) {
    try {
        const Config cfg = parse_args(argc, argv);
        const auto inputs = discover_inputs(cfg);
        if (inputs.empty()) {
            std::cerr << "No PLY files found.\n";
            return 1;
        }
        fs::create_directories(cfg.output_dir);
        const CornerFeatureDetector detector(cfg);
        for (const auto& input : inputs) {
            process_one(input, cfg, detector);
        }
        std::cout << "\nSaved results to " << fs::absolute(cfg.output_dir).string() << "\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}
