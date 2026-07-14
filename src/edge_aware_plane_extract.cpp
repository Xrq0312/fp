#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <queue>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include <pcl/common/common.h>
#include <pcl/conversions.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/ply_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>

namespace fs = std::filesystem;

using PointT = pcl::PointXYZ;
using CloudT = pcl::PointCloud<PointT>;
using CloudPtr = CloudT::Ptr;
using ColorPointT = pcl::PointXYZRGB;
using ColorCloudT = pcl::PointCloud<ColorPointT>;
using ColorCloudPtr = ColorCloudT::Ptr;

struct Config {
    fs::path input;
    fs::path output_dir = "results_edge_aware";
    float input_scale = 1000.0f;
    float surfel_size = 0.50f;
    int normal_k = 18;
    float smooth_angle_deg = 8.0f;
    float edge_angle_deg = 20.0f;
    float coplanar_distance = 0.35f;
    float merge_angle_deg = 5.0f;
    float merge_distance = 0.50f;
    float merge_gap = 8.0f;
    float assign_distance = 2.0f;
    float assign_margin = 2.0f;
    int grow_refine_iterations = 2;
    bool local_support_only = false;
    int support_halo = 1;
    float support_bin_size = 0.0f;
    int min_surfels = 20;
    int min_output_points = 5000;
    float min_extent = 3.0f;
    float min_aspect_ratio = 0.02f;
    int max_planes = 160;
    float normal_parallel_cos = 0.975f;
    float corner_support_radius_scale = 7.0f;
    float intersection_merge_scale = 4.5f;
    int min_corner_support = 8;
    int min_plane_local_support = 3;
    int min_corner_planes = 3;
    float corner_edge_radius_scale = 3.5f;
    std::vector<float> corner_edge_multi_scales{2.5f, 4.5f, 7.0f};
    int corner_edge_sample_limit = 40000;
    float max_corner_edge_score = 0.0f;
    float max_corner_edge_growth_ratio = 0.0f;
    bool save_plane_files = false;
    bool verbose = true;
};

struct Bounds {
    PointT min_pt;
    PointT max_pt;
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

struct Accumulator {
    Eigen::Vector3d sum{0.0, 0.0, 0.0};
    int count = 0;
};

struct Surfel {
    CellKey key;
    Eigen::Vector3f centroid{0.0f, 0.0f, 0.0f};
    Eigen::Vector3f normal{0.0f, 0.0f, 1.0f};
    float d = 0.0f;
    float curvature = 0.0f;
    int count = 0;
    bool edge_like = false;
};

struct PlanePatch {
    Eigen::Vector3f normal{0.0f, 0.0f, 1.0f};
    float d = 0.0f;
    Eigen::Vector3f centroid{0.0f, 0.0f, 0.0f};
    std::vector<int> surfels;
    CloudPtr output{new CloudT};
    float min_u = 0.0f;
    float max_u = 0.0f;
    float min_v = 0.0f;
    float max_v = 0.0f;
    float extent_u = 0.0f;
    float extent_v = 0.0f;
};

struct CornerCandidate {
    Eigen::Vector3f position{0.0f, 0.0f, 0.0f};
    std::vector<int> plane_ids;
    int support = 0;
};

static bool has_flag(const std::vector<std::string>& args, const std::string& flag) {
    return std::find(args.begin(), args.end(), flag) != args.end();
}

static std::optional<std::string> value_after(const std::vector<std::string>& args, const std::string& key) {
    for (std::size_t i = 0; i + 1 < args.size(); ++i) {
        if (args[i] == key) return args[i + 1];
    }
    return std::nullopt;
}

static Config parse_args(int argc, char** argv) {
    Config cfg;
    std::vector<std::string> args(argv + 1, argv + argc);
    if (args.empty() || has_flag(args, "--help") || has_flag(args, "-h")) {
        std::cout
            << "Usage: edge_aware_plane_extract input.ply [options]\n"
            << "Options: --output-dir, --input-scale, --surfel-size, --normal-k,\n"
            << "  --smooth-angle-deg, --edge-angle-deg, --coplanar-distance,\n"
            << "  --merge-*, --assign-*, --local-support-only,\n"
            << "  --support-halo, --support-bin-size, --save-plane-files,\n"
            << "  --min-surfels, --min-output-points, --corner-*.\n";
        std::exit(args.empty() ? 1 : 0);
    }

    for (const auto& arg : args) {
        if (!arg.empty() && arg.front() != '-' && cfg.input.empty()) cfg.input = arg;
    }
    if (cfg.input.empty()) throw std::runtime_error("Missing input PLY path.");

    if (auto v = value_after(args, "--output-dir")) cfg.output_dir = *v;
    if (auto v = value_after(args, "--input-scale")) cfg.input_scale = std::stof(*v);
    if (auto v = value_after(args, "--surfel-size")) cfg.surfel_size = std::stof(*v);
    if (auto v = value_after(args, "--normal-k")) cfg.normal_k = std::stoi(*v);
    if (auto v = value_after(args, "--smooth-angle-deg")) cfg.smooth_angle_deg = std::stof(*v);
    if (auto v = value_after(args, "--edge-angle-deg")) cfg.edge_angle_deg = std::stof(*v);
    if (auto v = value_after(args, "--coplanar-distance")) cfg.coplanar_distance = std::stof(*v);
    if (auto v = value_after(args, "--merge-angle-deg")) cfg.merge_angle_deg = std::stof(*v);
    if (auto v = value_after(args, "--merge-distance")) cfg.merge_distance = std::stof(*v);
    if (auto v = value_after(args, "--merge-gap")) cfg.merge_gap = std::stof(*v);
    if (auto v = value_after(args, "--assign-distance")) cfg.assign_distance = std::stof(*v);
    if (auto v = value_after(args, "--assign-margin")) cfg.assign_margin = std::stof(*v);
    if (auto v = value_after(args, "--grow-refine-iterations")) cfg.grow_refine_iterations = std::stoi(*v);
    cfg.local_support_only = has_flag(args, "--local-support-only");
    if (auto v = value_after(args, "--support-halo")) cfg.support_halo = std::stoi(*v);
    if (auto v = value_after(args, "--support-bin-size")) cfg.support_bin_size = std::stof(*v);
    if (auto v = value_after(args, "--min-surfels")) cfg.min_surfels = std::stoi(*v);
    if (auto v = value_after(args, "--min-output-points")) cfg.min_output_points = std::stoi(*v);
    if (auto v = value_after(args, "--min-extent")) cfg.min_extent = std::stof(*v);
    if (auto v = value_after(args, "--min-aspect-ratio")) cfg.min_aspect_ratio = std::stof(*v);
    if (auto v = value_after(args, "--max-planes")) cfg.max_planes = std::stoi(*v);
    if (auto v = value_after(args, "--normal-parallel-cos")) cfg.normal_parallel_cos = std::stof(*v);
    if (auto v = value_after(args, "--corner-support-radius-scale")) cfg.corner_support_radius_scale = std::stof(*v);
    if (auto v = value_after(args, "--intersection-merge-scale")) cfg.intersection_merge_scale = std::stof(*v);
    if (auto v = value_after(args, "--min-corner-support")) cfg.min_corner_support = std::stoi(*v);
    if (auto v = value_after(args, "--min-plane-local-support")) cfg.min_plane_local_support = std::stoi(*v);
    if (auto v = value_after(args, "--min-corner-planes")) cfg.min_corner_planes = std::stoi(*v);
    if (auto v = value_after(args, "--corner-edge-radius-scale")) cfg.corner_edge_radius_scale = std::stof(*v);
    if (auto v = value_after(args, "--corner-edge-sample-limit")) cfg.corner_edge_sample_limit = std::stoi(*v);
    if (auto v = value_after(args, "--max-corner-edge-score")) cfg.max_corner_edge_score = std::stof(*v);
    if (auto v = value_after(args, "--max-corner-edge-growth-ratio")) cfg.max_corner_edge_growth_ratio = std::stof(*v);
    cfg.save_plane_files = has_flag(args, "--save-plane-files");
    cfg.verbose = !has_flag(args, "--quiet");
    return cfg;
}

static CloudPtr load_cloud(const fs::path& path, float scale) {
    pcl::PCLPointCloud2 blob;
    if (pcl::io::loadPLYFile(path.string(), blob) != 0) {
        throw std::runtime_error("Failed to load PLY: " + path.string());
    }
    CloudPtr cloud(new CloudT);
    pcl::fromPCLPointCloud2(blob, *cloud);
    if (scale != 1.0f) {
        for (auto& p : cloud->points) {
            p.x *= scale;
            p.y *= scale;
            p.z *= scale;
        }
    }
    return cloud;
}

static Bounds compute_bounds(const CloudPtr& cloud) {
    Bounds bounds;
    pcl::getMinMax3D(*cloud, bounds.min_pt, bounds.max_pt);
    return bounds;
}

static CellKey point_key(const PointT& p, const Bounds& bounds, float size) {
    return {
        static_cast<int>(std::floor((p.x - bounds.min_pt.x) / size)),
        static_cast<int>(std::floor((p.y - bounds.min_pt.y) / size)),
        static_cast<int>(std::floor((p.z - bounds.min_pt.z) / size))
    };
}

static std::vector<Surfel> build_surfels(
    const CloudPtr& cloud,
    const Bounds& bounds,
    const Config& cfg,
    std::unordered_map<CellKey, int, CellKeyHash>& index_by_key,
    CloudPtr& surfel_cloud) {
    std::unordered_map<CellKey, Accumulator, CellKeyHash> cells;
    cells.reserve(cloud->size() / 20);
    for (const auto& p : cloud->points) {
        auto& acc = cells[point_key(p, bounds, cfg.surfel_size)];
        acc.sum += Eigen::Vector3d(p.x, p.y, p.z);
        ++acc.count;
    }

    std::vector<Surfel> surfels;
    surfels.reserve(cells.size());
    surfel_cloud.reset(new CloudT);
    surfel_cloud->reserve(cells.size());

    for (const auto& [key, acc] : cells) {
        if (acc.count < 3) continue;
        Surfel surfel;
        surfel.key = key;
        surfel.count = acc.count;
        const Eigen::Vector3d centroid = acc.sum / static_cast<double>(acc.count);
        surfel.centroid = centroid.cast<float>();
        index_by_key[surfel.key] = static_cast<int>(surfels.size());
        surfels.push_back(surfel);

        PointT q;
        q.x = static_cast<float>(centroid.x());
        q.y = static_cast<float>(centroid.y());
        q.z = static_cast<float>(centroid.z());
        surfel_cloud->push_back(q);
    }
    surfel_cloud->width = static_cast<std::uint32_t>(surfel_cloud->size());
    surfel_cloud->height = 1;
    surfel_cloud->is_dense = true;
    return surfels;
}

static void estimate_surfel_normals(std::vector<Surfel>& surfels, const CloudPtr& surfel_cloud, const Config& cfg) {
    pcl::search::Search<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::NormalEstimation<PointT, pcl::Normal> ne;
    ne.setInputCloud(surfel_cloud);
    ne.setSearchMethod(tree);
    ne.setKSearch(std::max(cfg.normal_k, 3));
    ne.compute(*normals);

    for (std::size_t i = 0; i < surfels.size(); ++i) {
        const auto& n = normals->points[i];
        Eigen::Vector3f normal(n.normal_x, n.normal_y, n.normal_z);
        if (!std::isfinite(normal.x()) || normal.squaredNorm() < 1e-8f) normal = Eigen::Vector3f::UnitZ();
        normal.normalize();
        surfels[i].normal = normal;
        surfels[i].d = -normal.dot(surfels[i].centroid);
        surfels[i].curvature = std::isfinite(n.curvature) ? n.curvature : 1.0f;
    }
}

static void mark_edge_surfels(
    std::vector<Surfel>& surfels,
    const std::unordered_map<CellKey, int, CellKeyHash>& index_by_key,
    const Config& cfg) {
    const float edge_cos = std::cos(cfg.edge_angle_deg / 180.0f * 3.14159265358979323846f);
    for (std::size_t i = 0; i < surfels.size(); ++i) {
        const CellKey key = surfels[i].key;
        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dz = -1; dz <= 1; ++dz) {
                    if (dx == 0 && dy == 0 && dz == 0) continue;
                    CellKey nk{key.x + dx, key.y + dy, key.z + dz};
                    auto it = index_by_key.find(nk);
                    if (it == index_by_key.end()) continue;
                    const int j = it->second;
                    if (j <= static_cast<int>(i)) continue;

                    const bool angle_edge = std::abs(surfels[i].normal.dot(surfels[static_cast<std::size_t>(j)].normal)) < edge_cos;
                    if (angle_edge) {
                        surfels[i].edge_like = true;
                        surfels[static_cast<std::size_t>(j)].edge_like = true;
                    }
                }
            }
        }
    }
}

static void plane_basis(const PlanePatch& patch, Eigen::Vector3f& axis_u, Eigen::Vector3f& axis_v) {
    axis_u = patch.normal.cross(Eigen::Vector3f::UnitZ());
    if (axis_u.squaredNorm() < 1e-6f) axis_u = patch.normal.cross(Eigen::Vector3f::UnitX());
    axis_u.normalize();
    axis_v = patch.normal.cross(axis_u).normalized();
}

static PlanePatch fit_patch_from_surfels(const std::vector<int>& ids, const std::vector<Surfel>& surfels) {
    PlanePatch patch;
    patch.surfels = ids;
    double total_weight = 0.0;
    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
    for (int id : ids) {
        const double w = static_cast<double>(std::max(surfels[static_cast<std::size_t>(id)].count, 1));
        centroid += w * surfels[static_cast<std::size_t>(id)].centroid.cast<double>();
        total_weight += w;
    }
    centroid /= std::max(total_weight, 1.0);
    patch.centroid = centroid.cast<float>();

    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    for (int id : ids) {
        const auto& surfel = surfels[static_cast<std::size_t>(id)];
        const Eigen::Vector3d q = surfel.centroid.cast<double>() - centroid;
        cov += static_cast<double>(std::max(surfel.count, 1)) * (q * q.transpose());
    }
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov);
    Eigen::Vector3d n = solver.eigenvectors().col(0);
    if (n.z() < 0.0) n = -n;
    patch.normal = n.cast<float>().normalized();
    patch.d = -patch.normal.dot(patch.centroid);
    return patch;
}

static void update_patch_envelope(PlanePatch& patch, const std::vector<Surfel>& surfels) {
    Eigen::Vector3f axis_u;
    Eigen::Vector3f axis_v;
    plane_basis(patch, axis_u, axis_v);
    patch.min_u = std::numeric_limits<float>::max();
    patch.max_u = std::numeric_limits<float>::lowest();
    patch.min_v = std::numeric_limits<float>::max();
    patch.max_v = std::numeric_limits<float>::lowest();
    for (int id : patch.surfels) {
        const Eigen::Vector3f q = surfels[static_cast<std::size_t>(id)].centroid - patch.centroid;
        const float u = q.dot(axis_u);
        const float v = q.dot(axis_v);
        patch.min_u = std::min(patch.min_u, u);
        patch.max_u = std::max(patch.max_u, u);
        patch.min_v = std::min(patch.min_v, v);
        patch.max_v = std::max(patch.max_v, v);
    }
    patch.extent_u = std::max(patch.max_u - patch.min_u, 0.0f);
    patch.extent_v = std::max(patch.max_v - patch.min_v, 0.0f);
}

static bool valid_shape(const PlanePatch& patch, const Config& cfg) {
    const float short_extent = std::min(patch.extent_u, patch.extent_v);
    const float long_extent = std::max(patch.extent_u, patch.extent_v);
    const float aspect = long_extent > 1e-6f ? short_extent / long_extent : 0.0f;
    return short_extent >= cfg.min_extent && aspect >= cfg.min_aspect_ratio;
}

static std::vector<int> surfel_neighbors(
    int surfel_id,
    const std::vector<Surfel>& surfels,
    const std::unordered_map<CellKey, int, CellKeyHash>& index_by_key) {
    std::vector<int> out;
    out.reserve(26);
    const CellKey base = surfels[static_cast<std::size_t>(surfel_id)].key;
    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dz = -1; dz <= 1; ++dz) {
                if (dx == 0 && dy == 0 && dz == 0) continue;
                const CellKey key{base.x + dx, base.y + dy, base.z + dz};
                auto it = index_by_key.find(key);
                if (it != index_by_key.end()) out.push_back(it->second);
            }
        }
    }
    return out;
}

static bool surfel_matches_plane(const Surfel& surfel, const PlanePatch& patch, const Config& cfg, float angle_cos, float distance_scale) {
    const float normal_dot = std::abs(surfel.normal.dot(patch.normal));
    if (normal_dot < angle_cos) return false;
    const float distance = std::abs(patch.normal.dot(surfel.centroid) + patch.d);
    return distance <= cfg.coplanar_distance * distance_scale;
}

static std::vector<PlanePatch> extract_seed_grown_patches(
    const std::vector<Surfel>& surfels,
    const std::unordered_map<CellKey, int, CellKeyHash>& index_by_key,
    const Config& cfg) {
    std::vector<int> order(surfels.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        const Surfel& lhs = surfels[static_cast<std::size_t>(a)];
        const Surfel& rhs = surfels[static_cast<std::size_t>(b)];
        if (lhs.edge_like != rhs.edge_like) return !lhs.edge_like;
        if (std::abs(lhs.curvature - rhs.curvature) > 1e-8f) return lhs.curvature < rhs.curvature;
        return lhs.count > rhs.count;
    });

    const int min_seed = std::max(6, cfg.min_surfels / 4);
    const int min_patch = std::max(cfg.min_surfels, 6);
    const float smooth_cos = std::cos(cfg.smooth_angle_deg / 180.0f * 3.14159265358979323846f);
    std::vector<int> owner(surfels.size(), -1);
    std::vector<PlanePatch> patches;

    for (int seed : order) {
        if (owner[static_cast<std::size_t>(seed)] >= 0 || surfels[static_cast<std::size_t>(seed)].edge_like) continue;

        std::vector<int> seed_ids{seed};
        for (int neighbor : surfel_neighbors(seed, surfels, index_by_key)) {
            if (owner[static_cast<std::size_t>(neighbor)] >= 0 || surfels[static_cast<std::size_t>(neighbor)].edge_like) continue;
            if (std::abs(surfels[static_cast<std::size_t>(seed)].normal.dot(surfels[static_cast<std::size_t>(neighbor)].normal)) >= smooth_cos) {
                seed_ids.push_back(neighbor);
            }
        }
        if (seed_ids.size() < static_cast<std::size_t>(min_seed)) continue;

        PlanePatch patch = fit_patch_from_surfels(seed_ids, surfels);
        std::vector<int> ids;
        ids.reserve(512);
        std::queue<int> q;
        std::set<int> touched;

        auto try_add = [&](int id, bool can_expand) {
            if (owner[static_cast<std::size_t>(id)] >= 0 || touched.find(id) != touched.end()) return;
            if (!surfel_matches_plane(surfels[static_cast<std::size_t>(id)], patch, cfg, smooth_cos, 1.0f)) return;
            touched.insert(id);
            ids.push_back(id);
            if (can_expand && !surfels[static_cast<std::size_t>(id)].edge_like) q.push(id);
        };

        for (int id : seed_ids) try_add(id, true);
        while (!q.empty()) {
            const int current = q.front();
            q.pop();
            for (int neighbor : surfel_neighbors(current, surfels, index_by_key)) {
                try_add(neighbor, true);
            }
        }

        for (int iter = 0; iter < std::max(cfg.grow_refine_iterations, 0); ++iter) {
            if (ids.size() < static_cast<std::size_t>(min_seed)) break;
            patch = fit_patch_from_surfels(ids, surfels);
            std::vector<int> frontier = ids;
            for (int current : frontier) {
                for (int neighbor : surfel_neighbors(current, surfels, index_by_key)) {
                    try_add(neighbor, !surfels[static_cast<std::size_t>(neighbor)].edge_like);
                }
            }
        }

        if (ids.size() < static_cast<std::size_t>(min_patch)) continue;
        patch = fit_patch_from_surfels(ids, surfels);
        update_patch_envelope(patch, surfels);
        if (!valid_shape(patch, cfg)) continue;

        const int patch_id = static_cast<int>(patches.size());
        for (int id : ids) owner[static_cast<std::size_t>(id)] = patch_id;
        patches.push_back(std::move(patch));
        if (patches.size() >= static_cast<std::size_t>(std::max(cfg.max_planes, 1))) break;
    }

    std::sort(patches.begin(), patches.end(), [](const PlanePatch& a, const PlanePatch& b) {
        return a.surfels.size() > b.surfels.size();
    });
    return patches;
}

static bool patches_can_merge(const PlanePatch& a, const PlanePatch& b, const Config& cfg) {
    const float normal_cos = std::cos(cfg.merge_angle_deg / 180.0f * 3.14159265358979323846f);
    const float dot = a.normal.dot(b.normal);
    if (std::abs(dot) < normal_cos) return false;
    const float b_d = dot >= 0.0f ? b.d : -b.d;
    if (std::abs(a.d - b_d) > cfg.merge_distance) return false;

    Eigen::Vector3f a_u;
    Eigen::Vector3f a_v;
    plane_basis(a, a_u, a_v);
    auto projected_range = [&](const PlanePatch& p) {
        Eigen::Vector3f p_u;
        Eigen::Vector3f p_v;
        plane_basis(p, p_u, p_v);
        std::array<float, 4> range{
            std::numeric_limits<float>::max(),
            std::numeric_limits<float>::lowest(),
            std::numeric_limits<float>::max(),
            std::numeric_limits<float>::lowest()
        };
        const std::array<Eigen::Vector2f, 4> corners{
            Eigen::Vector2f(p.min_u, p.min_v),
            Eigen::Vector2f(p.min_u, p.max_v),
            Eigen::Vector2f(p.max_u, p.min_v),
            Eigen::Vector2f(p.max_u, p.max_v)
        };
        for (const Eigen::Vector2f& corner : corners) {
            const Eigen::Vector3f q = p.centroid + p_u * corner.x() + p_v * corner.y();
            const Eigen::Vector3f centered = q - a.centroid;
            const float u = centered.dot(a_u);
            const float v = centered.dot(a_v);
            range[0] = std::min(range[0], u);
            range[1] = std::max(range[1], u);
            range[2] = std::min(range[2], v);
            range[3] = std::max(range[3], v);
        }
        return range;
    };

    const auto a_range = projected_range(a);
    const auto b_range = projected_range(b);
    const bool u_touch = !(a_range[1] + cfg.merge_gap < b_range[0] || b_range[1] + cfg.merge_gap < a_range[0]);
    const bool v_touch = !(a_range[3] + cfg.merge_gap < b_range[2] || b_range[3] + cfg.merge_gap < a_range[2]);
    return u_touch && v_touch;
}

static void merge_patches(std::vector<PlanePatch>& patches, const std::vector<Surfel>& surfels, const Config& cfg) {
    bool changed = true;
    while (changed) {
        changed = false;
        for (std::size_t i = 0; i < patches.size() && !changed; ++i) {
            for (std::size_t j = i + 1; j < patches.size(); ++j) {
                if (!patches_can_merge(patches[i], patches[j], cfg)) continue;
                patches[i].surfels.insert(patches[i].surfels.end(), patches[j].surfels.begin(), patches[j].surfels.end());
                patches[i] = fit_patch_from_surfels(patches[i].surfels, surfels);
                update_patch_envelope(patches[i], surfels);
                patches.erase(patches.begin() + static_cast<std::ptrdiff_t>(j));
                changed = true;
                break;
            }
        }
    }
    std::sort(patches.begin(), patches.end(), [](const PlanePatch& a, const PlanePatch& b) {
        return a.surfels.size() > b.surfels.size();
    });
}

static bool point_inside_patch_envelope(const PlanePatch& patch, const Eigen::Vector3f& q, float margin) {
    Eigen::Vector3f axis_u;
    Eigen::Vector3f axis_v;
    plane_basis(patch, axis_u, axis_v);
    const Eigen::Vector3f centered = q - patch.centroid;
    const float u = centered.dot(axis_u);
    const float v = centered.dot(axis_v);
    return u >= patch.min_u - margin && u <= patch.max_u + margin &&
           v >= patch.min_v - margin && v <= patch.max_v + margin;
}

struct BinKey {
    int u = 0;
    int v = 0;

    bool operator==(const BinKey& other) const {
        return u == other.u && v == other.v;
    }
};

struct BinKeyHash {
    std::size_t operator()(const BinKey& key) const {
        std::size_t h = static_cast<std::size_t>(key.u) * 73856093u;
        h ^= static_cast<std::size_t>(key.v) * 19349663u;
        return h;
    }
};

struct PatchSupport {
    Eigen::Vector3f origin{0.0f, 0.0f, 0.0f};
    Eigen::Vector3f axis_u{1.0f, 0.0f, 0.0f};
    Eigen::Vector3f axis_v{0.0f, 1.0f, 0.0f};
    float bin_size = 1.0f;
    std::unordered_set<BinKey, BinKeyHash> bins;

    BinKey key_for(const Eigen::Vector3f& q) const {
        const Eigen::Vector3f centered = q - origin;
        return {
            static_cast<int>(std::floor(centered.dot(axis_u) / bin_size)),
            static_cast<int>(std::floor(centered.dot(axis_v) / bin_size))
        };
    }

    bool contains(const Eigen::Vector3f& q) const {
        return bins.find(key_for(q)) != bins.end();
    }
};

static std::vector<PatchSupport> build_patch_supports(
    const std::vector<PlanePatch>& patches,
    const std::vector<Surfel>& surfels,
    const Config& cfg) {
    std::vector<PatchSupport> support(patches.size());
    const int halo = std::max(cfg.support_halo, 0);
    const int kernel = 2 * halo + 1;
    const std::size_t reserve_factor = static_cast<std::size_t>(kernel * kernel);
    const float bin_size = cfg.support_bin_size > 0.0f ? cfg.support_bin_size : cfg.surfel_size;
    for (std::size_t patch_id = 0; patch_id < patches.size(); ++patch_id) {
        PatchSupport& patch_support = support[patch_id];
        patch_support.origin = patches[patch_id].centroid;
        patch_support.bin_size = std::max(bin_size, 1e-4f);
        plane_basis(patches[patch_id], patch_support.axis_u, patch_support.axis_v);
        patch_support.bins.reserve(patches[patch_id].surfels.size() * reserve_factor);
        for (const int surfel_id : patches[patch_id].surfels) {
            const BinKey base = patch_support.key_for(surfels[static_cast<std::size_t>(surfel_id)].centroid);
            for (int du = -halo; du <= halo; ++du) {
                for (int dv = -halo; dv <= halo; ++dv) {
                    patch_support.bins.insert({base.u + du, base.v + dv});
                }
            }
        }
    }
    return support;
}

static void assign_full_points(
    const CloudPtr& cloud,
    const Bounds& bounds,
    const std::vector<Surfel>& surfels,
    std::vector<PlanePatch>& patches,
    const Config& cfg) {
    for (auto& patch : patches) patch.output.reset(new CloudT);
    const std::vector<PatchSupport> support = cfg.local_support_only ? build_patch_supports(patches, surfels, cfg) : std::vector<PatchSupport>{};

    for (const auto& p : cloud->points) {
        const Eigen::Vector3f q(p.x, p.y, p.z);
        (void)bounds;
        int best = -1;
        float best_distance = std::numeric_limits<float>::max();
        for (std::size_t i = 0; i < patches.size(); ++i) {
            if (cfg.local_support_only && !support[i].contains(q)) continue;
            const float distance = std::abs(patches[i].normal.dot(q) + patches[i].d);
            if (distance > cfg.assign_distance || distance >= best_distance) continue;
            if (!point_inside_patch_envelope(patches[i], q, cfg.assign_margin)) continue;
            best_distance = distance;
            best = static_cast<int>(i);
        }
        if (best >= 0) patches[static_cast<std::size_t>(best)].output->push_back(p);
    }

    patches.erase(
        std::remove_if(patches.begin(), patches.end(), [&](const PlanePatch& patch) {
            return patch.output->size() < static_cast<std::size_t>(std::max(cfg.min_output_points, 1));
        }),
        patches.end());

    for (auto& patch : patches) {
        patch.output->width = static_cast<std::uint32_t>(patch.output->size());
        patch.output->height = 1;
        patch.output->is_dense = true;
    }
}

static std::array<std::uint8_t, 3> color_for(std::size_t id) {
    std::uint32_t x = static_cast<std::uint32_t>(id + 1);
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

static ColorCloudPtr make_colored_patch_cloud(const std::vector<PlanePatch>& patches) {
    ColorCloudPtr out(new ColorCloudT);
    for (std::size_t i = 0; i < patches.size(); ++i) {
        const auto color = color_for(i);
        for (const auto& src : patches[i].output->points) {
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

static ColorCloudPtr make_edge_cloud(const std::vector<Surfel>& surfels) {
    ColorCloudPtr out(new ColorCloudT);
    for (const auto& surfel : surfels) {
        if (!surfel.edge_like) continue;
        ColorPointT p;
        p.x = surfel.centroid.x();
        p.y = surfel.centroid.y();
        p.z = surfel.centroid.z();
        p.r = 255;
        p.g = 40;
        p.b = 40;
        out->push_back(p);
    }
    out->width = static_cast<std::uint32_t>(out->size());
    out->height = 1;
    out->is_dense = true;
    return out;
}

static std::vector<int> build_surfel_labels(const std::vector<PlanePatch>& patches, std::size_t surfel_count) {
    std::vector<int> labels(surfel_count, -1);
    for (std::size_t i = 0; i < patches.size(); ++i) {
        for (int surfel_id : patches[i].surfels) {
            if (surfel_id >= 0 && static_cast<std::size_t>(surfel_id) < labels.size()) {
                labels[static_cast<std::size_t>(surfel_id)] = static_cast<int>(i);
            }
        }
    }
    return labels;
}

static ColorCloudPtr make_feature_cloud(
    const std::vector<Surfel>& surfels,
    const std::unordered_map<CellKey, int, CellKeyHash>& index_by_key,
    const std::vector<PlanePatch>& patches,
    bool corners) {
    const std::vector<int> labels = build_surfel_labels(patches, surfels.size());
    ColorCloudPtr out(new ColorCloudT);
    for (std::size_t i = 0; i < surfels.size(); ++i) {
        if (labels[i] < 0) continue;
        std::set<int> neighbor_labels{labels[i]};
        for (int neighbor : surfel_neighbors(static_cast<int>(i), surfels, index_by_key)) {
            const int label = labels[static_cast<std::size_t>(neighbor)];
            if (label >= 0) neighbor_labels.insert(label);
        }
        const std::size_t required = corners ? 3 : 2;
        if (neighbor_labels.size() < required) continue;
        ColorPointT p;
        p.x = surfels[i].centroid.x();
        p.y = surfels[i].centroid.y();
        p.z = surfels[i].centroid.z();
        p.r = corners ? 255 : 255;
        p.g = corners ? 255 : 160;
        p.b = corners ? 255 : 20;
        out->push_back(p);
    }
    out->width = static_cast<std::uint32_t>(out->size());
    out->height = 1;
    out->is_dense = true;
    return out;
}

static ColorCloudPtr make_plane_intersection_corners(const std::vector<PlanePatch>& patches, const Config& cfg) {
    ColorCloudPtr out(new ColorCloudT);
    const float det_threshold = 0.05f;
    const float merge_distance = std::max(cfg.surfel_size * 2.0f, 0.5f);
    const float envelope_margin = std::max(cfg.assign_margin, cfg.surfel_size * 4.0f);

    auto duplicate = [&](const Eigen::Vector3f& q) {
        for (const auto& p : out->points) {
            const Eigen::Vector3f existing(p.x, p.y, p.z);
            if ((existing - q).norm() <= merge_distance) return true;
        }
        return false;
    };

    for (std::size_t i = 0; i < patches.size(); ++i) {
        for (std::size_t j = i + 1; j < patches.size(); ++j) {
            for (std::size_t k = j + 1; k < patches.size(); ++k) {
                Eigen::Matrix3f a;
                a.row(0) = patches[i].normal.transpose();
                a.row(1) = patches[j].normal.transpose();
                a.row(2) = patches[k].normal.transpose();
                const float det = std::abs(a.determinant());
                if (det < det_threshold) continue;
                Eigen::Vector3f b(-patches[i].d, -patches[j].d, -patches[k].d);
                const Eigen::Vector3f q = a.colPivHouseholderQr().solve(b);
                if (!std::isfinite(q.x()) || !std::isfinite(q.y()) || !std::isfinite(q.z())) continue;
                if (!point_inside_patch_envelope(patches[i], q, envelope_margin)) continue;
                if (!point_inside_patch_envelope(patches[j], q, envelope_margin)) continue;
                if (!point_inside_patch_envelope(patches[k], q, envelope_margin)) continue;
                if (duplicate(q)) continue;
                ColorPointT p;
                p.x = q.x();
                p.y = q.y();
                p.z = q.z();
                p.r = 255;
                p.g = 255;
                p.b = 255;
                out->push_back(p);
            }
        }
    }
    out->width = static_cast<std::uint32_t>(out->size());
    out->height = 1;
    out->is_dense = true;
    return out;
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

static int nearby_support_count_on_patch(const PlanePatch& patch, const Eigen::Vector3f& p, float radius) {
    const float radius_sq = radius * radius;
    int count = 0;
    for (const auto& q : patch.output->points) {
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

static std::vector<CornerCandidate> merge_corner_candidates(std::vector<CornerCandidate> candidates, float merge_radius) {
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

static std::vector<CornerCandidate> extract_corners_filtered(
    const CloudPtr& cloud,
    const std::vector<PlanePatch>& patches,
    const Bounds& bounds,
    const Config& cfg) {
    std::vector<CornerCandidate> candidates;
    pcl::KdTreeFLANN<PointT> support_tree;
    support_tree.setInputCloud(cloud);

    const float support_radius = std::max(cfg.corner_support_radius_scale * cfg.surfel_size, 1e-4f);
    const float merge_radius = std::max(cfg.intersection_merge_scale * cfg.surfel_size, 1e-4f);
    const float bbox_margin = std::max(3.0f * cfg.surfel_size, cfg.assign_distance);
    const float envelope_margin = std::max(cfg.assign_margin, cfg.surfel_size * 2.0f);
    const float plane_distance_threshold = std::max(2.0f * cfg.surfel_size, cfg.assign_distance);

    for (std::size_t i = 0; i < patches.size(); ++i) {
        for (std::size_t j = i + 1; j < patches.size(); ++j) {
            if (std::abs(patches[i].normal.dot(patches[j].normal)) > cfg.normal_parallel_cos) continue;
            for (std::size_t k = j + 1; k < patches.size(); ++k) {
                if (std::abs(patches[i].normal.dot(patches[k].normal)) > cfg.normal_parallel_cos) continue;
                if (std::abs(patches[j].normal.dot(patches[k].normal)) > cfg.normal_parallel_cos) continue;

                Eigen::Vector3f p;
                if (!intersection_of_three_planes(patches[i], patches[j], patches[k], p)) continue;
                if (!within_bounds(p, bounds, bbox_margin)) continue;
                if (!point_inside_patch_envelope(patches[i], p, envelope_margin)) continue;
                if (!point_inside_patch_envelope(patches[j], p, envelope_margin)) continue;
                if (!point_inside_patch_envelope(patches[k], p, envelope_margin)) continue;
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

    auto merged = merge_corner_candidates(std::move(candidates), merge_radius);
    const CornerEdgeScorer edge_scorer(support_tree, cloud, cfg.surfel_size, cfg);
    const std::size_t min_planes = static_cast<std::size_t>(std::max(cfg.min_corner_planes, 1));
    merged.erase(
        std::remove_if(merged.begin(), merged.end(), [&](const CornerCandidate& candidate) {
            if (candidate.plane_ids.size() < min_planes) return true;
            return !edge_scorer.passes_filter(candidate);
        }),
        merged.end());
    return merged;
}

static ColorCloudPtr make_corner_cloud(const std::vector<CornerCandidate>& corners) {
    ColorCloudPtr out(new ColorCloudT);
    for (const auto& corner : corners) {
        ColorPointT p;
        p.x = corner.position.x();
        p.y = corner.position.y();
        p.z = corner.position.z();
        p.r = 255;
        p.g = 255;
        p.b = 255;
        out->push_back(p);
    }
    out->width = static_cast<std::uint32_t>(out->size());
    out->height = 1;
    out->is_dense = true;
    return out;
}

static void save_color_ply_allow_empty(const fs::path& path, const ColorCloudT& cloud) {
    if (!cloud.empty()) {
        pcl::io::savePLYFileASCII(path.string(), cloud);
        return;
    }
    std::ofstream out(path);
    out << "ply\n"
        << "format ascii 1.0\n"
        << "element vertex 0\n"
        << "property float x\n"
        << "property float y\n"
        << "property float z\n"
        << "property uchar red\n"
        << "property uchar green\n"
        << "property uchar blue\n"
        << "end_header\n";
}

static void save_report(const fs::path& path, const std::vector<PlanePatch>& patches) {
    std::ofstream out(path);
    out << "id,surfels,output_points,nx,ny,nz,d,cx,cy,cz,extent_u,extent_v\n";
    out.setf(std::ios::fixed, std::ios::floatfield);
    out.precision(6);
    for (std::size_t i = 0; i < patches.size(); ++i) {
        const auto& p = patches[i];
        out << i << ','
            << p.surfels.size() << ','
            << p.output->size() << ','
            << p.normal.x() << ','
            << p.normal.y() << ','
            << p.normal.z() << ','
            << p.d << ','
            << p.centroid.x() << ','
            << p.centroid.y() << ','
            << p.centroid.z() << ','
            << p.extent_u << ','
            << p.extent_v << '\n';
    }
}

static void save_corner_report(const fs::path& path, const std::vector<CornerCandidate>& corners) {
    std::ofstream out(path);
    out << "id,x,y,z,support,plane_count,plane_ids\n";
    out.setf(std::ios::fixed, std::ios::floatfield);
    out.precision(6);
    for (std::size_t i = 0; i < corners.size(); ++i) {
        out << i << ','
            << corners[i].position.x() << ','
            << corners[i].position.y() << ','
            << corners[i].position.z() << ','
            << corners[i].support << ','
            << corners[i].plane_ids.size() << ',';
        for (std::size_t j = 0; j < corners[i].plane_ids.size(); ++j) {
            if (j > 0) out << '|';
            out << corners[i].plane_ids[j];
        }
        out << '\n';
    }
}

static void save_plane_files(const fs::path& out_dir, const std::vector<PlanePatch>& patches) {
    const fs::path plane_dir = out_dir / "planes";
    fs::create_directories(plane_dir);
    for (std::size_t i = 0; i < patches.size(); ++i) {
        std::ostringstream name;
        name << "plane_" << std::setw(3) << std::setfill('0') << i << "_"
             << patches[i].output->size() << "pts.ply";
        pcl::io::savePLYFileASCII((plane_dir / name.str()).string(), *patches[i].output);
    }
}

int main(int argc, char** argv) {
    try {
        const Config cfg = parse_args(argc, argv);
        CloudPtr cloud = load_cloud(cfg.input, cfg.input_scale);
        const Bounds bounds = compute_bounds(cloud);

        std::unordered_map<CellKey, int, CellKeyHash> index_by_key;
        CloudPtr surfel_cloud;
        std::vector<Surfel> surfels = build_surfels(cloud, bounds, cfg, index_by_key, surfel_cloud);
        estimate_surfel_normals(surfels, surfel_cloud, cfg);
        mark_edge_surfels(surfels, index_by_key, cfg);
        std::vector<PlanePatch> patches = extract_seed_grown_patches(surfels, index_by_key, cfg);
        merge_patches(patches, surfels, cfg);
        assign_full_points(cloud, bounds, surfels, patches, cfg);
        const std::vector<CornerCandidate> corners = extract_corners_filtered(cloud, patches, bounds, cfg);

        fs::path out_dir = cfg.output_dir / cfg.input.stem();
        fs::create_directories(out_dir);
        pcl::io::savePLYFileASCII((out_dir / "edge_aware_plane_patches.ply").string(), *make_colored_patch_cloud(patches));
        pcl::io::savePLYFileASCII((out_dir / "edge_surfels.ply").string(), *make_edge_cloud(surfels));
        save_color_ply_allow_empty(out_dir / "feature_edges.ply", *make_feature_cloud(surfels, index_by_key, patches, false));
        save_color_ply_allow_empty(out_dir / "feature_corners.ply", *make_plane_intersection_corners(patches, cfg));
        save_color_ply_allow_empty(out_dir / "extracted_corners.ply", *make_corner_cloud(corners));
        save_report(out_dir / "edge_aware_plane_report.csv", patches);
        save_corner_report(out_dir / "corner_report.csv", corners);
        if (cfg.save_plane_files) save_plane_files(out_dir, patches);

        if (cfg.verbose) {
            std::size_t output_points = 0;
            for (const auto& patch : patches) output_points += patch.output->size();
            std::cout << "Input points: " << cloud->size() << "\n";
            std::cout << "Surfels: " << surfels.size() << "\n";
            std::cout << "Plane patches: " << patches.size() << "\n";
            std::cout << "Output points: " << output_points << "\n";
            std::cout << "Extracted corners: " << corners.size() << "\n";
            std::cout << "Saved to " << fs::absolute(out_dir).string() << "\n";
        }
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}
