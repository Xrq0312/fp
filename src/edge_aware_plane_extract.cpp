#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <set>
#include <string>
#include <unordered_map>
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

struct Config {
    fs::path input;
    fs::path output_dir = "results_edge_aware";
    float input_scale = 1.0f;
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
    int min_surfels = 20;
    int min_output_points = 5000;
    float min_extent = 3.0f;
    float min_aspect_ratio = 0.02f;
    int max_planes = 160;
    float corner_support_radius_scale = 7.0f;
    float intersection_merge_scale = 4.5f;
    int min_corner_support = 8;
    int min_plane_local_support = 3;
    bool has_probe = false;
    Eigen::Vector3f probe{0.0f, 0.0f, 0.0f};
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

struct Surfel {
    CellKey key;
    Eigen::Vector3f centroid{0.0f, 0.0f, 0.0f};
    Eigen::Vector3f normal{0.0f, 0.0f, 1.0f};
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
    float rms_distance = 0.0f;
};

struct CompetitionStats {
    std::size_t moved_surfels = 0;
};

struct CornerCandidate {
    Eigen::Vector3f position{0.0f, 0.0f, 0.0f};
    int support = 0;
};

static float scale_for_profile(const std::string& profile) {
    if (profile == "model-mm" || profile == "top-down") {
        return 1.0f;
    }
    if (profile == "model-m") {
        return 1000.0f;
    }
    if (profile == "top-left-cut2") {
        return 1000.0f;
    }
    throw std::runtime_error("Unknown profile: " + profile + ". Use model-mm, model-m, top-down, or top-left-cut2.");
}

static Config parse_args(int argc, char** argv) {
    Config cfg;
    std::vector<std::string> args(argv + 1, argv + argc);
    const auto has_arg = [&](const std::string& flag) {
        return std::find(args.begin(), args.end(), flag) != args.end();
    };
    if (args.empty() || has_arg("--help") || has_arg("-h")) {
        std::cout
            << "Usage: edge_aware_plane_extract input.ply [options]\n"
            << "Options:\n"
            << "  --profile model-mm|model-m|top-down|top-left-cut2\n"
            << "  --output-dir DIR\n"
            << "  --probe X Y Z  print plane-assignment diagnostics at a point\n"
            << "  --quiet\n";
        std::exit(args.empty() ? 1 : 0);
    }

    cfg.input_scale = scale_for_profile("model-mm");
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string& arg = args[i];
        if (arg == "--profile") {
            if (i + 1 >= args.size()) throw std::runtime_error("Missing value after --profile.");
            cfg.input_scale = scale_for_profile(args[++i]);
        } else if (arg == "--output-dir") {
            if (i + 1 >= args.size()) throw std::runtime_error("Missing value after --output-dir.");
            cfg.output_dir = args[++i];
        } else if (arg == "--probe") {
            if (i + 3 >= args.size()) throw std::runtime_error("--probe requires X Y Z.");
            const float x = std::stof(args[++i]);
            const float y = std::stof(args[++i]);
            const float z = std::stof(args[++i]);
            cfg.probe = Eigen::Vector3f(x, y, z);
            cfg.has_probe = true;
        } else if (arg == "--quiet") {
            cfg.verbose = false;
        } else if (!arg.empty() && arg.front() == '-') {
            throw std::runtime_error("Unknown option: " + arg + ". Use --profile instead of tuning internal thresholds.");
        } else if (cfg.input.empty()) {
            cfg.input = arg;
        } else {
            throw std::runtime_error("Unexpected positional argument: " + arg);
        }
    }
    if (cfg.input.empty()) throw std::runtime_error("Missing input PLY path.");
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
    struct Accumulator {
        Eigen::Vector3d sum{0.0, 0.0, 0.0};
        int count = 0;
    };

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
    double squared_error = 0.0;
    for (int id : ids) {
        const auto& surfel = surfels[static_cast<std::size_t>(id)];
        const double distance = static_cast<double>(patch.normal.dot(surfel.centroid) + patch.d);
        squared_error += static_cast<double>(std::max(surfel.count, 1)) * distance * distance;
    }
    patch.rms_distance = static_cast<float>(std::sqrt(squared_error / std::max(total_weight, 1.0)));
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

static std::vector<int> build_surfel_labels(const std::vector<PlanePatch>& patches, std::size_t surfel_count) {
    std::vector<int> labels(surfel_count, -1);
    for (std::size_t patch_id = 0; patch_id < patches.size(); ++patch_id) {
        for (int surfel_id : patches[patch_id].surfels) {
            if (surfel_id >= 0 && static_cast<std::size_t>(surfel_id) < labels.size()) {
                labels[static_cast<std::size_t>(surfel_id)] = static_cast<int>(patch_id);
            }
        }
    }
    return labels;
}

static float surfel_plane_score(const Surfel& surfel, const PlanePatch& patch, const Config& cfg) {
    const float distance_scale = std::max(cfg.coplanar_distance, 0.5f * cfg.surfel_size);
    const float distance = std::abs(patch.normal.dot(surfel.centroid) + patch.d);
    const float normal_dot = std::clamp(std::abs(surfel.normal.dot(patch.normal)), 0.0f, 1.0f);
    const float angle_scale = std::max(
        std::sin(cfg.smooth_angle_deg / 180.0f * 3.14159265358979323846f),
        1e-3f);
    const float angular_residual = std::sqrt(std::max(0.0f, 1.0f - normal_dot * normal_dot)) / angle_scale;
    const float angle_weight = surfel.edge_like ? 0.25f : 1.0f;
    const float distance_residual = distance / distance_scale;
    return distance_residual * distance_residual + angle_weight * angular_residual * angular_residual;
}

static CompetitionStats refine_patch_ownership(
    std::vector<PlanePatch>& patches,
    const std::vector<Surfel>& surfels,
    const std::unordered_map<CellKey, int, CellKeyHash>& index_by_key,
    const Config& cfg) {
    CompetitionStats stats;
    if (patches.empty() || surfels.empty()) return stats;

    std::vector<int> labels = build_surfel_labels(patches, surfels.size());
    const float smooth_cos = std::cos(cfg.smooth_angle_deg / 180.0f * 3.14159265358979323846f);
    const float envelope_margin = std::max(2.0f * cfg.surfel_size, cfg.coplanar_distance);
    std::vector<int> refined_labels = labels;

    for (std::size_t surfel_id = 0; surfel_id < surfels.size(); ++surfel_id) {
        const int current = labels[surfel_id];
        if (current < 0) continue;

        std::array<int, 27> candidates{};
        int candidate_count = 0;
        candidates[static_cast<std::size_t>(candidate_count++)] = current;
        for (int neighbor : surfel_neighbors(static_cast<int>(surfel_id), surfels, index_by_key)) {
            const int candidate = labels[static_cast<std::size_t>(neighbor)];
            if (candidate < 0) continue;
            bool duplicate = false;
            for (int k = 0; k < candidate_count; ++k) {
                if (candidates[static_cast<std::size_t>(k)] == candidate) {
                    duplicate = true;
                    break;
                }
            }
            if (!duplicate) candidates[static_cast<std::size_t>(candidate_count++)] = candidate;
        }

        int best = current;
        float best_score = surfel_plane_score(surfels[surfel_id], patches[static_cast<std::size_t>(current)], cfg);
        for (int k = 0; k < candidate_count; ++k) {
            const int candidate = candidates[static_cast<std::size_t>(k)];
            if (candidate == current) continue;
            const PlanePatch& patch = patches[static_cast<std::size_t>(candidate)];
            const float distance = std::abs(patch.normal.dot(surfels[surfel_id].centroid) + patch.d);
            if (distance > cfg.coplanar_distance) continue;
            if (!point_inside_patch_envelope(patch, surfels[surfel_id].centroid, envelope_margin)) continue;
            const float normal_dot = std::abs(surfels[surfel_id].normal.dot(patch.normal));
            if (!surfels[surfel_id].edge_like && normal_dot < smooth_cos) continue;

            const float score = surfel_plane_score(surfels[surfel_id], patch, cfg);
            if (score + 0.05f < best_score) {
                best = candidate;
                best_score = score;
            }
        }

        if (best != current) {
            refined_labels[surfel_id] = best;
            ++stats.moved_surfels;
        }
    }

    if (stats.moved_surfels > 0) {
        std::vector<std::vector<int>> ids_by_patch(patches.size());
        for (std::size_t surfel_id = 0; surfel_id < refined_labels.size(); ++surfel_id) {
            const int label = refined_labels[surfel_id];
            if (label >= 0) ids_by_patch[static_cast<std::size_t>(label)].push_back(static_cast<int>(surfel_id));
        }
        for (std::size_t patch_id = 0; patch_id < patches.size(); ++patch_id) {
            if (ids_by_patch[patch_id].size() < 3) continue;
            patches[patch_id] = fit_patch_from_surfels(ids_by_patch[patch_id], surfels);
            update_patch_envelope(patches[patch_id], surfels);
        }
    }

    patches.erase(
        std::remove_if(patches.begin(), patches.end(), [&](const PlanePatch& patch) {
            return patch.surfels.size() < static_cast<std::size_t>(std::max(cfg.min_surfels, 3)) || !valid_shape(patch, cfg);
        }),
        patches.end());
    std::sort(patches.begin(), patches.end(), [](const PlanePatch& a, const PlanePatch& b) {
        return a.surfels.size() > b.surfels.size();
    });
    return stats;
}

static float estimate_assign_margin(
    const CloudPtr& cloud,
    const std::vector<PlanePatch>& patches,
    const Config& cfg) {
    const float tight_margin = std::max(4.0f * cfg.surfel_size, 2.0f);
    const float loose_margin = std::max(24.0f * cfg.surfel_size, tight_margin);
    if (cloud->empty() || patches.empty()) return tight_margin;

    const std::size_t sample_limit = 200000;
    const std::size_t stride = std::max<std::size_t>(1, cloud->size() / sample_limit);
    std::size_t sampled = 0;
    std::size_t tight_assigned = 0;
    std::size_t loose_assigned = 0;
    const auto assignable = [&](const Eigen::Vector3f& q, float margin) {
        for (const auto& patch : patches) {
            const float distance = std::abs(patch.normal.dot(q) + patch.d);
            if (distance <= cfg.assign_distance && point_inside_patch_envelope(patch, q, margin)) return true;
        }
        return false;
    };

    for (std::size_t i = 0; i < cloud->size(); i += stride) {
        const PointT& p = cloud->points[i];
        const Eigen::Vector3f q(p.x, p.y, p.z);
        ++sampled;
        const bool tight = assignable(q, tight_margin);
        if (tight) {
            ++tight_assigned;
            ++loose_assigned;
            continue;
        }
        if (assignable(q, loose_margin)) {
            ++loose_assigned;
        }
    }

    const std::size_t rescued = loose_assigned - tight_assigned;
    if (sampled == 0 || rescued == 0) return tight_margin;

    const double rescued_rate = static_cast<double>(rescued) / static_cast<double>(sampled);
    const double estimated_rescued_points = rescued_rate * static_cast<double>(cloud->size());
    return (rescued_rate >= 0.003 || estimated_rescued_points >= 20000.0) ? loose_margin : tight_margin;
}

static void assign_full_points(
    const CloudPtr& cloud,
    std::vector<PlanePatch>& patches,
    const Config& cfg) {
    for (auto& patch : patches) patch.output.reset(new CloudT);

    for (const auto& p : cloud->points) {
        const Eigen::Vector3f q(p.x, p.y, p.z);
        int best = -1;
        float best_distance = std::numeric_limits<float>::max();
        for (std::size_t i = 0; i < patches.size(); ++i) {
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

static std::vector<CornerCandidate> merge_corner_candidates(std::vector<CornerCandidate> candidates, float merge_radius) {
    std::vector<CornerCandidate> merged;
    const float merge_sq = merge_radius * merge_radius;
    for (auto& candidate : candidates) {
        bool absorbed = false;
        for (auto& group : merged) {
            if ((group.position - candidate.position).squaredNorm() <= merge_sq) {
                if (candidate.support > group.support) group.position = candidate.position;
                group.support = std::max(group.support, candidate.support);
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
    const auto nearby_support_count = [&](const Eigen::Vector3f& p, float radius) {
        PointT probe;
        probe.x = p.x();
        probe.y = p.y();
        probe.z = p.z();
        std::vector<int> indices;
        std::vector<float> distances;
        return support_tree.radiusSearch(probe, radius, indices, distances);
    };

    const float support_radius = std::max(cfg.corner_support_radius_scale * cfg.surfel_size, 1e-4f);
    const float merge_radius = std::max(cfg.intersection_merge_scale * cfg.surfel_size, 1e-4f);
    const float bbox_margin = std::max(3.0f * cfg.surfel_size, cfg.assign_distance);
    const float envelope_margin = std::max(cfg.assign_margin, cfg.surfel_size * 2.0f);
    const float plane_distance_threshold = std::max(2.0f * cfg.surfel_size, cfg.assign_distance);
    const float parallel_cos = 0.975f;

    for (std::size_t i = 0; i < patches.size(); ++i) {
        for (std::size_t j = i + 1; j < patches.size(); ++j) {
            if (std::abs(patches[i].normal.dot(patches[j].normal)) > parallel_cos) continue;
            for (std::size_t k = j + 1; k < patches.size(); ++k) {
                if (std::abs(patches[i].normal.dot(patches[k].normal)) > parallel_cos) continue;
                if (std::abs(patches[j].normal.dot(patches[k].normal)) > parallel_cos) continue;

                Eigen::Vector3f p;
                if (!intersection_of_three_planes(patches[i], patches[j], patches[k], p)) continue;
                if (!within_bounds(p, bounds, bbox_margin)) continue;
                if (!point_inside_patch_envelope(patches[i], p, envelope_margin)) continue;
                if (!point_inside_patch_envelope(patches[j], p, envelope_margin)) continue;
                if (!point_inside_patch_envelope(patches[k], p, envelope_margin)) continue;
                if (std::abs(patches[i].normal.dot(p) + patches[i].d) > plane_distance_threshold) continue;
                if (std::abs(patches[j].normal.dot(p) + patches[j].d) > plane_distance_threshold) continue;
                if (std::abs(patches[k].normal.dot(p) + patches[k].d) > plane_distance_threshold) continue;

                const int support = nearby_support_count(p, support_radius);
                if (support < cfg.min_corner_support) continue;
                if (nearby_support_count_on_patch(patches[i], p, support_radius) < cfg.min_plane_local_support) continue;
                if (nearby_support_count_on_patch(patches[j], p, support_radius) < cfg.min_plane_local_support) continue;
                if (nearby_support_count_on_patch(patches[k], p, support_radius) < cfg.min_plane_local_support) continue;
                candidates.push_back({p, support});
            }
        }
    }

    return merge_corner_candidates(std::move(candidates), merge_radius);
}

static void write_color_ply_header(std::ofstream& out, std::size_t vertex_count) {
    out << "ply\n"
        << "format ascii 1.0\n"
        << "element vertex " << vertex_count << "\n"
        << "property float x\n"
        << "property float y\n"
        << "property float z\n"
        << "property uchar red\n"
        << "property uchar green\n"
        << "property uchar blue\n"
        << "end_header\n";
    out.setf(std::ios::fixed, std::ios::floatfield);
    out.precision(6);
}

static void save_plane_patches(const fs::path& path, const std::vector<PlanePatch>& patches) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Failed to write PLY: " + path.string());
    std::size_t vertex_count = 0;
    for (const auto& patch : patches) vertex_count += patch.output->size();
    write_color_ply_header(out, vertex_count);
    for (std::size_t i = 0; i < patches.size(); ++i) {
        const auto color = color_for(i);
        for (const auto& p : patches[i].output->points) {
            out << p.x << ' ' << p.y << ' ' << p.z << ' '
                << static_cast<int>(color[0]) << ' '
                << static_cast<int>(color[1]) << ' '
                << static_cast<int>(color[2]) << '\n';
        }
    }
}

static void report_probe(
    const Config& cfg,
    const Bounds& bounds,
    const std::vector<Surfel>& surfels,
    const std::unordered_map<CellKey, int, CellKeyHash>& index_by_key,
    const std::vector<PlanePatch>& patches) {
    if (!cfg.has_probe) return;

    PointT probe_point;
    probe_point.x = cfg.probe.x();
    probe_point.y = cfg.probe.y();
    probe_point.z = cfg.probe.z();
    const auto local_it = index_by_key.find(point_key(probe_point, bounds, cfg.surfel_size));
    const std::vector<int> labels = build_surfel_labels(patches, surfels.size());

    std::cout << "Probe: " << cfg.probe.transpose() << "\n";
    if (local_it == index_by_key.end()) {
        std::cout << "Probe surfel: none\n";
    } else {
        const int surfel_id = local_it->second;
        const Surfel& surfel = surfels[static_cast<std::size_t>(surfel_id)];
        std::cout << "Probe surfel: " << surfel_id
                  << ", owner=" << labels[static_cast<std::size_t>(surfel_id)]
                  << ", edge=" << (surfel.edge_like ? "yes" : "no")
                  << ", normal=" << surfel.normal.transpose() << "\n";
    }

    struct Candidate {
        std::size_t id = 0;
        float distance = 0.0f;
        float normal_dot = 0.0f;
        bool inside_tight = false;
        bool inside_assign = false;
    };
    std::vector<Candidate> candidates;
    candidates.reserve(patches.size());
    for (std::size_t i = 0; i < patches.size(); ++i) {
        const PlanePatch& patch = patches[i];
        Candidate candidate;
        candidate.id = i;
        candidate.distance = std::abs(patch.normal.dot(cfg.probe) + patch.d);
        candidate.inside_tight = point_inside_patch_envelope(patch, cfg.probe, 2.0f * cfg.surfel_size);
        candidate.inside_assign = point_inside_patch_envelope(patch, cfg.probe, cfg.assign_margin);
        if (local_it != index_by_key.end()) {
            candidate.normal_dot = std::abs(
                surfels[static_cast<std::size_t>(local_it->second)].normal.dot(patch.normal));
        }
        candidates.push_back(candidate);
    }
    std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
        return a.distance < b.distance;
    });

    int selected = -1;
    for (const Candidate& candidate : candidates) {
        if (candidate.distance <= cfg.assign_distance && candidate.inside_assign) {
            selected = static_cast<int>(candidate.id);
            break;
        }
    }
    std::cout << "Probe dense assignment: " << selected << "\n";
    const std::size_t shown = std::min<std::size_t>(candidates.size(), 8);
    for (std::size_t rank = 0; rank < shown; ++rank) {
        const Candidate& candidate = candidates[rank];
        const PlanePatch& patch = patches[candidate.id];
        std::cout << "Probe candidate " << candidate.id
                  << ": plane=(" << patch.normal.transpose() << "; d=" << patch.d << ')'
                  << ", distance=" << candidate.distance
                  << ", normal_dot=" << candidate.normal_dot
                  << ", tight=" << (candidate.inside_tight ? "in" : "out")
                  << ", assign=" << (candidate.inside_assign ? "in" : "out")
                  << ", accepted=" << (candidate.distance <= cfg.assign_distance && candidate.inside_assign ? "yes" : "no")
                  << ", rms=" << patch.rms_distance
                  << ", surfels=" << patch.surfels.size()
                  << ", dense=" << patch.output->size() << "\n";
    }
}

static void save_feature_edges(
    const fs::path& path,
    const std::vector<Surfel>& surfels,
    const std::unordered_map<CellKey, int, CellKeyHash>& index_by_key,
    const std::vector<PlanePatch>& patches) {
    const std::vector<int> labels = build_surfel_labels(patches, surfels.size());

    const auto is_feature_edge = [&](std::size_t i) {
        if (labels[i] < 0) return false;
        for (int neighbor : surfel_neighbors(static_cast<int>(i), surfels, index_by_key)) {
            const int label = labels[static_cast<std::size_t>(neighbor)];
            if (label >= 0 && label != labels[i]) return true;
        }
        return false;
    };

    std::size_t vertex_count = 0;
    for (std::size_t i = 0; i < surfels.size(); ++i) {
        if (is_feature_edge(i)) ++vertex_count;
    }

    std::ofstream out(path);
    if (!out) throw std::runtime_error("Failed to write PLY: " + path.string());
    write_color_ply_header(out, vertex_count);
    for (std::size_t i = 0; i < surfels.size(); ++i) {
        if (!is_feature_edge(i)) continue;
        out << surfels[i].centroid.x() << ' '
            << surfels[i].centroid.y() << ' '
            << surfels[i].centroid.z() << " 255 160 20\n";
    }
}

static void save_corners(const fs::path& path, const std::vector<CornerCandidate>& corners) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Failed to write PLY: " + path.string());
    write_color_ply_header(out, corners.size());
    for (const auto& corner : corners) {
        out << corner.position.x() << ' '
            << corner.position.y() << ' '
            << corner.position.z() << " 255 255 255\n";
    }
}

int main(int argc, char** argv) {
    try {
        Config cfg = parse_args(argc, argv);
        CloudPtr cloud = load_cloud(cfg.input, cfg.input_scale);
        Bounds bounds;
        pcl::getMinMax3D(*cloud, bounds.min_pt, bounds.max_pt);

        std::unordered_map<CellKey, int, CellKeyHash> index_by_key;
        CloudPtr surfel_cloud;
        std::vector<Surfel> surfels = build_surfels(cloud, bounds, cfg, index_by_key, surfel_cloud);
        estimate_surfel_normals(surfels, surfel_cloud, cfg);
        mark_edge_surfels(surfels, index_by_key, cfg);
        std::vector<PlanePatch> patches = extract_seed_grown_patches(surfels, index_by_key, cfg);
        const std::size_t grown_patch_count = patches.size();
        merge_patches(patches, surfels, cfg);
        const std::size_t initially_merged_patch_count = patches.size();
        const CompetitionStats competition = refine_patch_ownership(patches, surfels, index_by_key, cfg);
        const std::size_t competed_patch_count = patches.size();
        merge_patches(patches, surfels, cfg);
        cfg.assign_margin = estimate_assign_margin(cloud, patches, cfg);
        assign_full_points(cloud, patches, cfg);
        report_probe(cfg, bounds, surfels, index_by_key, patches);
        const std::vector<CornerCandidate> corners = extract_corners_filtered(cloud, patches, bounds, cfg);

        fs::path out_dir = cfg.output_dir / cfg.input.stem();
        fs::create_directories(out_dir);
        save_plane_patches(out_dir / "edge_aware_plane_patches.ply", patches);
        save_feature_edges(out_dir / "feature_edges.ply", surfels, index_by_key, patches);
        save_corners(out_dir / "extracted_corners.ply", corners);

        if (cfg.verbose) {
            std::size_t output_points = 0;
            for (const auto& patch : patches) output_points += patch.output->size();
            std::cout << "Input points: " << cloud->size() << "\n";
            std::cout << "Surfels: " << surfels.size() << "\n";
            std::cout << "Grown/merged/competed patches: "
                      << grown_patch_count << '/' << initially_merged_patch_count << '/' << competed_patch_count << "\n";
            std::cout << "Boundary-reassigned surfels: " << competition.moved_surfels << "\n";
            std::cout << "Assignment margin: " << cfg.assign_margin << "\n";
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
