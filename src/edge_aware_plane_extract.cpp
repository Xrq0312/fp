#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
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
    bool audit_assignments = false;
    bool distance_only = false;
    bool has_probe = false;
    Eigen::Vector3f probe{0.0f, 0.0f, 0.0f};
    bool verbose = true;
    std::string pipeline = "baseline";
    std::string corner_mode = "global";
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

struct BoundaryConstraint {
    int other_plane = -1;
    float expected_sign = 1.0f;
    float tolerance = 0.0f;
};

struct CompetitionStats {
    std::size_t moved_surfels = 0;
    std::size_t pruned_models = 0;
    int iterations = 0;
    double final_energy = 0.0;
};

struct TopologyCorrection {
    PointT point;
    int from_plane = -1;
    int to_plane = -1;
};

struct DenseAssignmentStats {
    std::size_t normal_guided_points = 0;
    std::size_t locally_supported_points = 0;
    std::size_t established_guided_points = 0;
    std::size_t topology_guided_points = 0;
    std::size_t boundary_rejected_candidates = 0;
    std::size_t boundary_constraints = 0;
    std::size_t max_candidates_per_point = 0;
    std::vector<TopologyCorrection> corrections;
};

struct StageTimings {
    double load_ms = 0.0;
    double surfel_build_ms = 0.0;
    double proposal_ms = 0.0;
    double global_relations_ms = 0.0;
    double initial_merge_ms = 0.0;
    double competition_ms = 0.0;
    double final_merge_ms = 0.0;
    double assign_margin_ms = 0.0;
    double dense_assignment_ms = 0.0;
    double finalize_ms = 0.0;
    double audit_ms = 0.0;
    double corners_ms = 0.0;
    double save_ms = 0.0;
    double total_ms = 0.0;
};

struct FinalizeStats {
    std::size_t absorbed_patches = 0;
    std::size_t absorbed_points = 0;
    std::size_t discarded_sparse_points = 0;
};

struct GlobalNormalStats {
    std::size_t reliable_surfels = 0;
    float curvature_limit = 0.0f;
    std::vector<Eigen::Vector3f> directions;
};

struct AssignmentAuditRow {
    std::size_t dense_points = 0;
    std::size_t audited_points = 0;
    std::size_t normal_mismatch_points = 0;
    std::size_t recoverable_mismatch_points = 0;
    float worst_normal_dot = 1.0f;
    Eigen::Vector3f mismatch_min = Eigen::Vector3f::Constant(std::numeric_limits<float>::max());
    Eigen::Vector3f mismatch_max = Eigen::Vector3f::Constant(std::numeric_limits<float>::lowest());
    std::vector<std::size_t> alternative_counts;
};

struct CornerCandidate {
    Eigen::Vector3f position{0.0f, 0.0f, 0.0f};
    int support = 0;
};

struct FinitePlaneIntersection {
    int plane_a = -1;
    int plane_b = -1;
    Eigen::Vector3f origin{0.0f, 0.0f, 0.0f};
    Eigen::Vector3f direction{1.0f, 0.0f, 0.0f};
    float min_t = 0.0f;
    float max_t = 0.0f;
    float boundary_rms = 0.0f;
    std::size_t support = 0;
};

struct CornerExtractionResult {
    std::vector<CornerCandidate> corners;
    std::vector<FinitePlaneIntersection> finite_intersections;
    std::size_t local_triples = 0;
    std::size_t ill_conditioned_triples = 0;
    float maximum_accepted_condition = 0.0f;
};

static float median_value(std::vector<float> values);

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
            << "  --pipeline baseline|multiscale|joint|topology  select an ablation pipeline\n"
            << "  --corner-mode global|adjacency  choose global triples or observed finite intersections\n"
            << "  --audit-assignments  write per-plane dense-assignment diagnostics\n"
            << "  --distance-only  diagnostic A/B: disable normal-guided dense assignment\n"
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
        } else if (arg == "--pipeline") {
            if (i + 1 >= args.size()) throw std::runtime_error("Missing value after --pipeline.");
            cfg.pipeline = args[++i];
            if (cfg.pipeline != "baseline" && cfg.pipeline != "multiscale" &&
                cfg.pipeline != "joint" && cfg.pipeline != "topology") {
                throw std::runtime_error(
                    "Unknown pipeline: " + cfg.pipeline +
                    ". Available: baseline, multiscale, joint, topology.");
            }
        } else if (arg == "--audit-assignments") {
            cfg.audit_assignments = true;
        } else if (arg == "--corner-mode") {
            if (i + 1 >= args.size()) throw std::runtime_error("Missing value after --corner-mode.");
            cfg.corner_mode = args[++i];
            if (cfg.corner_mode != "global" && cfg.corner_mode != "adjacency") {
                throw std::runtime_error(
                    "Unknown corner mode: " + cfg.corner_mode + ". Available: global, adjacency.");
            }
        } else if (arg == "--distance-only") {
            cfg.distance_only = true;
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
    if (cfg.pipeline != "baseline") {
        // Fine residual faces are real deliverables, not sparse leftovers.  The
        // stricter proposal validation remains the primary false-positive guard.
        cfg.min_output_points = 400;
        cfg.min_extent = std::max(1.5f, 3.0f * cfg.surfel_size);
    }
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

static void estimate_surfel_normals(
    std::vector<Surfel>& surfels,
    const CloudPtr& surfel_cloud,
    int normal_k) {
    pcl::search::Search<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::NormalEstimation<PointT, pcl::Normal> ne;
    ne.setInputCloud(surfel_cloud);
    ne.setSearchMethod(tree);
    ne.setKSearch(std::max(normal_k, 3));
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

                    const bool angle_edge = std::abs(
                        surfels[i].normal.dot(surfels[static_cast<std::size_t>(j)].normal)) < edge_cos;
                    if (angle_edge) {
                        surfels[i].edge_like = true;
                        surfels[static_cast<std::size_t>(j)].edge_like = true;
                    }
                }
            }
        }
    }
}

static std::vector<Surfel> fuse_multiscale_surfels(
    const std::vector<Surfel>& coarse,
    const std::vector<Surfel>& local) {
    if (coarse.size() != local.size()) {
        throw std::runtime_error("Multi-scale surfel arrays have different sizes.");
    }
    std::vector<Surfel> fused = coarse;
    for (std::size_t i = 0; i < fused.size(); ++i) {
        const bool prefer_local = (coarse[i].edge_like && !local[i].edge_like) ||
            local[i].curvature + 1e-8f < coarse[i].curvature;
        if (prefer_local) {
            fused[i].normal = local[i].normal;
            fused[i].curvature = local[i].curvature;
        }
        // A hard boundary must persist at both normal-support scales.  A jump
        // visible at only one scale is retained later as soft competition
        // evidence, but it must not suppress a complete small face here.
        fused[i].edge_like = coarse[i].edge_like && local[i].edge_like;
    }
    return fused;
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

static int closest_global_normal_family(
    const Eigen::Vector3f& normal,
    const GlobalNormalStats& global_normals,
    float minimum_dot) {
    int best = -1;
    float best_dot = minimum_dot;
    for (std::size_t i = 0; i < global_normals.directions.size(); ++i) {
        const float dot = std::abs(normal.dot(global_normals.directions[i]));
        if (dot > best_dot) {
            best_dot = dot;
            best = static_cast<int>(i);
        }
    }
    return best;
}

static bool patches_can_merge(
    const PlanePatch& a,
    const PlanePatch& b,
    const Config& cfg,
    const GlobalNormalStats& global_normals) {
    const float normal_cos = std::cos(cfg.merge_angle_deg / 180.0f * 3.14159265358979323846f);
    const float dot = a.normal.dot(b.normal);
    const float family_cos = std::cos(cfg.smooth_angle_deg / 180.0f * 3.14159265358979323846f);
    const int a_family = closest_global_normal_family(a.normal, global_normals, family_cos);
    const int b_family = closest_global_normal_family(b.normal, global_normals, family_cos);
    const bool same_global_family = a_family >= 0 && a_family == b_family;
    if (std::abs(dot) < normal_cos && !same_global_family) return false;
    if (same_global_family) {
        const Eigen::Vector3f& direction = global_normals.directions[static_cast<std::size_t>(a_family)];
        if (std::abs(direction.dot(a.centroid - b.centroid)) > cfg.merge_distance) return false;
    } else {
        const float b_d = dot >= 0.0f ? b.d : -b.d;
        if (std::abs(a.d - b_d) > cfg.merge_distance) return false;
    }

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

static void merge_patches_impl(
    std::vector<PlanePatch>& patches,
    const std::vector<Surfel>& surfels,
    const Config& cfg,
    const GlobalNormalStats& global_normals) {
    bool changed = true;
    while (changed) {
        changed = false;
        for (std::size_t i = 0; i < patches.size() && !changed; ++i) {
            for (std::size_t j = i + 1; j < patches.size(); ++j) {
                if (!patches_can_merge(patches[i], patches[j], cfg, global_normals)) continue;
                patches[i].surfels.insert(patches[i].surfels.end(), patches[j].surfels.begin(), patches[j].surfels.end());
                // Multi-scale proposal passes intentionally overlap.  Keep one
                // vote per surfel when the same physical face is rediscovered.
                std::sort(patches[i].surfels.begin(), patches[i].surfels.end());
                patches[i].surfels.erase(
                    std::unique(patches[i].surfels.begin(), patches[i].surfels.end()),
                    patches[i].surfels.end());
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

static std::size_t merge_patches_adaptive(
    std::vector<PlanePatch>& patches,
    const std::vector<Surfel>& surfels,
    const Config& cfg,
    const GlobalNormalStats& global_normals) {
    const GlobalNormalStats no_global_relaxation;
    merge_patches_impl(patches, surfels, cfg, no_global_relaxation);

    // A single relaxed merge can be an accidental alignment.  Repeated merges
    // into the same globally stable direction are evidence of a systematically
    // fragmented face, which is the case this relaxation is meant to repair.
    std::vector<PlanePatch> globally_merged = patches;
    merge_patches_impl(globally_merged, surfels, cfg, global_normals);
    const std::size_t relaxed_merges = patches.size() - globally_merged.size();
    if (relaxed_merges >= 2) patches = std::move(globally_merged);
    return relaxed_merges >= 2 ? relaxed_merges : 0;
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

static std::vector<std::vector<BoundaryConstraint>> build_boundary_constraints(
    const std::vector<PlanePatch>& patches,
    const std::vector<Surfel>& surfels,
    const std::unordered_map<CellKey, int, CellKeyHash>& index_by_key,
    const Config& cfg) {
    std::vector<std::vector<BoundaryConstraint>> constraints(patches.size());
    if (patches.size() < 2) return constraints;
    const std::vector<int> labels = build_surfel_labels(patches, surfels.size());
    std::vector<float> log_support;
    log_support.reserve(patches.size());
    for (const PlanePatch& patch : patches) {
        log_support.push_back(std::log(static_cast<float>(std::max<std::size_t>(patch.surfels.size(), 1))));
    }
    const float median_log_support = median_value(log_support);
    std::vector<float> support_deviations;
    support_deviations.reserve(log_support.size());
    for (float value : log_support) support_deviations.push_back(std::abs(value - median_log_support));
    const float support_sigma = 1.4826f * median_value(std::move(support_deviations));
    const float boundary_authority_floor = median_log_support - 3.0f * support_sigma;
    std::set<std::pair<int, int>> adjacent_pairs;
    for (std::size_t surfel_id = 0; surfel_id < labels.size(); ++surfel_id) {
        const int first = labels[surfel_id];
        if (first < 0) continue;
        for (int neighbor : surfel_neighbors(static_cast<int>(surfel_id), surfels, index_by_key)) {
            const int second = labels[static_cast<std::size_t>(neighbor)];
            if (second < 0 || second == first) continue;
            adjacent_pairs.emplace(std::min(first, second), std::max(first, second));
        }
    }

    const float nonparallel_cos = std::cos(
        cfg.smooth_angle_deg / 180.0f * 3.14159265358979323846f);
    for (const auto& pair : adjacent_pairs) {
        const int first = pair.first;
        const int second = pair.second;
        const PlanePatch& a = patches[static_cast<std::size_t>(first)];
        const PlanePatch& b = patches[static_cast<std::size_t>(second)];
        if (std::abs(a.normal.dot(b.normal)) >= nonparallel_cos) continue;
        const float fit_sigma = std::hypot(a.rms_distance, b.rms_distance);
        const float tolerance = std::max(3.0f * fit_sigma, 1e-6f * cfg.surfel_size);

        const auto add_if_one_sided = [&](int owner, int other) {
            // Local/small proposals remain valid output models, but a support
            // outlier may not clip a better-established neighbor.  The lower
            // bound is inferred from this scene's log-support distribution, so
            // uniformly small scenes retain full boundary authority.
            if (log_support[static_cast<std::size_t>(other)] < boundary_authority_floor) return;
            std::size_t positive = 0;
            std::size_t negative = 0;
            const PlanePatch& owner_patch = patches[static_cast<std::size_t>(owner)];
            const PlanePatch& other_patch = patches[static_cast<std::size_t>(other)];
            for (int surfel_id : owner_patch.surfels) {
                const float signed_distance = other_patch.normal.dot(
                    surfels[static_cast<std::size_t>(surfel_id)].centroid) + other_patch.d;
                if (signed_distance > tolerance) {
                    ++positive;
                } else if (signed_distance < -tolerance) {
                    ++negative;
                }
            }
            const std::size_t decisive = positive + negative;
            if (decisive == 0) return;
            // A terminating boundary must be genuinely one-sided.  Merely
            // having a dominant side is insufficient: a large plane around an
            // internal opening can have unequal areas on the two sides of an
            // adjacent wall while still continuing across its infinite plane.
            // The 3-sigma dead band above absorbs fit noise; any decisive
            // support on both sides therefore vetoes the half-space constraint.
            if (positive > 0 && negative > 0) return;
            constraints[static_cast<std::size_t>(owner)].push_back({
                other,
                positive >= negative ? 1.0f : -1.0f,
                tolerance
            });
        };
        add_if_one_sided(first, second);
        add_if_one_sided(second, first);
    }
    return constraints;
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

static float joint_surfel_data_cost(
    const Surfel& surfel,
    const PlanePatch& patch,
    const Config& cfg) {
    const float distance = std::abs(patch.normal.dot(surfel.centroid) + patch.d);
    const float distance_limit = std::max(2.0f * cfg.coplanar_distance, cfg.surfel_size);
    if (distance > distance_limit) return std::numeric_limits<float>::infinity();
    if (!point_inside_patch_envelope(
            patch, surfel.centroid, std::max(2.0f * cfg.surfel_size, cfg.coplanar_distance))) {
        return std::numeric_limits<float>::infinity();
    }
    const float relaxed_angle = std::min(18.0f, 1.5f * cfg.smooth_angle_deg);
    const float relaxed_cos = std::cos(relaxed_angle / 180.0f * 3.14159265358979323846f);
    if (!surfel.edge_like && std::abs(surfel.normal.dot(patch.normal)) < relaxed_cos) {
        return std::numeric_limits<float>::infinity();
    }
    return std::min(surfel_plane_score(surfel, patch, cfg), 25.0f);
}

static CompetitionStats refine_patch_ownership_joint(
    std::vector<PlanePatch>& patches,
    const std::vector<Surfel>& surfels,
    const std::unordered_map<CellKey, int, CellKeyHash>& index_by_key,
    const Config& cfg) {
    CompetitionStats stats;
    if (patches.empty() || surfels.empty()) return stats;

    std::vector<std::vector<int>> neighbors(surfels.size());
    for (std::size_t i = 0; i < surfels.size(); ++i) {
        neighbors[i] = surfel_neighbors(static_cast<int>(i), surfels, index_by_key);
    }

    std::vector<int> labels = build_surfel_labels(patches, surfels.size());
    std::vector<bool> active(patches.size(), true);
    const int maximum_iterations = 5;
    const float outlier_cost = 4.0f;

    for (int iteration = 0; iteration < maximum_iterations; ++iteration) {
        std::vector<int> next_labels = labels;
        std::size_t moved_this_iteration = 0;
        double iteration_energy = 0.0;

        for (std::size_t surfel_id = 0; surfel_id < surfels.size(); ++surfel_id) {
            const Surfel& surfel = surfels[surfel_id];
            const float smooth_weight = surfel.edge_like ? 0.02f : 0.10f;
            int best_label = -1;
            float best_cost = outlier_cost;
            for (int neighbor : neighbors[surfel_id]) {
                if (labels[static_cast<std::size_t>(neighbor)] >= 0) best_cost += 0.5f * smooth_weight;
            }

            for (std::size_t patch_id = 0; patch_id < patches.size(); ++patch_id) {
                if (!active[patch_id] || patches[patch_id].surfels.empty()) continue;
                float cost = joint_surfel_data_cost(surfel, patches[patch_id], cfg);
                if (!std::isfinite(cost)) continue;
                for (int neighbor : neighbors[surfel_id]) {
                    const int neighbor_label = labels[static_cast<std::size_t>(neighbor)];
                    if (neighbor_label >= 0 && neighbor_label != static_cast<int>(patch_id)) {
                        cost += smooth_weight;
                    } else if (neighbor_label < 0) {
                        cost += 0.25f * smooth_weight;
                    }
                }

                // Deterministic inertia prevents equal-cost labels from
                // oscillating when two hypotheses explain a boundary equally.
                const float inertia = labels[surfel_id] == static_cast<int>(patch_id) ? -0.02f : 0.0f;
                if (cost + inertia < best_cost) {
                    best_cost = cost;
                    best_label = static_cast<int>(patch_id);
                }
            }

            next_labels[surfel_id] = best_label;
            iteration_energy += best_cost;
            if (best_label != labels[surfel_id]) ++moved_this_iteration;
        }

        labels.swap(next_labels);
        stats.moved_surfels += moved_this_iteration;
        stats.final_energy = iteration_energy;
        stats.iterations = iteration + 1;

        std::vector<std::vector<int>> ids_by_patch(patches.size());
        for (std::size_t surfel_id = 0; surfel_id < labels.size(); ++surfel_id) {
            const int label = labels[surfel_id];
            if (label >= 0) ids_by_patch[static_cast<std::size_t>(label)].push_back(static_cast<int>(surfel_id));
        }
        for (std::size_t patch_id = 0; patch_id < patches.size(); ++patch_id) {
            if (ids_by_patch[patch_id].size() < 3) {
                active[patch_id] = false;
                patches[patch_id].surfels.clear();
                continue;
            }
            patches[patch_id] = fit_patch_from_surfels(ids_by_patch[patch_id], surfels);
            update_patch_envelope(patches[patch_id], surfels);
        }
        if (moved_this_iteration == 0) break;
    }

    // PEARL-style label cost: a small model is redundant when almost all of its
    // surfels can move to another active model with negligible data-cost loss.
    const std::size_t small_model_limit = static_cast<std::size_t>(std::max(4 * cfg.min_surfels, 64));
    for (std::size_t patch_id = 0; patch_id < patches.size(); ++patch_id) {
        if (!active[patch_id] || patches[patch_id].surfels.empty() ||
            patches[patch_id].surfels.size() > small_model_limit) {
            continue;
        }
        std::size_t explainable = 0;
        for (int surfel_id : patches[patch_id].surfels) {
            const Surfel& surfel = surfels[static_cast<std::size_t>(surfel_id)];
            const float current_cost = joint_surfel_data_cost(surfel, patches[patch_id], cfg);
            float alternative_cost = std::numeric_limits<float>::infinity();
            for (std::size_t candidate = 0; candidate < patches.size(); ++candidate) {
                if (candidate == patch_id || !active[candidate] || patches[candidate].surfels.empty()) continue;
                alternative_cost = std::min(
                    alternative_cost, joint_surfel_data_cost(surfel, patches[candidate], cfg));
            }
            if (std::isfinite(alternative_cost) && alternative_cost <= current_cost + 0.20f) ++explainable;
        }
        if (explainable * 10 >= patches[patch_id].surfels.size() * 9) {
            active[patch_id] = false;
            ++stats.pruned_models;
        }
    }

    // Reassign support from a pruned label once; leaving it as an outlier is
    // preferable to forcing it onto a geometrically incompatible surface.
    for (std::size_t surfel_id = 0; surfel_id < labels.size(); ++surfel_id) {
        const int current = labels[surfel_id];
        if (current < 0 || active[static_cast<std::size_t>(current)]) continue;
        int best = -1;
        float best_cost = outlier_cost;
        for (std::size_t candidate = 0; candidate < patches.size(); ++candidate) {
            if (!active[candidate] || patches[candidate].surfels.empty()) continue;
            const float cost = joint_surfel_data_cost(surfels[surfel_id], patches[candidate], cfg);
            if (cost < best_cost) {
                best = static_cast<int>(candidate);
                best_cost = cost;
            }
        }
        labels[surfel_id] = best;
    }

    std::vector<std::vector<int>> final_ids(patches.size());
    for (std::size_t surfel_id = 0; surfel_id < labels.size(); ++surfel_id) {
        const int label = labels[surfel_id];
        if (label >= 0 && active[static_cast<std::size_t>(label)]) {
            final_ids[static_cast<std::size_t>(label)].push_back(static_cast<int>(surfel_id));
        }
    }
    std::vector<PlanePatch> retained;
    retained.reserve(patches.size());
    for (std::size_t patch_id = 0; patch_id < patches.size(); ++patch_id) {
        if (!active[patch_id] ||
            final_ids[patch_id].size() < static_cast<std::size_t>(std::max(cfg.min_surfels, 3))) {
            continue;
        }
        PlanePatch patch = fit_patch_from_surfels(final_ids[patch_id], surfels);
        update_patch_envelope(patch, surfels);
        if (valid_shape(patch, cfg)) retained.push_back(std::move(patch));
    }
    std::sort(retained.begin(), retained.end(), [](const PlanePatch& a, const PlanePatch& b) {
        return a.surfels.size() > b.surfels.size();
    });
    patches.swap(retained);
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

static DenseAssignmentStats assign_full_points(
    const CloudPtr& cloud,
    std::vector<PlanePatch>& patches,
    const std::vector<Surfel>& surfels,
    const std::unordered_map<CellKey, int, CellKeyHash>& index_by_key,
    const Bounds& bounds,
    const Config& cfg,
    bool enable_normal_guidance) {
    DenseAssignmentStats stats;
    for (auto& patch : patches) patch.output.reset(new CloudT);
    const float smooth_cos = std::cos(cfg.smooth_angle_deg / 180.0f * 3.14159265358979323846f);
    const float edge_cos = std::cos(cfg.edge_angle_deg / 180.0f * 3.14159265358979323846f);
    const int support_radius = 2;
    const std::vector<int> labels = build_surfel_labels(patches, surfels.size());
    const bool topology_aware = cfg.pipeline == "topology";
    const std::vector<std::vector<BoundaryConstraint>> boundary_constraints = topology_aware
        ? build_boundary_constraints(patches, surfels, index_by_key, cfg)
        : std::vector<std::vector<BoundaryConstraint>>(patches.size());
    for (const auto& constraints : boundary_constraints) stats.boundary_constraints += constraints.size();

    struct DenseCandidate {
        int id = -1;
        float distance = std::numeric_limits<float>::max();
        float normal_dot = 1.0f;
        int local_support = 0;
        bool inside_tight = false;
    };

    std::vector<DenseCandidate> candidates;
    candidates.reserve(patches.size());

    for (const auto& p : cloud->points) {
        const Eigen::Vector3f q(p.x, p.y, p.z);
        const CellKey key = point_key(p, bounds, cfg.surfel_size);
        const auto local = index_by_key.find(key);
        const Surfel* local_surfel = local != index_by_key.end()
            ? &surfels[static_cast<std::size_t>(local->second)]
            : nullptr;
        const int local_owner = local != index_by_key.end()
            ? labels[static_cast<std::size_t>(local->second)]
            : -1;
        const int established_local_owner = local_surfel != nullptr && !local_surfel->edge_like
            ? local_owner
            : -1;
        const bool has_local_normal = enable_normal_guidance && local_surfel != nullptr;

        candidates.clear();
        for (std::size_t i = 0; i < patches.size(); ++i) {
            const float distance = std::abs(patches[i].normal.dot(q) + patches[i].d);
            if (distance > cfg.assign_distance) continue;
            if (!point_inside_patch_envelope(patches[i], q, cfg.assign_margin)) continue;
            candidates.emplace_back();
            DenseCandidate& candidate = candidates.back();
            candidate.id = static_cast<int>(i);
            candidate.distance = distance;
            candidate.normal_dot = has_local_normal
                ? std::abs(local_surfel->normal.dot(patches[i].normal))
                : 1.0f;
            candidate.inside_tight = point_inside_patch_envelope(
                patches[i], q, 2.0f * cfg.surfel_size);
        }
        const std::size_t candidate_count = candidates.size();
        stats.max_candidates_per_point = std::max(stats.max_candidates_per_point, candidate_count);
        if (candidate_count == 0) continue;

        // Only ambiguous points pay for the local support lookup.  The lookup is
        // in the original 3-D surfel grid, so it cannot be fooled by a large 2-D
        // patch rectangle extending through an unrelated surface.
        const bool needs_local_support = candidate_count > 1;
        if (needs_local_support) {
            for (int dx = -support_radius; dx <= support_radius; ++dx) {
                for (int dy = -support_radius; dy <= support_radius; ++dy) {
                    for (int dz = -support_radius; dz <= support_radius; ++dz) {
                        const auto neighbor = index_by_key.find({key.x + dx, key.y + dy, key.z + dz});
                        if (neighbor == index_by_key.end()) continue;
                        const int label = labels[static_cast<std::size_t>(neighbor->second)];
                        for (std::size_t i = 0; i < candidate_count; ++i) {
                            if (candidates[i].id == label) {
                                ++candidates[i].local_support;
                                break;
                            }
                        }
                    }
                }
            }
        }

        std::size_t distance_best = candidate_count;
        for (std::size_t i = 0; i < candidate_count; ++i) {
            if (distance_best == candidate_count ||
                candidates[i].distance < candidates[distance_best].distance) {
                distance_best = i;
            }
        }
        if (distance_best == candidate_count) continue;

        // Preserve the joint pipeline's dense assignment exactly as the base
        // decision.  Topology is a conservative correction applied below, not
        // a replacement classifier for every point in the cloud.
        std::size_t selected = distance_best;
        std::size_t normal_best = candidate_count;
        const float candidate_normal_cos = local_surfel != nullptr && local_surfel->edge_like
            ? edge_cos : smooth_cos;
        for (std::size_t i = 0; i < candidate_count; ++i) {
            if (has_local_normal && (candidates[i].inside_tight || candidates[i].local_support > 0) &&
                candidates[i].normal_dot >= candidate_normal_cos &&
                candidates[i].distance <= candidates[distance_best].distance + cfg.coplanar_distance &&
                (normal_best == candidate_count || candidates[i].distance < candidates[normal_best].distance)) {
                normal_best = i;
            }
        }

        const bool distance_has_bad_normal = candidates[distance_best].normal_dot < smooth_cos;
        const bool distance_lacks_real_support = candidates[distance_best].local_support == 0;
        const bool edge_normal_is_decisive = local_surfel == nullptr || !local_surfel->edge_like ||
            candidates[distance_best].normal_dot < edge_cos;
        const bool correct_loose_or_unsupported = !candidates[distance_best].inside_tight ||
            distance_lacks_real_support;
        if (normal_best < candidate_count && distance_has_bad_normal && edge_normal_is_decisive &&
            correct_loose_or_unsupported) {
            selected = normal_best;
            if (selected != distance_best) {
                ++stats.normal_guided_points;
                if (local_surfel != nullptr && !local_surfel->edge_like &&
                    !candidates[distance_best].inside_tight && candidates[selected].inside_tight) {
                    ++stats.established_guided_points;
                }
                if (candidates[distance_best].inside_tight && distance_lacks_real_support) {
                    ++stats.locally_supported_points;
                }
            }
        }

        const auto crossed_neighbor = [&](const DenseCandidate& candidate) {
            if (!topology_aware || established_local_owner == candidate.id) return -1;
            float largest_violation = 0.0f;
            int crossed = -1;
            for (const BoundaryConstraint& constraint :
                    boundary_constraints[static_cast<std::size_t>(candidate.id)]) {
                const PlanePatch& other = patches[static_cast<std::size_t>(constraint.other_plane)];
                const float signed_distance = constraint.expected_sign * (other.normal.dot(q) + other.d);
                if (signed_distance >= -constraint.tolerance) continue;
                const float violation = -constraint.tolerance - signed_distance;
                if (violation > largest_violation) {
                    largest_violation = violation;
                    crossed = constraint.other_plane;
                }
            }
            return crossed;
        };

        const int crossed = crossed_neighbor(candidates[selected]);
        if (crossed >= 0) {
            std::size_t corrected = candidate_count;
            for (std::size_t i = 0; i < candidate_count; ++i) {
                if (candidates[i].id != crossed) continue;
                if (crossed_neighbor(candidates[i]) >= 0) continue;
                // Boundary geometry may only overturn the joint decision when
                // the adjacent feasible model has no weaker observed neighborhood
                // support.  This prevents a thin transition/edge proposal from
                // acquiring points merely because it lies on the allowed side.
                if (candidates[i].local_support < candidates[selected].local_support) continue;
                if (corrected == candidate_count || candidates[i].distance < candidates[corrected].distance) {
                    corrected = i;
                }
            }
            // Never turn a classified point into an outlier solely because all
            // candidates violate uncertain fitted boundaries.  Correct only
            // when the ordinary joint candidate set contains a feasible model.
            if (corrected < candidate_count) {
                stats.corrections.push_back({
                    p,
                    candidates[selected].id,
                    candidates[corrected].id
                });
                selected = corrected;
                ++stats.topology_guided_points;
                ++stats.boundary_rejected_candidates;
                if (candidates[selected].local_support > 0) {
                    ++stats.locally_supported_points;
                }
            }
        }

        const int best = candidates[selected].id;
        if (best >= 0) patches[static_cast<std::size_t>(best)].output->push_back(p);
    }

    return stats;
}

static Eigen::Vector3f canonical_normal(Eigen::Vector3f normal) {
    Eigen::Index dominant = 0;
    normal.cwiseAbs().maxCoeff(&dominant);
    if (normal[dominant] < 0.0f) normal = -normal;
    return normal;
}

static float median_value(std::vector<float> values) {
    if (values.empty()) return 0.0f;
    const std::size_t middle = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(middle), values.end());
    const float upper = values[middle];
    if (values.size() % 2 != 0) return upper;
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(middle - 1), values.end());
    return 0.5f * (upper + values[middle - 1]);
}

static GlobalNormalStats estimate_global_normal_families(
    const std::vector<Surfel>& surfels,
    const Config& cfg) {
    GlobalNormalStats stats;
    if (surfels.empty()) return stats;

    std::vector<float> curvatures;
    curvatures.reserve(surfels.size());
    for (const Surfel& surfel : surfels) {
        if (std::isfinite(surfel.curvature)) curvatures.push_back(std::max(surfel.curvature, 0.0f));
    }
    const float median_curvature = median_value(curvatures);
    std::vector<float> deviations;
    deviations.reserve(curvatures.size());
    for (float curvature : curvatures) deviations.push_back(std::abs(curvature - median_curvature));
    const float curvature_mad = median_value(std::move(deviations));
    stats.curvature_limit = median_curvature + std::max(3.0f * curvature_mad, median_curvature);

    std::vector<int> reliable;
    reliable.reserve(surfels.size());
    for (std::size_t i = 0; i < surfels.size(); ++i) {
        if (surfels[i].curvature <= stats.curvature_limit) reliable.push_back(static_cast<int>(i));
    }
    std::sort(reliable.begin(), reliable.end(), [&](int a, int b) {
        const Surfel& lhs = surfels[static_cast<std::size_t>(a)];
        const Surfel& rhs = surfels[static_cast<std::size_t>(b)];
        if (std::abs(lhs.curvature - rhs.curvature) > 1e-8f) return lhs.curvature < rhs.curvature;
        return lhs.count > rhs.count;
    });
    stats.reliable_surfels = reliable.size();

    struct Family {
        Eigen::Vector3d weighted_sum{0.0, 0.0, 0.0};
        Eigen::Vector3f direction{0.0f, 0.0f, 1.0f};
        double weight = 0.0;
        std::size_t members = 0;
    };
    std::vector<Family> families;
    const float family_angle = std::max(1.0f, 0.5f * cfg.smooth_angle_deg);
    const float family_cos = std::cos(family_angle / 180.0f * 3.14159265358979323846f);
    for (int id : reliable) {
        const Surfel& surfel = surfels[static_cast<std::size_t>(id)];
        const Eigen::Vector3f normal = canonical_normal(surfel.normal);
        int best = -1;
        float best_dot = family_cos;
        for (std::size_t i = 0; i < families.size(); ++i) {
            const float dot = families[i].direction.dot(normal);
            if (dot > best_dot) {
                best_dot = dot;
                best = static_cast<int>(i);
            }
        }

        const double weight = static_cast<double>(std::max(surfel.count, 1));
        if (best < 0) {
            Family family;
            family.weighted_sum = weight * normal.cast<double>();
            family.direction = normal;
            family.weight = weight;
            family.members = 1;
            families.push_back(family);
        } else {
            Family& family = families[static_cast<std::size_t>(best)];
            family.weighted_sum += weight * normal.cast<double>();
            family.weight += weight;
            ++family.members;
            family.direction = family.weighted_sum.cast<float>().normalized();
        }
    }

    const std::size_t minimum_members = static_cast<std::size_t>(std::max(cfg.min_surfels, 3));
    families.erase(
        std::remove_if(families.begin(), families.end(), [&](const Family& family) {
            return family.members < minimum_members;
        }),
        families.end());
    std::sort(families.begin(), families.end(), [](const Family& a, const Family& b) {
        return a.weight > b.weight;
    });
    stats.directions.reserve(families.size());
    for (const Family& family : families) stats.directions.push_back(family.direction);
    return stats;
}

static FinalizeStats finalize_dense_patches(
    std::vector<PlanePatch>& patches,
    const Config& cfg) {
    FinalizeStats stats;
    if (patches.empty()) return stats;

    const std::size_t minimum_points = static_cast<std::size_t>(std::max(cfg.min_output_points, 1));
    std::vector<bool> keep(patches.size(), false);
    for (std::size_t i = 0; i < patches.size(); ++i) {
        keep[i] = patches[i].output->size() >= minimum_points;
    }

    // Before applying the existing output-support filter, rescue an entire small
    // patch when all of its points have a valid destination on established faces.
    // This preserves points without turning a few isolated leftovers into a new
    // plane.  Partially unsupported patches still follow the original filter.
    for (std::size_t patch_id = 0; patch_id < patches.size(); ++patch_id) {
        if (keep[patch_id]) continue;
        std::vector<int> targets;
        targets.reserve(patches[patch_id].output->size());
        bool fully_absorbable = true;
        for (const PointT& point : patches[patch_id].output->points) {
            const Eigen::Vector3f q(point.x, point.y, point.z);
            int best = -1;
            float best_distance = std::numeric_limits<float>::max();
            for (std::size_t candidate = 0; candidate < patches.size(); ++candidate) {
                if (!keep[candidate]) continue;
                const float distance = std::abs(patches[candidate].normal.dot(q) + patches[candidate].d);
                if (distance > cfg.assign_distance || distance >= best_distance) continue;
                if (!point_inside_patch_envelope(patches[candidate], q, cfg.assign_margin)) continue;
                best = static_cast<int>(candidate);
                best_distance = distance;
            }
            if (best < 0) {
                fully_absorbable = false;
                break;
            }
            targets.push_back(best);
        }

        if (!fully_absorbable) {
            stats.discarded_sparse_points += patches[patch_id].output->size();
            continue;
        }

        for (std::size_t i = 0; i < targets.size(); ++i) {
            patches[static_cast<std::size_t>(targets[i])].output->push_back(
                patches[patch_id].output->points[i]);
        }
        ++stats.absorbed_patches;
        stats.absorbed_points += targets.size();
    }

    std::vector<PlanePatch> retained;
    retained.reserve(patches.size() - stats.absorbed_patches);
    for (std::size_t i = 0; i < patches.size(); ++i) {
        if (keep[i]) retained.push_back(std::move(patches[i]));
    }
    patches.swap(retained);

    for (auto& patch : patches) {
        patch.output->width = static_cast<std::uint32_t>(patch.output->size());
        patch.output->height = 1;
        patch.output->is_dense = true;
    }
    return stats;
}

static std::vector<AssignmentAuditRow> audit_dense_assignments(
    const std::vector<PlanePatch>& patches,
    const std::vector<Surfel>& surfels,
    const std::unordered_map<CellKey, int, CellKeyHash>& index_by_key,
    const Bounds& bounds,
    const Config& cfg) {
    std::vector<AssignmentAuditRow> audit(patches.size());
    const float smooth_cos = std::cos(cfg.smooth_angle_deg / 180.0f * 3.14159265358979323846f);

    for (std::size_t patch_id = 0; patch_id < patches.size(); ++patch_id) {
        AssignmentAuditRow& row = audit[patch_id];
        row.dense_points = patches[patch_id].output->size();
        row.alternative_counts.resize(patches.size(), 0);

        for (const PointT& point : patches[patch_id].output->points) {
            const auto local = index_by_key.find(point_key(point, bounds, cfg.surfel_size));
            if (local == index_by_key.end()) continue;
            const Surfel& surfel = surfels[static_cast<std::size_t>(local->second)];
            ++row.audited_points;

            const float assigned_dot = std::abs(surfel.normal.dot(patches[patch_id].normal));
            row.worst_normal_dot = std::min(row.worst_normal_dot, assigned_dot);
            if (assigned_dot >= smooth_cos) continue;
            ++row.normal_mismatch_points;

            int best_alternative = -1;
            float best_distance = std::numeric_limits<float>::max();
            for (std::size_t candidate_id = 0; candidate_id < patches.size(); ++candidate_id) {
                if (candidate_id == patch_id) continue;
                const PlanePatch& candidate = patches[candidate_id];
                const float normal_dot = std::abs(surfel.normal.dot(candidate.normal));
                if (normal_dot < smooth_cos) continue;
                const Eigen::Vector3f q(point.x, point.y, point.z);
                const float distance = std::abs(candidate.normal.dot(q) + candidate.d);
                if (distance > cfg.assign_distance || distance >= best_distance) continue;
                if (!point_inside_patch_envelope(candidate, q, 2.0f * cfg.surfel_size)) continue;
                best_distance = distance;
                best_alternative = static_cast<int>(candidate_id);
            }

            if (best_alternative >= 0) {
                ++row.recoverable_mismatch_points;
                ++row.alternative_counts[static_cast<std::size_t>(best_alternative)];
                const Eigen::Vector3f q(point.x, point.y, point.z);
                row.mismatch_min = row.mismatch_min.cwiseMin(q);
                row.mismatch_max = row.mismatch_max.cwiseMax(q);
            }
        }
    }
    return audit;
}

static void save_assignment_audit(
    const fs::path& path,
    const std::vector<AssignmentAuditRow>& audit) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Failed to write assignment audit: " + path.string());
    out << "plane_id,dense_points,audited_points,normal_mismatch_points,recoverable_mismatch_points,"
           "mismatch_rate,dominant_alternative,dominant_alternative_points,worst_normal_dot,"
           "min_x,min_y,min_z,max_x,max_y,max_z\n";
    for (std::size_t plane_id = 0; plane_id < audit.size(); ++plane_id) {
        const AssignmentAuditRow& row = audit[plane_id];
        std::size_t dominant_alternative = 0;
        std::size_t dominant_count = 0;
        for (std::size_t i = 0; i < row.alternative_counts.size(); ++i) {
            if (row.alternative_counts[i] > dominant_count) {
                dominant_count = row.alternative_counts[i];
                dominant_alternative = i;
            }
        }
        const double mismatch_rate = row.audited_points > 0
            ? static_cast<double>(row.recoverable_mismatch_points) / static_cast<double>(row.audited_points)
            : 0.0;
        out << plane_id << ',' << row.dense_points << ',' << row.audited_points << ','
            << row.normal_mismatch_points << ',' << row.recoverable_mismatch_points << ','
            << mismatch_rate << ',';
        if (dominant_count > 0) out << dominant_alternative;
        out << ',' << dominant_count << ',' << row.worst_normal_dot << ',';
        if (row.recoverable_mismatch_points > 0) {
            out << row.mismatch_min.x() << ',' << row.mismatch_min.y() << ',' << row.mismatch_min.z() << ','
                << row.mismatch_max.x() << ',' << row.mismatch_max.y() << ',' << row.mismatch_max.z();
        } else {
            out << ",,,,,";
        }
        out << '\n';
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

static bool intersection_of_two_planes(
    const PlanePatch& a,
    const PlanePatch& b,
    Eigen::Vector3f& origin,
    Eigen::Vector3f& direction) {
    direction = a.normal.cross(b.normal);
    const float direction_norm = direction.norm();
    if (direction_norm < 1e-5f) return false;
    direction /= direction_norm;

    Eigen::Matrix3f system;
    system.row(0) = a.normal.transpose();
    system.row(1) = b.normal.transpose();
    system.row(2) = direction.transpose();
    origin = system.colPivHouseholderQr().solve(Eigen::Vector3f(-a.d, -b.d, 0.0f));
    return origin.allFinite();
}

static bool intersection_of_three_planes_svd(
    const PlanePatch& a,
    const PlanePatch& b,
    const PlanePatch& c,
    Eigen::Vector3f& out,
    float& condition_number) {
    Eigen::Matrix3f system;
    system.row(0) = a.normal.transpose();
    system.row(1) = b.normal.transpose();
    system.row(2) = c.normal.transpose();
    Eigen::JacobiSVD<Eigen::Matrix3f> svd(system, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const Eigen::Vector3f singular = svd.singularValues();
    if (singular.x() <= 0.0f || singular.z() <= singular.x() * 1e-4f) {
        condition_number = std::numeric_limits<float>::infinity();
        return false;
    }
    condition_number = singular.x() / singular.z();
    out = svd.solve(Eigen::Vector3f(-a.d, -b.d, -c.d));
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

static std::vector<FinitePlaneIntersection> build_finite_plane_intersections(
    const std::vector<PlanePatch>& patches,
    const std::vector<Surfel>& surfels,
    const std::unordered_map<CellKey, int, CellKeyHash>& index_by_key,
    const Config& cfg) {
    if (patches.size() < 2 || surfels.empty()) return {};
    const std::vector<int> labels = build_surfel_labels(patches, surfels.size());
    const std::size_t plane_count = patches.size();
    std::vector<std::vector<Eigen::Vector3f>> interface_samples(plane_count * plane_count);

    for (std::size_t surfel_id = 0; surfel_id < surfels.size(); ++surfel_id) {
        const int label_a = labels[surfel_id];
        if (label_a < 0) continue;
        for (int neighbor : surfel_neighbors(static_cast<int>(surfel_id), surfels, index_by_key)) {
            if (neighbor <= static_cast<int>(surfel_id)) continue;
            const int label_b = labels[static_cast<std::size_t>(neighbor)];
            if (label_b < 0 || label_a == label_b) continue;
            const int a = std::min(label_a, label_b);
            const int b = std::max(label_a, label_b);
            interface_samples[static_cast<std::size_t>(a) * plane_count + static_cast<std::size_t>(b)]
                .push_back(0.5f * (surfels[surfel_id].centroid +
                    surfels[static_cast<std::size_t>(neighbor)].centroid));
        }
    }

    std::vector<FinitePlaneIntersection> intersections;
    const std::size_t minimum_support = static_cast<std::size_t>(std::max(cfg.min_plane_local_support, 3));
    const float maximum_boundary_distance = std::max(2.5f * cfg.surfel_size, 2.0f * cfg.coplanar_distance);
    for (std::size_t a = 0; a < plane_count; ++a) {
        for (std::size_t b = a + 1; b < plane_count; ++b) {
            const std::vector<Eigen::Vector3f>& samples = interface_samples[a * plane_count + b];
            if (samples.size() < minimum_support) continue;

            FinitePlaneIntersection edge;
            edge.plane_a = static_cast<int>(a);
            edge.plane_b = static_cast<int>(b);
            if (!intersection_of_two_planes(patches[a], patches[b], edge.origin, edge.direction)) continue;

            std::vector<float> parameters;
            parameters.reserve(samples.size());
            double squared_error = 0.0;
            for (const Eigen::Vector3f& sample : samples) {
                const float t = edge.direction.dot(sample - edge.origin);
                const Eigen::Vector3f projected = edge.origin + t * edge.direction;
                const float distance = (sample - projected).norm();
                if (distance > maximum_boundary_distance) continue;
                parameters.push_back(t);
                squared_error += static_cast<double>(distance) * static_cast<double>(distance);
            }
            if (parameters.size() < minimum_support) continue;

            std::sort(parameters.begin(), parameters.end());
            edge.min_t = parameters.front();
            edge.max_t = parameters.back();
            edge.support = parameters.size();
            edge.boundary_rms = static_cast<float>(std::sqrt(squared_error / parameters.size()));
            intersections.push_back(edge);
        }
    }
    return intersections;
}

static CornerExtractionResult extract_corners_from_adjacency(
    const CloudPtr& cloud,
    const std::vector<PlanePatch>& patches,
    const std::vector<Surfel>& surfels,
    const std::unordered_map<CellKey, int, CellKeyHash>& index_by_key,
    const Bounds& bounds,
    const Config& cfg) {
    CornerExtractionResult result;
    result.finite_intersections =
        build_finite_plane_intersections(patches, surfels, index_by_key, cfg);
    if (patches.size() < 3 || result.finite_intersections.size() < 2) return result;

    const std::size_t plane_count = patches.size();
    std::vector<int> edge_by_pair(plane_count * plane_count, -1);
    std::vector<std::vector<int>> graph(plane_count);
    for (std::size_t edge_id = 0; edge_id < result.finite_intersections.size(); ++edge_id) {
        const FinitePlaneIntersection& edge = result.finite_intersections[edge_id];
        const std::size_t a = static_cast<std::size_t>(edge.plane_a);
        const std::size_t b = static_cast<std::size_t>(edge.plane_b);
        edge_by_pair[a * plane_count + b] = static_cast<int>(edge_id);
        edge_by_pair[b * plane_count + a] = static_cast<int>(edge_id);
        graph[a].push_back(edge.plane_b);
        graph[b].push_back(edge.plane_a);
    }

    std::set<std::array<int, 3>> local_triples;
    for (std::size_t center = 0; center < graph.size(); ++center) {
        std::vector<int>& neighbors = graph[center];
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
        for (std::size_t i = 0; i < neighbors.size(); ++i) {
            for (std::size_t j = i + 1; j < neighbors.size(); ++j) {
                std::array<int, 3> triple{
                    static_cast<int>(center), neighbors[i], neighbors[j]
                };
                std::sort(triple.begin(), triple.end());
                local_triples.insert(triple);
            }
        }
    }
    result.local_triples = local_triples.size();

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
    std::vector<CornerCandidate> candidates;
    for (const std::array<int, 3>& triple : local_triples) {
        const std::size_t i = static_cast<std::size_t>(triple[0]);
        const std::size_t j = static_cast<std::size_t>(triple[1]);
        const std::size_t k = static_cast<std::size_t>(triple[2]);
        Eigen::Vector3f p;
        float condition_number = 0.0f;
        if (!intersection_of_three_planes_svd(
                patches[i], patches[j], patches[k], p, condition_number)) {
            ++result.ill_conditioned_triples;
            continue;
        }
        result.maximum_accepted_condition =
            std::max(result.maximum_accepted_condition, condition_number);
        if (!within_bounds(p, bounds, bbox_margin)) continue;
        if (!point_inside_patch_envelope(patches[i], p, envelope_margin) ||
            !point_inside_patch_envelope(patches[j], p, envelope_margin) ||
            !point_inside_patch_envelope(patches[k], p, envelope_margin)) {
            continue;
        }

        int supported_lines = 0;
        const std::array<std::pair<std::size_t, std::size_t>, 3> pairs{{
            {i, j}, {i, k}, {j, k}
        }};
        for (const auto& pair : pairs) {
            const int edge_id = edge_by_pair[pair.first * plane_count + pair.second];
            if (edge_id < 0) continue;
            const FinitePlaneIntersection& edge =
                result.finite_intersections[static_cast<std::size_t>(edge_id)];
            const float t = edge.direction.dot(p - edge.origin);
            const float line_margin = std::max(3.0f * cfg.surfel_size, 2.0f * edge.boundary_rms);
            if (t >= edge.min_t - line_margin && t <= edge.max_t + line_margin) ++supported_lines;
        }
        // Two observed boundary branches are sufficient: the third one may be
        // hidden by occlusion or absent from a partial scan.
        if (supported_lines < 2) continue;

        const int support = nearby_support_count(p, support_radius);
        if (support < cfg.min_corner_support) continue;
        if (nearby_support_count_on_patch(patches[i], p, support_radius) < cfg.min_plane_local_support) continue;
        if (nearby_support_count_on_patch(patches[j], p, support_radius) < cfg.min_plane_local_support) continue;
        if (nearby_support_count_on_patch(patches[k], p, support_radius) < cfg.min_plane_local_support) continue;
        candidates.push_back({p, support});
    }
    result.corners = merge_corner_candidates(std::move(candidates), merge_radius);
    return result;
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

static void save_topology_corrections(
    const fs::path& ply_path,
    const fs::path& csv_path,
    const std::vector<TopologyCorrection>& corrections) {
    std::ofstream ply(ply_path);
    if (!ply) throw std::runtime_error("Failed to write topology corrections: " + ply_path.string());
    write_color_ply_header(ply, corrections.size());
    for (const TopologyCorrection& correction : corrections) {
        ply << correction.point.x << ' ' << correction.point.y << ' ' << correction.point.z
            << " 255 48 48\n";
    }

    std::ofstream csv(csv_path);
    if (!csv) throw std::runtime_error("Failed to write topology corrections: " + csv_path.string());
    csv << "x,y,z,from_plane,to_plane\n";
    csv << std::setprecision(9);
    for (const TopologyCorrection& correction : corrections) {
        csv << correction.point.x << ',' << correction.point.y << ',' << correction.point.z << ','
            << correction.from_plane << ',' << correction.to_plane << '\n';
    }
}

static void save_plane_parameters_csv(const fs::path& path, const std::vector<PlanePatch>& patches) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Failed to write plane parameters: " + path.string());
    out << "plane_id,surfels,dense_points,nx,ny,nz,d,cx,cy,cz,extent_u,extent_v,rms_distance\n";
    out << std::setprecision(9);
    for (std::size_t i = 0; i < patches.size(); ++i) {
        const PlanePatch& patch = patches[i];
        out << i << ',' << patch.surfels.size() << ',' << patch.output->size() << ','
            << patch.normal.x() << ',' << patch.normal.y() << ',' << patch.normal.z() << ',' << patch.d << ','
            << patch.centroid.x() << ',' << patch.centroid.y() << ',' << patch.centroid.z() << ','
            << patch.extent_u << ',' << patch.extent_v << ',' << patch.rms_distance << '\n';
    }
}

static void report_probe(
    const Config& cfg,
    const Bounds& bounds,
    const std::vector<Surfel>& surfels,
    const std::unordered_map<CellKey, int, CellKeyHash>& index_by_key,
    const std::vector<PlanePatch>& patches,
    bool enable_normal_guidance) {
    if (!cfg.has_probe) return;

    PointT probe_point;
    probe_point.x = cfg.probe.x();
    probe_point.y = cfg.probe.y();
    probe_point.z = cfg.probe.z();
    const CellKey probe_key = point_key(probe_point, bounds, cfg.surfel_size);
    const auto local_it = index_by_key.find(probe_key);
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
        int local_support = 0;
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
    for (int dx = -2; dx <= 2; ++dx) {
        for (int dy = -2; dy <= 2; ++dy) {
            for (int dz = -2; dz <= 2; ++dz) {
                const auto neighbor = index_by_key.find({probe_key.x + dx, probe_key.y + dy, probe_key.z + dz});
                if (neighbor == index_by_key.end()) continue;
                const int label = labels[static_cast<std::size_t>(neighbor->second)];
                if (label >= 0 && static_cast<std::size_t>(label) < candidates.size()) {
                    ++candidates[static_cast<std::size_t>(label)].local_support;
                }
            }
        }
    }
    std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
        return a.distance < b.distance;
    });

    int distance_selected = -1;
    int normal_selected = -1;
    const float smooth_cos = std::cos(cfg.smooth_angle_deg / 180.0f * 3.14159265358979323846f);
    const float edge_cos = std::cos(cfg.edge_angle_deg / 180.0f * 3.14159265358979323846f);
    const bool has_local_normal = enable_normal_guidance && local_it != index_by_key.end();
    const bool local_is_edge = local_it != index_by_key.end() &&
        surfels[static_cast<std::size_t>(local_it->second)].edge_like;
    const float candidate_normal_cos = local_is_edge ? edge_cos : smooth_cos;
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        const Candidate& candidate = candidates[i];
        if (candidate.distance > cfg.assign_distance || !candidate.inside_assign) continue;
        if (distance_selected < 0) distance_selected = static_cast<int>(i);
        const Candidate& distance_candidate = candidates[static_cast<std::size_t>(distance_selected)];
        if (has_local_normal && (candidate.inside_tight || candidate.local_support > 0) &&
            candidate.normal_dot >= candidate_normal_cos &&
            candidate.distance <= distance_candidate.distance + cfg.coplanar_distance &&
            normal_selected < 0) {
            normal_selected = static_cast<int>(i);
        }
    }
    int selected = distance_selected;
    if (distance_selected >= 0 && normal_selected >= 0) {
        const Candidate& distance_candidate = candidates[static_cast<std::size_t>(distance_selected)];
        const bool distance_has_bad_normal = distance_candidate.normal_dot < smooth_cos;
        const bool edge_normal_is_decisive = !local_is_edge || distance_candidate.normal_dot < edge_cos;
        if (distance_has_bad_normal && edge_normal_is_decisive &&
            (!distance_candidate.inside_tight || distance_candidate.local_support == 0)) {
            selected = normal_selected;
        }
    }
    std::cout << "Probe dense assignment: "
              << (selected >= 0 ? static_cast<int>(candidates[static_cast<std::size_t>(selected)].id) : -1) << "\n";
    const std::size_t shown = std::min<std::size_t>(candidates.size(), 8);
    for (std::size_t rank = 0; rank < shown; ++rank) {
        const Candidate& candidate = candidates[rank];
        const PlanePatch& patch = patches[candidate.id];
        std::cout << "Probe candidate " << candidate.id
                  << ": plane=(" << patch.normal.transpose() << "; d=" << patch.d << ')'
                  << ", distance=" << candidate.distance
                  << ", normal_dot=" << candidate.normal_dot
                  << ", local_support=" << candidate.local_support
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

static void save_finite_intersections(
    const fs::path& path,
    const std::vector<FinitePlaneIntersection>& intersections) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Failed to write finite intersections: " + path.string());
    out << "plane_a,plane_b,support,boundary_rms,start_x,start_y,start_z,end_x,end_y,end_z\n";
    out << std::setprecision(9);
    for (const FinitePlaneIntersection& edge : intersections) {
        const Eigen::Vector3f start = edge.origin + edge.min_t * edge.direction;
        const Eigen::Vector3f end = edge.origin + edge.max_t * edge.direction;
        out << edge.plane_a << ',' << edge.plane_b << ',' << edge.support << ',' << edge.boundary_rms << ','
            << start.x() << ',' << start.y() << ',' << start.z() << ','
            << end.x() << ',' << end.y() << ',' << end.z() << '\n';
    }
}

int main(int argc, char** argv) {
    try {
        using Clock = std::chrono::steady_clock;
        const auto program_start = Clock::now();
        auto stage_start = program_start;
        const auto elapsed_ms = [](Clock::time_point start, Clock::time_point end) {
            return std::chrono::duration<double, std::milli>(end - start).count();
        };
        StageTimings timings;

        Config cfg = parse_args(argc, argv);
        CloudPtr cloud = load_cloud(cfg.input, cfg.input_scale);
        Bounds bounds;
        pcl::getMinMax3D(*cloud, bounds.min_pt, bounds.max_pt);
        auto now = Clock::now();
        timings.load_ms = elapsed_ms(stage_start, now);
        stage_start = now;

        std::unordered_map<CellKey, int, CellKeyHash> index_by_key;
        CloudPtr surfel_cloud;
        std::vector<Surfel> surfels = build_surfels(cloud, bounds, cfg, index_by_key, surfel_cloud);
        now = Clock::now();
        timings.surfel_build_ms = elapsed_ms(stage_start, now);
        stage_start = now;
        estimate_surfel_normals(surfels, surfel_cloud, cfg.normal_k);

        std::vector<PlanePatch> patches;
        std::size_t coarse_candidate_count = 0;
        std::size_t local_candidate_count = 0;
        if (cfg.pipeline == "baseline") {
            mark_edge_surfels(surfels, index_by_key, cfg);
            patches = extract_seed_grown_patches(surfels, index_by_key, cfg);
            coarse_candidate_count = patches.size();
        } else {
            std::vector<Surfel> coarse_surfels = surfels;
            std::vector<Surfel> local_surfels = surfels;
            estimate_surfel_normals(local_surfels, surfel_cloud, std::max(6, cfg.normal_k / 2));
            mark_edge_surfels(coarse_surfels, index_by_key, cfg);
            mark_edge_surfels(local_surfels, index_by_key, cfg);

            std::vector<PlanePatch> coarse_patches =
                extract_seed_grown_patches(coarse_surfels, index_by_key, cfg);
            coarse_candidate_count = coarse_patches.size();

            surfels = fuse_multiscale_surfels(coarse_surfels, local_surfels);
            Config local_cfg = cfg;
            local_cfg.min_surfels = std::max(8, cfg.min_surfels / 2);
            local_cfg.min_extent = std::max(1.5f, 3.0f * cfg.surfel_size);
            local_cfg.smooth_angle_deg = std::min(10.0f, cfg.smooth_angle_deg + 2.0f);
            local_cfg.max_planes = std::max(cfg.max_planes, 240);
            std::vector<PlanePatch> local_patches =
                extract_seed_grown_patches(surfels, index_by_key, local_cfg);
            local_candidate_count = local_patches.size();

            patches = std::move(coarse_patches);
            patches.insert(
                patches.end(),
                std::make_move_iterator(local_patches.begin()),
                std::make_move_iterator(local_patches.end()));
        }
        now = Clock::now();
        timings.proposal_ms = elapsed_ms(stage_start, now);
        stage_start = now;

        const GlobalNormalStats global_normals = estimate_global_normal_families(surfels, cfg);
        now = Clock::now();
        timings.global_relations_ms = elapsed_ms(stage_start, now);
        stage_start = now;
        const std::size_t grown_patch_count = patches.size();
        std::size_t global_family_merges = merge_patches_adaptive(patches, surfels, cfg, global_normals);
        now = Clock::now();
        timings.initial_merge_ms = elapsed_ms(stage_start, now);
        stage_start = now;
        const std::size_t initially_merged_patch_count = patches.size();
        const CompetitionStats competition =
            (cfg.pipeline == "joint" || cfg.pipeline == "topology")
            ? refine_patch_ownership_joint(patches, surfels, index_by_key, cfg)
            : refine_patch_ownership(patches, surfels, index_by_key, cfg);
        now = Clock::now();
        timings.competition_ms = elapsed_ms(stage_start, now);
        stage_start = now;
        const std::size_t competed_patch_count = patches.size();
        global_family_merges += merge_patches_adaptive(patches, surfels, cfg, global_normals);
        now = Clock::now();
        timings.final_merge_ms = elapsed_ms(stage_start, now);
        stage_start = now;
        cfg.assign_margin = estimate_assign_margin(cloud, patches, cfg);
        now = Clock::now();
        timings.assign_margin_ms = elapsed_ms(stage_start, now);
        stage_start = now;
        DenseAssignmentStats dense_assignment = assign_full_points(
            cloud, patches, surfels, index_by_key, bounds, cfg, true);
        // Isolated conflicts are often unstable edge normals. Require evidence comparable
        // to a fraction of one deliverable patch and to the cloud size before changing labels.
        const std::size_t minimum_guided_evidence = std::max<std::size_t>(
            static_cast<std::size_t>(std::max(cfg.min_output_points, 1)) / 4,
            cloud->size() / 10000);
        const bool topology_guidance_applied = cfg.pipeline == "topology";
        const bool normal_guidance_applied = topology_guidance_applied ||
            (!cfg.distance_only && dense_assignment.established_guided_points >= minimum_guided_evidence);
        if (!normal_guidance_applied) {
            assign_full_points(cloud, patches, surfels, index_by_key, bounds, cfg, false);
        }
        now = Clock::now();
        timings.dense_assignment_ms = elapsed_ms(stage_start, now);
        stage_start = now;
        const FinalizeStats finalize_stats = finalize_dense_patches(patches, cfg);
        now = Clock::now();
        timings.finalize_ms = elapsed_ms(stage_start, now);
        stage_start = now;
        const std::vector<AssignmentAuditRow> assignment_audit = cfg.audit_assignments
            ? audit_dense_assignments(patches, surfels, index_by_key, bounds, cfg)
            : std::vector<AssignmentAuditRow>{};
        report_probe(cfg, bounds, surfels, index_by_key, patches, normal_guidance_applied);
        now = Clock::now();
        timings.audit_ms = elapsed_ms(stage_start, now);
        stage_start = now;
        CornerExtractionResult corner_result;
        if (cfg.corner_mode == "adjacency") {
            corner_result = extract_corners_from_adjacency(
                cloud, patches, surfels, index_by_key, bounds, cfg);
        } else {
            corner_result.corners = extract_corners_filtered(cloud, patches, bounds, cfg);
        }
        const std::vector<CornerCandidate>& corners = corner_result.corners;
        now = Clock::now();
        timings.corners_ms = elapsed_ms(stage_start, now);
        stage_start = now;

        fs::path out_dir = cfg.output_dir / cfg.input.stem();
        fs::create_directories(out_dir);
        save_plane_patches(out_dir / "edge_aware_plane_patches.ply", patches);
        save_plane_parameters_csv(out_dir / "plane_parameters.csv", patches);
        save_feature_edges(out_dir / "feature_edges.ply", surfels, index_by_key, patches);
        save_corners(out_dir / "extracted_corners.ply", corners);
        if (cfg.corner_mode == "adjacency") {
            save_finite_intersections(
                out_dir / "finite_plane_intersections.csv", corner_result.finite_intersections);
        }
        if (cfg.pipeline == "topology") {
            save_topology_corrections(
                out_dir / "topology_corrections.ply",
                out_dir / "topology_corrections.csv",
                dense_assignment.corrections);
        }
        if (cfg.audit_assignments) {
            save_assignment_audit(out_dir / "assignment_audit.csv", assignment_audit);
        }
        now = Clock::now();
        timings.save_ms = elapsed_ms(stage_start, now);
        timings.total_ms = elapsed_ms(program_start, now);

        if (cfg.verbose) {
            std::size_t output_points = 0;
            for (const auto& patch : patches) output_points += patch.output->size();
            std::cout << "Pipeline: " << cfg.pipeline << "\n";
            std::cout << "Input points: " << cloud->size() << "\n";
            std::cout << "Surfels: " << surfels.size() << "\n";
            std::cout << "Global normal families/reliable surfels: "
                      << global_normals.directions.size() << '/' << global_normals.reliable_surfels << "\n";
            std::cout << "Coarse/local candidate patches: "
                      << coarse_candidate_count << '/' << local_candidate_count << "\n";
            std::cout << "Accepted global-family merges: " << global_family_merges << "\n";
            std::cout << "Grown/merged/competed patches: "
                      << grown_patch_count << '/' << initially_merged_patch_count << '/' << competed_patch_count << "\n";
            std::cout << "Boundary-reassigned surfels: " << competition.moved_surfels << "\n";
            if (cfg.pipeline == "joint" || cfg.pipeline == "topology") {
                std::cout << "Joint iterations/pruned models/energy: "
                          << competition.iterations << '/' << competition.pruned_models << '/'
                          << competition.final_energy << "\n";
            }
            std::cout << "Normal-guided dense candidates: " << dense_assignment.normal_guided_points
                      << ", established evidence: " << dense_assignment.established_guided_points
                      << ", applied: " << (normal_guidance_applied ? "yes" : "no") << "\n";
            std::cout << "Local-support dense candidates: " << dense_assignment.locally_supported_points << "\n";
            std::cout << "Topology-guided dense assignments: "
                      << dense_assignment.topology_guided_points << "\n";
            std::cout << "Boundary-rejected dense candidates: "
                      << dense_assignment.boundary_rejected_candidates << "\n";
            std::cout << "Active plane-side constraints: "
                      << dense_assignment.boundary_constraints << "\n";
            std::cout << "Maximum dense candidates per point: "
                      << dense_assignment.max_candidates_per_point << "\n";
            std::cout << "Safely absorbed small patches/points: "
                      << finalize_stats.absorbed_patches << '/' << finalize_stats.absorbed_points << "\n";
            std::cout << "Discarded sparse patch points: " << finalize_stats.discarded_sparse_points << "\n";
            std::cout << "Assignment margin: " << cfg.assign_margin << "\n";
            std::cout << "Plane patches: " << patches.size() << "\n";
            std::cout << "Output points: " << output_points << "\n";
            std::cout << "Extracted corners: " << corners.size() << "\n";
            std::cout << "Corner mode: " << cfg.corner_mode << "\n";
            if (cfg.corner_mode == "adjacency") {
                std::cout << "Finite intersections/local triples/ill-conditioned/max condition: "
                          << corner_result.finite_intersections.size() << '/'
                          << corner_result.local_triples << '/'
                          << corner_result.ill_conditioned_triples << '/'
                          << corner_result.maximum_accepted_condition << "\n";
            }
            std::cout << "Timing load/surfel/proposal (ms): "
                      << timings.load_ms << '/' << timings.surfel_build_ms << '/' << timings.proposal_ms << "\n";
            std::cout << "Timing global/merge1/competition/merge2 (ms): "
                      << timings.global_relations_ms << '/' << timings.initial_merge_ms << '/'
                      << timings.competition_ms << '/' << timings.final_merge_ms << "\n";
            std::cout << "Timing margin/dense/finalize/audit/corners/save (ms): "
                      << timings.assign_margin_ms << '/' << timings.dense_assignment_ms << '/'
                      << timings.finalize_ms << '/' << timings.audit_ms << '/' << timings.corners_ms << '/'
                      << timings.save_ms << "\n";
            std::cout << "Timing total (ms): " << timings.total_ms << "\n";
            std::cout << "Saved to " << fs::absolute(out_dir).string() << "\n";
        }
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}
