#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
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
#include <unordered_set>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/vcm_estimate_normals.h>

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

using CgalKernel = CGAL::Exact_predicates_inexact_constructions_kernel;

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
    bool edge_residual_proposals = false;
    bool binary_plane_output = false;
    std::string parallel_relations = "none";
    bool finite_boundary_postprocess = false;
    bool finite_support_region_optimization = false;
    std::string region_normal_estimator = "pca";
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

using SurfelNeighborhoods = std::vector<std::vector<int>>;

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
    bool requires_local_support = false;
    int pre_finalize_plane_id = -1;
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
    int final_plane = -1;
};

struct DenseAssignmentStats {
    std::size_t normal_guided_points = 0;
    std::size_t locally_supported_points = 0;
    std::size_t established_guided_points = 0;
    std::size_t edge_owner_guided_points = 0;
    std::size_t topology_guided_points = 0;
    std::size_t boundary_rejected_candidates = 0;
    std::size_t boundary_constraints = 0;
    std::size_t max_candidates_per_point = 0;
    std::size_t occupancy_rejected_candidates = 0;
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
    double relation_refinement_ms = 0.0;
    double assign_margin_ms = 0.0;
    double dense_assignment_ms = 0.0;
    double finalize_ms = 0.0;
    double boundary_postprocess_ms = 0.0;
    double support_region_ms = 0.0;
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

struct ResidualNoveltyStats {
    std::size_t generated = 0;
    std::size_t accepted = 0;
    std::size_t rejected = 0;
    double maximum_estimated_novel_points = 0.0;
};

struct RelationRefinementStats {
    std::size_t clusters = 0;
    std::size_t constrained_planes = 0;
    float maximum_cluster_spread_deg = 0.0f;
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

struct FiniteBoundaryPostprocessStats {
    std::size_t finite_intersections = 0;
    std::size_t directed_constraints = 0;
    std::size_t trusted_geometry_points = 0;
    std::size_t support_cells = 0;
    std::size_t boundary_candidate_points = 0;
    std::size_t finite_competitors = 0;
    std::size_t margin_rejected_points = 0;
    std::size_t corrected_points = 0;
    std::size_t rejected_without_local_support = 0;
    std::vector<TopologyCorrection> corrections;
};

struct FiniteSupportRegionDecision {
    PointT point;
    int from_plane = -1;
    int to_plane = -1;
    int component = -1;
    float from_energy = 0.0f;
    float to_energy = 0.0f;
    float from_support_distance = 0.0f;
    float to_support_distance = 0.0f;
    float normal_confidence = 0.0f;
    float from_residual = 0.0f;
    float to_residual = 0.0f;
    float from_normal_dot = 0.0f;
    float to_normal_dot = 0.0f;
    float node_retain_energy = 0.0f;
    float node_target_energy = 0.0f;
    bool candidate_from_missing_support = false;
    bool candidate_from_normal_conflict = false;
    bool owner_observation_ambiguous = false;
    bool accepted_by_residual = false;
    bool accepted_by_normal = false;
    int final_plane = -1;
};

struct FiniteSupportRegionStats {
    std::size_t trusted_geometry_points = 0;
    std::size_t occupied_support_cells = 0;
    std::size_t ambiguous_observation_cells = 0;
    std::size_t inferred_sampling_gap_cells = 0;
    std::size_t support_inconsistent_points = 0;
    std::size_t competition_nodes = 0;
    std::size_t competition_components = 0;
    std::size_t optimized_components = 0;
    std::size_t retained_components = 0;
    std::size_t mixed_surfel_points = 0;
    std::size_t local_geometry_rejections = 0;
    std::size_t support_preservation_rejections = 0;
    std::size_t corrected_points = 0;
    std::vector<FiniteSupportRegionDecision> decisions;
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
            << "  --proposal-initial-cap N  diagnostic initial proposal capacity; expands automatically\n"
            << "  --edge-residual-proposals  propose planes from unexplained persistent boundary surfels\n"
            << "  --binary-plane-output  write the large colored plane PLY in binary format\n"
            << "  --parallel-relations none|soft|hard  optional global parallel-normal refinement\n"
            << "  --finite-boundary-postprocess  correct dense points beyond observed finite intersections\n"
            << "  --finite-support-region-optimization  optimize connected finite-support competition regions\n"
            << "  --region-normal-estimator pca|pca-vcm-consensus  finite-region normal evidence mode\n"
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
        } else if (arg == "--proposal-initial-cap") {
            if (i + 1 >= args.size()) {
                throw std::runtime_error("Missing value after --proposal-initial-cap.");
            }
            cfg.max_planes = std::stoi(args[++i]);
            if (cfg.max_planes < 1) {
                throw std::runtime_error("--proposal-initial-cap must be positive.");
            }
        } else if (arg == "--edge-residual-proposals") {
            cfg.edge_residual_proposals = true;
        } else if (arg == "--binary-plane-output") {
            cfg.binary_plane_output = true;
        } else if (arg == "--parallel-relations") {
            if (i + 1 >= args.size()) throw std::runtime_error("Missing value after --parallel-relations.");
            cfg.parallel_relations = args[++i];
            if (cfg.parallel_relations != "none" && cfg.parallel_relations != "soft" &&
                cfg.parallel_relations != "hard") {
                throw std::runtime_error(
                    "Unknown parallel relation mode: " + cfg.parallel_relations +
                    ". Available: none, soft, hard.");
            }
        } else if (arg == "--finite-boundary-postprocess") {
            cfg.finite_boundary_postprocess = true;
        } else if (arg == "--finite-support-region-optimization") {
            cfg.finite_support_region_optimization = true;
        } else if (arg == "--region-normal-estimator") {
            if (i + 1 >= args.size()) throw std::runtime_error("--region-normal-estimator requires a value");
            cfg.region_normal_estimator = args[++i];
            if (cfg.region_normal_estimator != "pca" &&
                cfg.region_normal_estimator != "pca-vcm-consensus") {
                throw std::runtime_error(
                    "--region-normal-estimator must be pca or pca-vcm-consensus");
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
    if (cfg.finite_boundary_postprocess && cfg.finite_support_region_optimization) {
        throw std::runtime_error(
            "--finite-boundary-postprocess and --finite-support-region-optimization "
            "are independent ablations and cannot be enabled together.");
    }
    if (!cfg.finite_support_region_optimization && cfg.region_normal_estimator != "pca") {
        throw std::runtime_error(
            "The VCM consensus mode requires --finite-support-region-optimization.");
    }
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
    const SurfelNeighborhoods& neighborhoods,
    const Config& cfg) {
    const float edge_cos = std::cos(cfg.edge_angle_deg / 180.0f * 3.14159265358979323846f);
    for (std::size_t i = 0; i < surfels.size(); ++i) {
        for (int j : neighborhoods[i]) {
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

static SurfelNeighborhoods build_surfel_neighborhoods(
    const std::vector<Surfel>& surfels,
    const std::unordered_map<CellKey, int, CellKeyHash>& index_by_key) {
    SurfelNeighborhoods neighborhoods(surfels.size());
    for (std::size_t surfel_id = 0; surfel_id < surfels.size(); ++surfel_id) {
        std::vector<int>& out = neighborhoods[surfel_id];
        const CellKey base = surfels[surfel_id].key;
        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dz = -1; dz <= 1; ++dz) {
                    if (dx == 0 && dy == 0 && dz == 0) continue;
                    const CellKey key{base.x + dx, base.y + dy, base.z + dz};
                    const auto it = index_by_key.find(key);
                    if (it != index_by_key.end()) out.push_back(it->second);
                }
            }
        }
    }
    return neighborhoods;
}

static bool surfel_matches_plane(const Surfel& surfel, const PlanePatch& patch, const Config& cfg, float angle_cos, float distance_scale) {
    const float normal_dot = std::abs(surfel.normal.dot(patch.normal));
    if (normal_dot < angle_cos) return false;
    const float distance = std::abs(patch.normal.dot(surfel.centroid) + patch.d);
    return distance <= cfg.coplanar_distance * distance_scale;
}

static std::vector<PlanePatch> extract_seed_grown_patches(
    const std::vector<Surfel>& surfels,
    const SurfelNeighborhoods& neighborhoods,
    const Config& cfg,
    bool& capacity_exceeded,
    const std::vector<std::uint8_t>* eligible) {
    capacity_exceeded = false;
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
    const auto is_eligible = [&](int id) {
        return eligible == nullptr || (*eligible)[static_cast<std::size_t>(id)] != 0;
    };

    for (int seed : order) {
        if (!is_eligible(seed) || owner[static_cast<std::size_t>(seed)] >= 0 ||
            surfels[static_cast<std::size_t>(seed)].edge_like) continue;

        std::vector<int> seed_ids{seed};
        for (int neighbor : neighborhoods[static_cast<std::size_t>(seed)]) {
            if (!is_eligible(neighbor) || owner[static_cast<std::size_t>(neighbor)] >= 0 ||
                surfels[static_cast<std::size_t>(neighbor)].edge_like) continue;
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
            if (!is_eligible(id) || owner[static_cast<std::size_t>(id)] >= 0 ||
                touched.find(id) != touched.end()) return;
            if (!surfel_matches_plane(surfels[static_cast<std::size_t>(id)], patch, cfg, smooth_cos, 1.0f)) return;
            touched.insert(id);
            ids.push_back(id);
            if (can_expand && !surfels[static_cast<std::size_t>(id)].edge_like) q.push(id);
        };

        for (int id : seed_ids) try_add(id, true);
        while (!q.empty()) {
            const int current = q.front();
            q.pop();
            for (int neighbor : neighborhoods[static_cast<std::size_t>(current)]) {
                try_add(neighbor, true);
            }
        }

        for (int iter = 0; iter < std::max(cfg.grow_refine_iterations, 0); ++iter) {
            if (ids.size() < static_cast<std::size_t>(min_seed)) break;
            patch = fit_patch_from_surfels(ids, surfels);
            std::vector<int> frontier = ids;
            for (int current : frontier) {
                for (int neighbor : neighborhoods[static_cast<std::size_t>(current)]) {
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
        if (patches.size() > static_cast<std::size_t>(std::max(cfg.max_planes, 1))) {
            capacity_exceeded = true;
            patches.resize(static_cast<std::size_t>(std::max(cfg.max_planes, 1)));
            break;
        }
    }

    std::sort(patches.begin(), patches.end(), [](const PlanePatch& a, const PlanePatch& b) {
        return a.surfels.size() > b.surfels.size();
    });
    return patches;
}

static std::vector<PlanePatch> extract_all_seed_grown_patches(
    const std::vector<Surfel>& surfels,
    const SurfelNeighborhoods& neighborhoods,
    const Config& cfg,
    std::size_t& capacity_expansions,
    const std::vector<std::uint8_t>* eligible = nullptr) {
    capacity_expansions = 0;
    Config expanded_cfg = cfg;
    expanded_cfg.max_planes = std::max(expanded_cfg.max_planes, 1);
    const std::size_t minimum_patch = static_cast<std::size_t>(std::max(cfg.min_surfels, 6));
    const std::size_t maximum_possible = std::max<std::size_t>(1, surfels.size() / minimum_patch + 1);
    while (true) {
        bool capacity_exceeded = false;
        std::vector<PlanePatch> patches = extract_seed_grown_patches(
            surfels, neighborhoods, expanded_cfg, capacity_exceeded, eligible);
        if (!capacity_exceeded) return patches;

        ++capacity_expansions;
        const std::size_t current = static_cast<std::size_t>(expanded_cfg.max_planes);
        const std::size_t expanded = std::min(maximum_possible, std::max(current + 1, 2 * current));
        if (expanded <= current) {
            throw std::runtime_error("Plane proposal capacity could not be expanded safely.");
        }
        expanded_cfg.max_planes = static_cast<int>(std::min<std::size_t>(
            expanded, static_cast<std::size_t>(std::numeric_limits<int>::max())));
    }
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
                const bool requires_local_support =
                    patches[i].requires_local_support || patches[j].requires_local_support;
                patches[i] = fit_patch_from_surfels(patches[i].surfels, surfels);
                patches[i].requires_local_support = requires_local_support;
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

static std::vector<int> build_final_surfel_labels(
    const std::vector<PlanePatch>& patches,
    std::size_t surfel_count,
    const std::unordered_map<CellKey, int, CellKeyHash>& index_by_key,
    const Bounds& bounds,
    float surfel_size) {
    // Boyer-Moore finds a possible strict majority without allocating a
    // surfel-by-plane matrix. A second pass verifies it. Mixed cells remain
    // unknown instead of inheriting the pre-optimization owner.
    std::vector<int> candidate(surfel_count, -1);
    std::vector<int> balance(surfel_count, 0);
    std::vector<std::uint32_t> total(surfel_count, 0);
    for (std::size_t plane_id = 0; plane_id < patches.size(); ++plane_id) {
        for (const PointT& point : patches[plane_id].output->points) {
            const auto found = index_by_key.find(point_key(point, bounds, surfel_size));
            if (found == index_by_key.end()) continue;
            const std::size_t surfel_id = static_cast<std::size_t>(found->second);
            ++total[surfel_id];
            if (balance[surfel_id] == 0) {
                candidate[surfel_id] = static_cast<int>(plane_id);
                balance[surfel_id] = 1;
            } else if (candidate[surfel_id] == static_cast<int>(plane_id)) {
                ++balance[surfel_id];
            } else {
                --balance[surfel_id];
            }
        }
    }
    std::vector<std::uint32_t> candidate_count(surfel_count, 0);
    for (std::size_t plane_id = 0; plane_id < patches.size(); ++plane_id) {
        for (const PointT& point : patches[plane_id].output->points) {
            const auto found = index_by_key.find(point_key(point, bounds, surfel_size));
            if (found == index_by_key.end()) continue;
            const std::size_t surfel_id = static_cast<std::size_t>(found->second);
            if (candidate[surfel_id] == static_cast<int>(plane_id)) {
                ++candidate_count[surfel_id];
            }
        }
    }
    for (std::size_t surfel_id = 0; surfel_id < surfel_count; ++surfel_id) {
        if (candidate_count[surfel_id] * 2 <= total[surfel_id]) candidate[surfel_id] = -1;
    }
    return candidate;
}

static std::vector<std::vector<BoundaryConstraint>> build_boundary_constraints(
    const std::vector<PlanePatch>& patches,
    const std::vector<Surfel>& surfels,
    const SurfelNeighborhoods& neighborhoods,
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
        for (int neighbor : neighborhoods[surfel_id]) {
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
    const SurfelNeighborhoods& neighborhoods,
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
        for (int neighbor : neighborhoods[surfel_id]) {
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
            const bool requires_local_support = patches[patch_id].requires_local_support;
            patches[patch_id] = fit_patch_from_surfels(ids_by_patch[patch_id], surfels);
            patches[patch_id].requires_local_support = requires_local_support;
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
    const SurfelNeighborhoods& neighborhoods,
    const Config& cfg) {
    CompetitionStats stats;
    if (patches.empty() || surfels.empty()) return stats;

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
            for (int neighbor : neighborhoods[surfel_id]) {
                if (labels[static_cast<std::size_t>(neighbor)] >= 0) best_cost += 0.5f * smooth_weight;
            }

            for (std::size_t patch_id = 0; patch_id < patches.size(); ++patch_id) {
                if (!active[patch_id] || patches[patch_id].surfels.empty()) continue;
                float cost = joint_surfel_data_cost(surfel, patches[patch_id], cfg);
                if (!std::isfinite(cost)) continue;
                for (int neighbor : neighborhoods[surfel_id]) {
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
            const bool requires_local_support = patches[patch_id].requires_local_support;
            patches[patch_id] = fit_patch_from_surfels(ids_by_patch[patch_id], surfels);
            patches[patch_id].requires_local_support = requires_local_support;
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
        patch.requires_local_support = patches[patch_id].requires_local_support;
        update_patch_envelope(patch, surfels);
        if (valid_shape(patch, cfg)) retained.push_back(std::move(patch));
    }
    std::sort(retained.begin(), retained.end(), [](const PlanePatch& a, const PlanePatch& b) {
        return a.surfels.size() > b.surfels.size();
    });
    patches.swap(retained);
    return stats;
}

static ResidualNoveltyStats retain_novel_residual_proposals(
    std::vector<PlanePatch>& residual_patches,
    const std::vector<PlanePatch>& existing_patches,
    const CloudPtr& cloud,
    const std::vector<Surfel>& surfels,
    const std::unordered_map<CellKey, int, CellKeyHash>& index_by_key,
    const Bounds& bounds,
    const Config& cfg) {
    ResidualNoveltyStats stats;
    stats.generated = residual_patches.size();
    if (residual_patches.empty() || cloud->empty()) return stats;

    const std::size_t sample_limit = 200000;
    const std::size_t stride = std::max<std::size_t>(1, cloud->size() / sample_limit);
    std::vector<std::size_t> sample_indices;
    sample_indices.reserve(std::min(cloud->size(), sample_limit + 1));
    for (std::size_t point_id = 0; point_id < cloud->size(); point_id += stride) {
        sample_indices.push_back(point_id);
    }

    const float existing_margin = std::max(24.0f * cfg.surfel_size, 2.0f);
    std::vector<std::uint8_t> explained(sample_indices.size(), 0);
    for (std::size_t sample_id = 0; sample_id < sample_indices.size(); ++sample_id) {
        const PointT& point = cloud->points[sample_indices[sample_id]];
        const Eigen::Vector3f q(point.x, point.y, point.z);
        for (const PlanePatch& patch : existing_patches) {
            if (std::abs(patch.normal.dot(q) + patch.d) > cfg.assign_distance) continue;
            if (!point_inside_patch_envelope(patch, q, existing_margin)) continue;
            explained[sample_id] = 1;
            break;
        }
    }

    std::vector<PlanePatch> retained;
    retained.reserve(residual_patches.size());
    std::vector<std::uint8_t> member(surfels.size(), 0);
    for (PlanePatch& patch : residual_patches) {
        std::fill(member.begin(), member.end(), static_cast<std::uint8_t>(0));
        for (int surfel_id : patch.surfels) member[static_cast<std::size_t>(surfel_id)] = 1;

        std::size_t novel_samples = 0;
        for (std::size_t sample_id = 0; sample_id < sample_indices.size(); ++sample_id) {
            const PointT& point = cloud->points[sample_indices[sample_id]];
            const Eigen::Vector3f q(point.x, point.y, point.z);
            if (std::abs(patch.normal.dot(q) + patch.d) > cfg.assign_distance) continue;
            if (!point_inside_patch_envelope(patch, q, cfg.assign_margin)) continue;

            const CellKey key = point_key(point, bounds, cfg.surfel_size);
            bool locally_supported = false;
            for (int dx = -1; dx <= 1 && !locally_supported; ++dx) {
                for (int dy = -1; dy <= 1 && !locally_supported; ++dy) {
                    for (int dz = -1; dz <= 1; ++dz) {
                        const auto neighbor = index_by_key.find({key.x + dx, key.y + dy, key.z + dz});
                        if (neighbor != index_by_key.end() &&
                            member[static_cast<std::size_t>(neighbor->second)] != 0) {
                            locally_supported = true;
                            break;
                        }
                    }
                }
            }
            if (locally_supported && explained[sample_id] == 0) ++novel_samples;
        }

        const double estimated_novel_points = sample_indices.empty() ? 0.0 :
            static_cast<double>(novel_samples) * static_cast<double>(cloud->size()) /
                static_cast<double>(sample_indices.size());
        stats.maximum_estimated_novel_points =
            std::max(stats.maximum_estimated_novel_points, estimated_novel_points);
        if (estimated_novel_points >= static_cast<double>(std::max(cfg.min_output_points, 1))) {
            retained.push_back(std::move(patch));
            ++stats.accepted;
        } else {
            ++stats.rejected;
        }
    }
    residual_patches.swap(retained);
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
    const SurfelNeighborhoods& neighborhoods,
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
    // Region optimization is a true final-stage postprocess: it must start
    // from the same dense assignment as the topology baseline.  Disabling the
    // embedded guidance here silently changed the ablation input (notably on
    // crossing_planes) even when the region optimizer made zero corrections.
    // The older finite-boundary pass remains a replacement ablation.
    const bool dense_boundary_guidance = topology_aware && !cfg.finite_boundary_postprocess;
    const std::vector<std::vector<BoundaryConstraint>> boundary_constraints = dense_boundary_guidance
        ? build_boundary_constraints(patches, surfels, neighborhoods, cfg)
        : std::vector<std::vector<BoundaryConstraint>>(patches.size());
    for (const auto& constraints : boundary_constraints) stats.boundary_constraints += constraints.size();

    // A tiny support fragment can have a locally coherent normal while still
    // being a spurious boundary model.  Infer which owners are established
    // from this scene's support distribution, using the same robust lower-tail
    // rule as the directed boundary constraints.
    std::vector<float> owner_log_support;
    owner_log_support.reserve(patches.size());
    for (const PlanePatch& patch : patches) {
        owner_log_support.push_back(
            std::log(static_cast<float>(std::max<std::size_t>(patch.surfels.size(), 1))));
    }
    const float owner_median_log_support = median_value(owner_log_support);
    std::vector<float> owner_support_deviations;
    owner_support_deviations.reserve(owner_log_support.size());
    for (float value : owner_log_support) {
        owner_support_deviations.push_back(std::abs(value - owner_median_log_support));
    }
    const float owner_support_sigma = 1.4826f * median_value(std::move(owner_support_deviations));
    const float owner_authority_floor = owner_median_log_support - 3.0f * owner_support_sigma;

    struct DenseCandidate {
        int id = -1;
        float distance = std::numeric_limits<float>::max();
        float normal_dot = 1.0f;
        int local_support = 0;
        int strict_local_support = 0;
        bool inside_tight = false;
        bool requires_local_support = false;
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
        const int authoritative_local_owner = local_surfel != nullptr && local_owner >= 0 &&
                owner_log_support[static_cast<std::size_t>(local_owner)] >= owner_authority_floor
            ? local_owner : -1;
        const int established_local_owner = local_surfel != nullptr && !local_surfel->edge_like
            ? authoritative_local_owner : -1;
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
            candidate.requires_local_support = patches[i].requires_local_support;
        }
        std::size_t candidate_count = candidates.size();
        stats.max_candidates_per_point = std::max(stats.max_candidates_per_point, candidate_count);
        if (candidate_count == 0) continue;

        // Only ambiguous points pay for the local support lookup.  The lookup is
        // in the original 3-D surfel grid, so it cannot be fooled by a large 2-D
        // patch rectangle extending through an unrelated surface.
        const bool needs_local_support = candidate_count > 1 ||
            std::any_of(candidates.begin(), candidates.end(), [](const DenseCandidate& candidate) {
                return candidate.requires_local_support;
            });
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
                                if (std::abs(dx) <= 1 && std::abs(dy) <= 1 && std::abs(dz) <= 1) {
                                    ++candidates[i].strict_local_support;
                                }
                                break;
                            }
                        }
                    }
                }
            }
        }

        const std::size_t candidates_before_occupancy = candidates.size();
        candidates.erase(
            std::remove_if(candidates.begin(), candidates.end(), [](const DenseCandidate& candidate) {
                return candidate.requires_local_support && candidate.strict_local_support == 0;
            }),
            candidates.end());
        stats.occupancy_rejected_candidates += candidates_before_occupancy - candidates.size();
        candidate_count = candidates.size();
        if (candidate_count == 0) continue;

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

        // A non-edge surfel is an observed, locally smooth piece of one plane.
        // Do not let a different infinite plane steal its dense points merely
        // because the two fitted planes are very close near an intersection.
        // The earlier conservative rule only corrected loose-envelope or
        // unsupported candidates; that left thin one-cell spill bands whenever
        // the wrong plane's rectangular envelope and neighborhood overlapped
        // the true face. This override is deliberately limited to topology
        // mode and to an established, normal-consistent local owner. Mixed
        // edge surfels therefore keep the original pointwise decision.
        if (topology_aware && established_local_owner >= 0 &&
            candidates[selected].id != established_local_owner &&
            candidates[selected].normal_dot < edge_cos) {
            std::size_t owner_candidate = candidate_count;
            for (std::size_t i = 0; i < candidate_count; ++i) {
                if (candidates[i].id != established_local_owner) continue;
                if (candidates[i].normal_dot < smooth_cos) continue;
                const PlanePatch& owner_patch = patches[static_cast<std::size_t>(candidates[i].id)];
                const PlanePatch& selected_patch = patches[static_cast<std::size_t>(candidates[selected].id)];
                const float fit_uncertainty = std::max(
                    3.0f * std::hypot(owner_patch.rms_distance, selected_patch.rms_distance),
                    1e-6f * cfg.surfel_size);
                if (candidates[i].distance > candidates[selected].distance + fit_uncertainty) continue;
                owner_candidate = i;
                break;
            }
            if (owner_candidate < candidate_count) {
                selected = owner_candidate;
                ++stats.normal_guided_points;
                ++stats.established_guided_points;
            }
        }

        // An edge cell is not automatically unreliable. At a measured sharp
        // boundary its PCA normal can still agree almost exactly with one
        // established face while rejecting the pointwise nearest infinite
        // plane. Previously every edge cell was excluded from the local-owner
        // safeguard, which created thin, connected stripes inside the true
        // face whenever both rectangular envelopes overlapped. Admit the
        // observed owner only when normal, finite envelope, neighborhood
        // support, and fitted uncertainty all agree.
        if (topology_aware && enable_normal_guidance && local_surfel != nullptr &&
            local_surfel->edge_like && authoritative_local_owner >= 0 &&
            candidates[selected].id != authoritative_local_owner &&
            candidates[selected].inside_tight &&
            candidates[selected].normal_dot < edge_cos) {
            std::size_t owner_candidate = candidate_count;
            for (std::size_t i = 0; i < candidate_count; ++i) {
                if (candidates[i].id != authoritative_local_owner) continue;
                if (!candidates[i].inside_tight) continue;
                const bool support_dominates = candidates[i].normal_dot >= smooth_cos &&
                    candidates[i].local_support > candidates[selected].local_support &&
                    candidates[i].strict_local_support >= candidates[selected].strict_local_support;
                const bool relative_normal_dominates = candidates[i].normal_dot >= edge_cos &&
                    candidates[i].normal_dot - candidates[selected].normal_dot >= 1.0f - edge_cos &&
                    candidates[i].strict_local_support >= cfg.min_plane_local_support;
                if (!support_dominates && !relative_normal_dominates) continue;
                const PlanePatch& owner_patch = patches[static_cast<std::size_t>(candidates[i].id)];
                const PlanePatch& selected_patch = patches[static_cast<std::size_t>(candidates[selected].id)];
                const float fit_uncertainty = std::max(
                    3.0f * std::hypot(owner_patch.rms_distance, selected_patch.rms_distance),
                    1e-6f * cfg.surfel_size);
                if (candidates[i].distance > candidates[selected].distance + fit_uncertainty) continue;
                owner_candidate = i;
                break;
            }
            if (owner_candidate < candidate_count) {
                stats.corrections.push_back({
                    p,
                    candidates[selected].id,
                    candidates[owner_candidate].id
                });
                selected = owner_candidate;
                ++stats.normal_guided_points;
                ++stats.edge_owner_guided_points;
                ++stats.locally_supported_points;
            }
        }

        const auto crossed_neighbor = [&](const DenseCandidate& candidate) {
            if (!dense_boundary_guidance || established_local_owner == candidate.id) return -1;
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

static RelationRefinementStats refine_parallel_relations(
    std::vector<PlanePatch>& patches,
    const std::vector<Surfel>& surfels,
    const Config& cfg) {
    RelationRefinementStats stats;
    if (cfg.parallel_relations == "none" || patches.size() < 2) return stats;
    const float relation_angle_deg = std::max(0.25f, 0.2f * cfg.merge_angle_deg);
    const float relation_cos = std::cos(
        relation_angle_deg / 180.0f * 3.14159265358979323846f);
    std::vector<std::vector<std::size_t>> clusters;
    clusters.reserve(patches.size());
    for (std::size_t patch_id = 0; patch_id < patches.size(); ++patch_id) {
        clusters.push_back({patch_id});
    }

    while (true) {
        std::size_t best_a = clusters.size();
        std::size_t best_b = clusters.size();
        float best_spread = std::numeric_limits<float>::max();
        for (std::size_t a = 0; a < clusters.size(); ++a) {
            for (std::size_t b = a + 1; b < clusters.size(); ++b) {
                bool compatible = true;
                float spread = 0.0f;
                for (std::size_t first : clusters[a]) {
                    for (std::size_t second : clusters[b]) {
                        const float dot = std::clamp(
                            std::abs(patches[first].normal.dot(patches[second].normal)), 0.0f, 1.0f);
                        if (dot < relation_cos) {
                            compatible = false;
                            break;
                        }
                        spread = std::max(
                            spread,
                            std::acos(dot) * 180.0f / 3.14159265358979323846f);
                    }
                    if (!compatible) break;
                }
                if (compatible && spread < best_spread) {
                    best_spread = spread;
                    best_a = a;
                    best_b = b;
                }
            }
        }
        if (best_a == clusters.size()) break;
        clusters[best_a].insert(
            clusters[best_a].end(), clusters[best_b].begin(), clusters[best_b].end());
        clusters.erase(clusters.begin() + static_cast<std::ptrdiff_t>(best_b));
    }

    for (const std::vector<std::size_t>& cluster : clusters) {
        if (cluster.size() < 2) continue;
        Eigen::Vector3d weighted_normal = Eigen::Vector3d::Zero();
        const Eigen::Vector3f reference = patches[cluster.front()].normal;
        float cluster_spread = 0.0f;
        for (std::size_t patch_id : cluster) {
            const float sign = reference.dot(patches[patch_id].normal) >= 0.0f ? 1.0f : -1.0f;
            const double weight = static_cast<double>(std::max<std::size_t>(patches[patch_id].surfels.size(), 1));
            weighted_normal += weight * sign * patches[patch_id].normal.cast<double>();
            const float dot = std::clamp(
                std::abs(reference.dot(patches[patch_id].normal)), 0.0f, 1.0f);
            cluster_spread = std::max(
                cluster_spread,
                std::acos(dot) * 180.0f / 3.14159265358979323846f);
        }
        if (weighted_normal.squaredNorm() <= 1e-12) continue;
        const Eigen::Vector3f family_normal = weighted_normal.normalized().cast<float>();
        for (std::size_t patch_id : cluster) {
            PlanePatch& patch = patches[patch_id];
            const float sign = family_normal.dot(patch.normal) >= 0.0f ? 1.0f : -1.0f;
            const Eigen::Vector3f target = sign * family_normal;
            patch.normal = cfg.parallel_relations == "hard"
                ? target
                : (patch.normal + target).normalized();
            patch.d = -patch.normal.dot(patch.centroid);
            update_patch_envelope(patch, surfels);
        }
        ++stats.clusters;
        stats.constrained_planes += cluster.size();
        stats.maximum_cluster_spread_deg = std::max(stats.maximum_cluster_spread_deg, cluster_spread);
    }
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

static std::vector<CornerCandidate> merge_corner_candidates_complete_link(
    std::vector<CornerCandidate> candidates,
    float merge_radius) {
    const auto position_less = [](const CornerCandidate& a, const CornerCandidate& b) {
        if (a.position.x() != b.position.x()) return a.position.x() < b.position.x();
        if (a.position.y() != b.position.y()) return a.position.y() < b.position.y();
        return a.position.z() < b.position.z();
    };
    std::sort(candidates.begin(), candidates.end(), position_less);
    std::vector<std::vector<std::size_t>> clusters;
    clusters.reserve(candidates.size());
    for (std::size_t i = 0; i < candidates.size(); ++i) clusters.push_back({i});
    const float merge_sq = merge_radius * merge_radius;

    while (true) {
        std::size_t best_a = clusters.size();
        std::size_t best_b = clusters.size();
        float best_complete_distance = std::numeric_limits<float>::max();
        for (std::size_t a = 0; a < clusters.size(); ++a) {
            for (std::size_t b = a + 1; b < clusters.size(); ++b) {
                float complete_distance = 0.0f;
                bool mergeable = true;
                for (std::size_t first : clusters[a]) {
                    for (std::size_t second : clusters[b]) {
                        complete_distance = std::max(
                            complete_distance,
                            (candidates[first].position - candidates[second].position).squaredNorm());
                        if (complete_distance > merge_sq) {
                            mergeable = false;
                            break;
                        }
                    }
                    if (!mergeable) break;
                }
                if (mergeable && complete_distance < best_complete_distance) {
                    best_complete_distance = complete_distance;
                    best_a = a;
                    best_b = b;
                }
            }
        }
        if (best_a == clusters.size()) break;
        clusters[best_a].insert(
            clusters[best_a].end(), clusters[best_b].begin(), clusters[best_b].end());
        clusters.erase(clusters.begin() + static_cast<std::ptrdiff_t>(best_b));
    }

    std::vector<CornerCandidate> merged;
    merged.reserve(clusters.size());
    for (const std::vector<std::size_t>& cluster : clusters) {
        CornerCandidate representative = candidates[cluster.front()];
        for (std::size_t candidate_id : cluster) {
            const CornerCandidate& candidate = candidates[candidate_id];
            if (candidate.support > representative.support ||
                (candidate.support == representative.support && position_less(candidate, representative))) {
                representative = candidate;
            }
        }
        merged.push_back(representative);
    }
    std::sort(merged.begin(), merged.end(), [&](const CornerCandidate& a, const CornerCandidate& b) {
        if (a.support != b.support) return a.support > b.support;
        return position_less(a, b);
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
    const SurfelNeighborhoods& neighborhoods,
    const Config& cfg,
    const std::vector<int>* final_labels = nullptr) {
    if (patches.size() < 2 || surfels.empty()) return {};
    const std::vector<int> labels = final_labels != nullptr
        ? *final_labels : build_surfel_labels(patches, surfels.size());
    const std::size_t plane_count = patches.size();
    std::vector<std::vector<Eigen::Vector3f>> interface_samples(plane_count * plane_count);

    for (std::size_t surfel_id = 0; surfel_id < surfels.size(); ++surfel_id) {
        const int label_a = labels[surfel_id];
        if (label_a < 0) continue;
        for (int neighbor : neighborhoods[surfel_id]) {
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

static FiniteBoundaryPostprocessStats correct_points_from_raw_geometry(
    const CloudPtr& cloud,
    std::vector<PlanePatch>& patches,
    const std::vector<Surfel>& surfels,
    const SurfelNeighborhoods& neighborhoods,
    const Config& cfg) {
    FiniteBoundaryPostprocessStats stats;
    if (patches.size() < 2 || cloud->empty()) return stats;

    // Existing adjacency is used only as a cheap pair proposal.  Every decision
    // below -- trusted support, crossing/termination, and finite-face cost -- is
    // rebuilt from the raw coordinates without consulting their current labels.
    const std::vector<FinitePlaneIntersection> intersections =
        build_finite_plane_intersections(patches, surfels, neighborhoods, cfg);
    stats.finite_intersections = intersections.size();
    if (intersections.empty()) return stats;

    struct GridKey {
        int u = 0;
        int v = 0;
        bool operator==(const GridKey& other) const { return u == other.u && v == other.v; }
    };
    struct GridKeyHash {
        std::size_t operator()(const GridKey& key) const {
            return static_cast<std::size_t>(key.u) * 73856093u ^
                static_cast<std::size_t>(key.v) * 19349663u;
        }
    };
    struct RawSupportGrid {
        Eigen::Vector3f axis_u{1.0f, 0.0f, 0.0f};
        Eigen::Vector3f axis_v{0.0f, 1.0f, 0.0f};
        float fit_sigma = 0.0f;
        float point_sigma = 0.0f;
        std::unordered_set<GridKey, GridKeyHash> occupied;
    };

    const float grid_size = 0.5f * cfg.surfel_size;
    const float envelope_margin = 2.0f * cfg.surfel_size;
    std::vector<RawSupportGrid> grids(patches.size());
    for (std::size_t patch_id = 0; patch_id < patches.size(); ++patch_id) {
        plane_basis(patches[patch_id], grids[patch_id].axis_u, grids[patch_id].axis_v);
        grids[patch_id].fit_sigma = std::max(
            patches[patch_id].rms_distance, 1e-4f * cfg.surfel_size);
        grids[patch_id].point_sigma = std::max(
            grids[patch_id].fit_sigma, 0.05f * cfg.surfel_size);
        grids[patch_id].occupied.reserve(
            std::max<std::size_t>(patches[patch_id].surfels.size() * 4, 16));
    }

    const auto support_key = [&](std::size_t patch_id, const Eigen::Vector3f& q) {
        const RawSupportGrid& grid = grids[patch_id];
        const Eigen::Vector3f centered = q - patches[patch_id].centroid;
        return GridKey{
            static_cast<int>(std::floor(centered.dot(grid.axis_u) / grid_size)),
            static_cast<int>(std::floor(centered.dot(grid.axis_v) / grid_size))
        };
    };
    const auto inside_envelope_fast = [&](std::size_t patch_id, const Eigen::Vector3f& q) {
        const PlanePatch& patch = patches[patch_id];
        const RawSupportGrid& grid = grids[patch_id];
        const Eigen::Vector3f centered = q - patch.centroid;
        const float u = centered.dot(grid.axis_u);
        const float v = centered.dot(grid.axis_v);
        return u >= patch.min_u - envelope_margin && u <= patch.max_u + envelope_margin &&
            v >= patch.min_v - envelope_margin && v <= patch.max_v + envelope_margin;
    };

    // A raw point contributes to a support region only when exactly one plane
    // explains it within model uncertainty.  Points near an intersection remain
    // deliberately unlabelled, creating an uncertainty band rather than a fake
    // boundary observation.
    for (const PointT& point : cloud->points) {
        const Eigen::Vector3f q(point.x, point.y, point.z);
        std::size_t best = patches.size();
        float best_z = std::numeric_limits<float>::infinity();
        float second_z = std::numeric_limits<float>::infinity();
        for (std::size_t patch_id = 0; patch_id < patches.size(); ++patch_id) {
            if (!inside_envelope_fast(patch_id, q)) continue;
            const float residual = std::abs(patches[patch_id].normal.dot(q) + patches[patch_id].d);
            const float z = residual / grids[patch_id].point_sigma;
            if (z < best_z) {
                second_z = best_z;
                best_z = z;
                best = patch_id;
            } else if (z < second_z) {
                second_z = z;
            }
        }
        if (best == patches.size() || best_z > 3.0f ||
            second_z < 3.0f || second_z < best_z + 1.5f) {
            continue;
        }
        grids[best].occupied.insert(support_key(best, q));
        ++stats.trusted_geometry_points;
    }
    for (const RawSupportGrid& grid : grids) stats.support_cells += grid.occupied.size();

    const auto cell_center = [&](std::size_t patch_id, const GridKey& key) {
        return patches[patch_id].centroid +
            (static_cast<float>(key.u) + 0.5f) * grid_size * grids[patch_id].axis_u +
            (static_cast<float>(key.v) + 0.5f) * grid_size * grids[patch_id].axis_v;
    };
    struct RawBoundarySegment {
        int owner = -1;
        int other = -1;
        const FinitePlaneIntersection* edge = nullptr;
        float min_t = 0.0f;
        float max_t = 0.0f;
        float expected_sign = 1.0f;
        float side_tolerance = 0.0f;
    };
    std::vector<std::vector<RawBoundarySegment>> constraints(patches.size());
    // Evidence for a finite intersection must actually approach the line.
    // A broad surfel-neighborhood band bridges separate terraces and creates
    // a mathematically valid but physically unobserved infinite-plane crossing.
    // Two occupancy cells retain tolerance for sampling gaps without spanning
    // a distinct nearby surface.
    const float local_band = 2.0f * grid_size;
    const float endpoint_margin = 4.0f * cfg.surfel_size;

    for (const FinitePlaneIntersection& edge : intersections) {
        const auto side_bins = [&](int plane_id, int other_id) {
            std::unordered_map<int, std::uint8_t> bins;
            const PlanePatch& plane = patches[static_cast<std::size_t>(plane_id)];
            const PlanePatch& other = patches[static_cast<std::size_t>(other_id)];
            const float sine = plane.normal.cross(other.normal).norm();
            if (sine <= 1e-4f) return bins;
            const float ambiguity_band = std::max(
                3.0f * std::hypot(
                    grids[static_cast<std::size_t>(plane_id)].point_sigma,
                    grids[static_cast<std::size_t>(other_id)].point_sigma) / sine,
                grid_size);
            for (const GridKey& key : grids[static_cast<std::size_t>(plane_id)].occupied) {
                const Eigen::Vector3f q = cell_center(static_cast<std::size_t>(plane_id), key);
                const float t = edge.direction.dot(q - edge.origin);
                if (t < edge.min_t - endpoint_margin || t > edge.max_t + endpoint_margin) continue;
                const float side = (other.normal.dot(q) + other.d) / sine;
                if (std::abs(side) > local_band || std::abs(side) <= ambiguity_band) continue;
                bins[static_cast<int>(std::floor(t / grid_size))] |= side > 0.0f ? 1 : 2;
            }
            return bins;
        };

        const std::unordered_map<int, std::uint8_t> bins_a =
            side_bins(edge.plane_a, edge.plane_b);
        const std::unordered_map<int, std::uint8_t> bins_b =
            side_bins(edge.plane_b, edge.plane_a);
        const int minimum_bin = static_cast<int>(std::floor((edge.min_t - endpoint_margin) / grid_size));
        const int maximum_bin = static_cast<int>(std::floor((edge.max_t + endpoint_margin) / grid_size));

        const auto add_segments = [&](int owner, int other,
                const std::unordered_map<int, std::uint8_t>& owner_bins,
                const std::unordered_map<int, std::uint8_t>& other_bins) {
            const float sine = patches[static_cast<std::size_t>(owner)].normal.cross(
                patches[static_cast<std::size_t>(other)].normal).norm();
            if (sine <= 1e-4f) return;
            const float side_tolerance = std::max(
                3.0f * std::hypot(
                    grids[static_cast<std::size_t>(owner)].fit_sigma,
                    grids[static_cast<std::size_t>(other)].fit_sigma) / sine,
                1e-6f * cfg.surfel_size);
            int run_begin = 0;
            int run_end = -1;
            std::uint8_t run_sign = 0;
            const auto flush_run = [&]() {
                if (run_sign == 0 || run_end - run_begin + 1 < 2) return;
                constraints[static_cast<std::size_t>(owner)].push_back({
                    owner,
                    other,
                    &edge,
                    static_cast<float>(run_begin) * grid_size,
                    static_cast<float>(run_end + 1) * grid_size,
                    run_sign == 1 ? 1.0f : -1.0f,
                    side_tolerance
                });
                ++stats.directed_constraints;
            };
            for (int bin = minimum_bin; bin <= maximum_bin; ++bin) {
                std::uint8_t owner_evidence = 0;
                std::uint8_t other_evidence = 0;
                for (int offset = -1; offset <= 1; ++offset) {
                    const auto owner_value = owner_bins.find(bin + offset);
                    if (owner_value != owner_bins.end()) owner_evidence |= owner_value->second;
                    const auto other_value = other_bins.find(bin + offset);
                    if (other_value != other_bins.end()) other_evidence |= other_value->second;
                }
                const std::uint8_t state =
                    other_evidence != 0 && (owner_evidence == 1 || owner_evidence == 2)
                    ? owner_evidence : 0;
                if (state == run_sign && state != 0) {
                    run_end = bin;
                    continue;
                }
                flush_run();
                run_sign = state;
                run_begin = bin;
                run_end = bin;
            }
            flush_run();
        };
        add_segments(edge.plane_a, edge.plane_b, bins_a, bins_b);
        add_segments(edge.plane_b, edge.plane_a, bins_b, bins_a);
    }
    if (stats.directed_constraints == 0) return stats;

    const auto support_distance = [&](std::size_t plane_id, const Eigen::Vector3f& q) {
        const RawSupportGrid& grid = grids[plane_id];
        const PlanePatch& patch = patches[plane_id];
        const Eigen::Vector3f centered = q - patch.centroid;
        const float u = centered.dot(grid.axis_u) / grid_size;
        const float v = centered.dot(grid.axis_v) / grid_size;
        const int base_u = static_cast<int>(std::floor(u));
        const int base_v = static_cast<int>(std::floor(v));
        float best = std::numeric_limits<float>::infinity();
        constexpr int maximum_ring = 8;
        for (int du = -maximum_ring; du <= maximum_ring; ++du) {
            for (int dv = -maximum_ring; dv <= maximum_ring; ++dv) {
                if (grid.occupied.find({base_u + du, base_v + dv}) == grid.occupied.end()) continue;
                const float delta_u = u - (static_cast<float>(base_u + du) + 0.5f);
                const float delta_v = v - (static_cast<float>(base_v + dv) + 0.5f);
                best = std::min(best, std::sqrt(delta_u * delta_u + delta_v * delta_v) * grid_size);
            }
        }
        return std::max(0.0f, best - 0.70710678f * grid_size);
    };
    const auto finite_energy = [&](std::size_t plane_id, const Eigen::Vector3f& q, float& lateral) {
        lateral = support_distance(plane_id, q);
        if (!std::isfinite(lateral)) return std::numeric_limits<float>::infinity();
        const float residual = std::abs(patches[plane_id].normal.dot(q) + patches[plane_id].d);
        const float normal_term = residual / grids[plane_id].point_sigma;
        // Both terms are orthogonal components of the same Euclidean distance
        // to a finite planar support.  Scaling the lateral term by the much
        // coarser occupancy cell would systematically favor an infinite plane
        // just beyond its observed boundary.
        const float lateral_term = lateral / grids[plane_id].point_sigma;
        return normal_term * normal_term + lateral_term * lateral_term;
    };
    const auto violation = [&](const Eigen::Vector3f& q, const RawBoundarySegment& segment) {
        const PlanePatch& owner = patches[static_cast<std::size_t>(segment.owner)];
        const PlanePatch& other = patches[static_cast<std::size_t>(segment.other)];
        const float residual = owner.normal.dot(q) + owner.d;
        const Eigen::Vector3f projected = q - residual * owner.normal;
        const float t = segment.edge->direction.dot(projected - segment.edge->origin);
        if (t < segment.min_t || t > segment.max_t) return 0.0f;
        const float sine = owner.normal.cross(other.normal).norm();
        if (sine <= 1e-4f) return 0.0f;
        const float side = (other.normal.dot(projected) + other.d) / sine;
        // A directed half-space is valid only in the local cross-section that
        // observed this boundary.  Extending it across a concave face or a
        // face with a hole lets one side of the hole erase the opposite side.
        if (std::abs(side) > local_band) return 0.0f;
        return std::max(0.0f, -segment.expected_sign * side - segment.side_tolerance);
    };

    std::vector<std::vector<PointT>> incoming(patches.size());
    for (std::size_t owner = 0; owner < patches.size(); ++owner) {
        if (constraints[owner].empty()) continue;
        CloudPtr retained(new CloudT);
        retained->reserve(patches[owner].output->size());
        for (const PointT& point : patches[owner].output->points) {
            const Eigen::Vector3f q(point.x, point.y, point.z);
            float largest_owner_violation = 0.0f;
            for (const RawBoundarySegment& segment : constraints[owner]) {
                largest_owner_violation = std::max(
                    largest_owner_violation, violation(q, segment));
            }
            if (largest_owner_violation <= 0.0f) {
                retained->push_back(point);
                continue;
            }
            ++stats.boundary_candidate_points;

            float owner_lateral = 0.0f;
            float owner_energy = finite_energy(owner, q, owner_lateral);
            // Raster cells deliberately cover a finite area and can otherwise
            // hide sub-cell spill.  An independently observed termination is
            // a sharper lower bound on distance to the owner's finite support.
            owner_lateral = std::max(owner_lateral, largest_owner_violation);
            const float owner_residual = std::abs(
                patches[owner].normal.dot(q) + patches[owner].d);
            owner_energy = std::pow(owner_residual / grids[owner].point_sigma, 2.0f) +
                std::pow(owner_lateral / grids[owner].point_sigma, 2.0f);
            int destination = static_cast<int>(owner);
            float best_energy = owner_energy;
            for (std::size_t target_id = 0; target_id < patches.size(); ++target_id) {
                if (target_id == owner || !inside_envelope_fast(target_id, q)) continue;
                const float residual = std::abs(
                    patches[target_id].normal.dot(q) + patches[target_id].d);
                if (residual > cfg.assign_distance ||
                    residual / grids[target_id].point_sigma > 4.0f) {
                    continue;
                }
                float target_lateral = 0.0f;
                const float target_energy = finite_energy(target_id, q, target_lateral);
                if (!std::isfinite(target_energy) || target_lateral > 2.0f * grid_size) {
                    ++stats.rejected_without_local_support;
                    continue;
                }
                ++stats.finite_competitors;
                if (target_energy >= best_energy) continue;
                bool target_is_feasible = true;
                for (const RawBoundarySegment& target_segment : constraints[target_id]) {
                    if (violation(q, target_segment) > 0.0f) {
                        target_is_feasible = false;
                        break;
                    }
                }
                if (!target_is_feasible) continue;
                destination = static_cast<int>(target_id);
                best_energy = target_energy;
            }
            if (best_energy + 1e-4f >= owner_energy) {
                if (destination != static_cast<int>(owner)) ++stats.margin_rejected_points;
                destination = static_cast<int>(owner);
            }
            if (destination == static_cast<int>(owner)) {
                retained->push_back(point);
            } else {
                incoming[static_cast<std::size_t>(destination)].push_back(point);
                stats.corrections.push_back({point, static_cast<int>(owner), destination});
                ++stats.corrected_points;
            }
        }
        retained->width = static_cast<std::uint32_t>(retained->size());
        retained->height = 1;
        retained->is_dense = true;
        patches[owner].output = retained;
    }
    for (std::size_t patch_id = 0; patch_id < patches.size(); ++patch_id) {
        patches[patch_id].output->insert(
            patches[patch_id].output->end(), incoming[patch_id].begin(), incoming[patch_id].end());
        patches[patch_id].output->width = static_cast<std::uint32_t>(patches[patch_id].output->size());
        patches[patch_id].output->height = 1;
        patches[patch_id].output->is_dense = true;
    }
    return stats;
}

static FiniteSupportRegionStats optimize_finite_support_regions(
    const CloudPtr& cloud,
    std::vector<PlanePatch>& patches,
    const std::vector<Surfel>& surfels,
    const Bounds& bounds,
    const Config& cfg) {
    FiniteSupportRegionStats stats;
    if (patches.size() < 2 || cloud->empty()) return stats;

    struct GridKey {
        int u = 0;
        int v = 0;
        bool operator==(const GridKey& other) const { return u == other.u && v == other.v; }
    };
    struct GridKeyHash {
        std::size_t operator()(const GridKey& key) const {
            return static_cast<std::size_t>(key.u) * 73856093u ^
                static_cast<std::size_t>(key.v) * 19349663u;
        }
    };
    struct SupportDomain {
        Eigen::Vector3f axis_u{1.0f, 0.0f, 0.0f};
        Eigen::Vector3f axis_v{0.0f, 1.0f, 0.0f};
        float point_sigma = 0.0f;
        std::unordered_set<GridKey, GridKeyHash> occupied;
        std::unordered_set<GridKey, GridKeyHash> ambiguous;
        std::unordered_set<GridKey, GridKeyHash> sampling_gaps;
    };

    // The support raster is finer than the extraction surfel, but its physical
    // scale is unchanged from the existing raw-geometry audit.  No boundary or
    // half-space tolerance is enlarged by this option.
    const float grid_size = 0.5f * cfg.surfel_size;
    const float envelope_margin = 2.0f * cfg.surfel_size;
    std::vector<SupportDomain> domains(patches.size());
    for (std::size_t plane_id = 0; plane_id < patches.size(); ++plane_id) {
        plane_basis(patches[plane_id], domains[plane_id].axis_u, domains[plane_id].axis_v);
        domains[plane_id].point_sigma = std::max(
            patches[plane_id].rms_distance, 0.05f * cfg.surfel_size);
        domains[plane_id].occupied.reserve(
            std::max<std::size_t>(patches[plane_id].surfels.size() * 4, 16));
        domains[plane_id].ambiguous.reserve(
            std::max<std::size_t>(patches[plane_id].surfels.size(), 16));
    }
    const auto inside_envelope = [&](std::size_t plane_id, const Eigen::Vector3f& q) {
        const Eigen::Vector3f centered = q - patches[plane_id].centroid;
        const float u = centered.dot(domains[plane_id].axis_u);
        const float v = centered.dot(domains[plane_id].axis_v);
        return u >= patches[plane_id].min_u - envelope_margin &&
            u <= patches[plane_id].max_u + envelope_margin &&
            v >= patches[plane_id].min_v - envelope_margin &&
            v <= patches[plane_id].max_v + envelope_margin;
    };
    const auto support_key = [&](std::size_t plane_id, const Eigen::Vector3f& q) {
        const Eigen::Vector3f centered = q - patches[plane_id].centroid;
        return GridKey{
            static_cast<int>(std::floor(centered.dot(domains[plane_id].axis_u) / grid_size)),
            static_cast<int>(std::floor(centered.dot(domains[plane_id].axis_v) / grid_size))
        };
    };

    // Only residual-unique raw observations create support.  Dense output
    // labels are deliberately unavailable here, preventing circular evidence.
    for (const PointT& point : cloud->points) {
        const Eigen::Vector3f q(point.x, point.y, point.z);
        std::size_t best = patches.size();
        float best_z = std::numeric_limits<float>::infinity();
        float second_z = std::numeric_limits<float>::infinity();
        std::vector<std::size_t> plausible;
        for (std::size_t plane_id = 0; plane_id < patches.size(); ++plane_id) {
            if (!inside_envelope(plane_id, q)) continue;
            const float residual = std::abs(
                patches[plane_id].normal.dot(q) + patches[plane_id].d);
            const float z = residual / domains[plane_id].point_sigma;
            if (z <= 3.0f) plausible.push_back(plane_id);
            if (z < best_z) {
                second_z = best_z;
                best_z = z;
                best = plane_id;
            } else if (z < second_z) {
                second_z = z;
            }
        }
        if (best == patches.size() || best_z > 3.0f ||
            second_z < 3.0f || second_z < best_z + 1.5f) {
            for (std::size_t plane_id : plausible) {
                domains[plane_id].ambiguous.insert(support_key(plane_id, q));
            }
            continue;
        }
        domains[best].occupied.insert(support_key(best, q));
        ++stats.trusted_geometry_points;
    }

    // Fill only isolated, bracketed one-cell sampling gaps.  Larger empty
    // components, concave exterior regions, and holes are retained as true
    // unsupported space; this is intentionally not a morphological closing.
    for (SupportDomain& domain : domains) {
        stats.occupied_support_cells += domain.occupied.size();
        stats.ambiguous_observation_cells += domain.ambiguous.size();
        std::unordered_set<GridKey, GridKeyHash> candidates;
        candidates.reserve(domain.occupied.size());
        for (const GridKey& key : domain.occupied) {
            for (int du = -1; du <= 1; ++du) {
                for (int dv = -1; dv <= 1; ++dv) {
                    if (du == 0 && dv == 0) continue;
                    const GridKey candidate{key.u + du, key.v + dv};
                    if (domain.occupied.find(candidate) == domain.occupied.end()) {
                        candidates.insert(candidate);
                    }
                }
            }
        }
        for (const GridKey& key : candidates) {
            int neighbors = 0;
            for (int du = -1; du <= 1; ++du) {
                for (int dv = -1; dv <= 1; ++dv) {
                    if (du == 0 && dv == 0) continue;
                    if (domain.occupied.find({key.u + du, key.v + dv}) != domain.occupied.end()) {
                        ++neighbors;
                    }
                }
            }
            const bool bracket_u =
                domain.occupied.find({key.u - 1, key.v}) != domain.occupied.end() &&
                domain.occupied.find({key.u + 1, key.v}) != domain.occupied.end();
            const bool bracket_v =
                domain.occupied.find({key.u, key.v - 1}) != domain.occupied.end() &&
                domain.occupied.find({key.u, key.v + 1}) != domain.occupied.end();
            if (neighbors >= 6 && bracket_u && bracket_v) {
                domain.sampling_gaps.insert(key);
            }
        }
        stats.inferred_sampling_gap_cells += domain.sampling_gaps.size();
    }

    const auto support_distance = [&](std::size_t plane_id, const Eigen::Vector3f& q) {
        const GridKey base = support_key(plane_id, q);
        const SupportDomain& domain = domains[plane_id];
        float best = std::numeric_limits<float>::infinity();
        // Component admission only distinguishes direct/one-cell support from
        // unsupported space. Searching farther does not change feasibility and
        // dominates CPU time on multi-million-point clouds.
        constexpr int maximum_ring = 2;
        for (int du = -maximum_ring; du <= maximum_ring; ++du) {
            for (int dv = -maximum_ring; dv <= maximum_ring; ++dv) {
                const GridKey key{base.u + du, base.v + dv};
                // Ambiguous observations are an uncertainty mask, not positive
                // support for every nearby plane. Treating them as occupied
                // lets an infinite competing plane grow through a real edge.
                // They are consulted separately when deciding whether missing
                // owner support is conclusive.
                if (domain.occupied.find(key) == domain.occupied.end() &&
                    domain.sampling_gaps.find(key) == domain.sampling_gaps.end()) {
                    continue;
                }
                const float delta_u = static_cast<float>(du);
                const float delta_v = static_cast<float>(dv);
                best = std::min(best,
                    std::sqrt(delta_u * delta_u + delta_v * delta_v) * grid_size);
            }
        }
        return best;
    };
    const float smooth_cos = std::cos(
        cfg.smooth_angle_deg / 180.0f * 3.14159265358979323846f);
    const float edge_cos = std::cos(
        cfg.edge_angle_deg / 180.0f * 3.14159265358979323846f);

    struct PointEvidence {
        PointT point;
        std::size_t point_index = 0;
        int owner = -1;
        std::vector<int> labels;
        float owner_support = std::numeric_limits<float>::infinity();
        float normal_confidence = 0.0f;
        Eigen::Vector3f normal{0.0f, 0.0f, 0.0f};
        float validation_normal_confidence = 0.0f;
        Eigen::Vector3f validation_normal{0.0f, 0.0f, 0.0f};
        bool candidate_from_missing_support = false;
        bool candidate_from_normal_conflict = false;
        bool owner_observation_ambiguous = false;
    };
    struct FineKey {
        int x = 0;
        int y = 0;
        int z = 0;
        int plane_a = -1;
        int plane_b = -1;
        bool operator==(const FineKey& other) const {
            return x == other.x && y == other.y && z == other.z &&
                plane_a == other.plane_a && plane_b == other.plane_b;
        }
    };
    struct FineKeyHash {
        std::size_t operator()(const FineKey& key) const {
            std::size_t h = static_cast<std::size_t>(key.x) * 73856093u;
            h ^= static_cast<std::size_t>(key.y) * 19349663u;
            h ^= static_cast<std::size_t>(key.z) * 83492791u;
            h ^= static_cast<std::size_t>(key.plane_a + 1) * 2654435761u;
            h ^= static_cast<std::size_t>(key.plane_b + 1) * 2246822519u;
            return h;
        }
    };
    struct RegionNode {
        FineKey key;
        int plane_a = -1;
        int plane_b = -1;
        int majority_owner = -1;
        std::vector<PointEvidence> points;
        std::vector<int> labels;
        std::vector<int> neighbors;
        std::vector<float> unary;
        Eigen::Vector3f normal{0.0f, 0.0f, 0.0f};
        float normal_confidence = 0.0f;
    };

    struct FineNormalKey {
        int x = 0;
        int y = 0;
        int z = 0;
        bool operator==(const FineNormalKey& other) const {
            return x == other.x && y == other.y && z == other.z;
        }
    };
    struct FineNormalKeyHash {
        std::size_t operator()(const FineNormalKey& key) const {
            return static_cast<std::size_t>(key.x) * 73856093u ^
                static_cast<std::size_t>(key.y) * 19349663u ^
                static_cast<std::size_t>(key.z) * 83492791u;
        }
    };
    struct FineNormalAccumulator {
        double sx = 0.0;
        double sy = 0.0;
        double sz = 0.0;
        double sxx = 0.0;
        double sxy = 0.0;
        double sxz = 0.0;
        double syy = 0.0;
        double syz = 0.0;
        double szz = 0.0;
        std::size_t count = 0;
    };
    struct FineNormalEstimate {
        Eigen::Vector3f normal{0.0f, 0.0f, 0.0f};
        float confidence = 0.0f;
    };
    const auto fine_normal_key = [&](const PointT& point) {
        return FineNormalKey{
            static_cast<int>(std::floor((point.x - bounds.min_pt.x) / grid_size)),
            static_cast<int>(std::floor((point.y - bounds.min_pt.y) / grid_size)),
            static_cast<int>(std::floor((point.z - bounds.min_pt.z) / grid_size))
        };
    };
    std::unordered_map<FineNormalKey, FineNormalAccumulator, FineNormalKeyHash>
        fine_normal_accumulators;
    std::unordered_map<CellKey, FineNormalEstimate, CellKeyHash> vcm_normal_by_surfel;
    if (cfg.region_normal_estimator == "pca-vcm-consensus") {
        std::vector<CgalKernel::Point_3> points;
        points.reserve(surfels.size());
        for (const Surfel& surfel : surfels) {
            points.emplace_back(
                static_cast<double>(surfel.centroid.x()),
                static_cast<double>(surfel.centroid.y()),
                static_cast<double>(surfel.centroid.z()));
        }
        std::vector<std::array<double, 6>> covariance;
        CGAL::compute_vcm(
            points, covariance,
            2.0 * static_cast<double>(cfg.surfel_size),
            static_cast<double>(cfg.surfel_size));
        vcm_normal_by_surfel.reserve(surfels.size());
        for (std::size_t i = 0; i < surfels.size() && i < covariance.size(); ++i) {
            const std::array<double, 6>& value = covariance[i];
            Eigen::Matrix3d matrix;
            matrix << value[0], value[1], value[2],
                value[1], value[3], value[4],
                value[2], value[4], value[5];
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(matrix);
            FineNormalEstimate estimate;
            if (solver.info() == Eigen::Success) {
                const Eigen::Vector3d eigenvalues = solver.eigenvalues().cwiseMax(0.0);
                const double largest = std::max(eigenvalues[2], 1e-12);
                estimate.normal = solver.eigenvectors().col(2).cast<float>().normalized();
                estimate.confidence = static_cast<float>(std::clamp(
                    (eigenvalues[2] - eigenvalues[1]) / largest, 0.0, 1.0));
            }
            vcm_normal_by_surfel.emplace(surfels[i].key, estimate);
        }
    }
    fine_normal_accumulators.reserve(std::max<std::size_t>(surfels.size() * 4, 16));
    for (const PointT& point : cloud->points) {
        FineNormalAccumulator& accumulator = fine_normal_accumulators[fine_normal_key(point)];
        const double x = point.x;
        const double y = point.y;
        const double z = point.z;
        accumulator.sx += x;
        accumulator.sy += y;
        accumulator.sz += z;
        accumulator.sxx += x * x;
        accumulator.sxy += x * y;
        accumulator.sxz += x * z;
        accumulator.syy += y * y;
        accumulator.syz += y * z;
        accumulator.szz += z * z;
        ++accumulator.count;
    }
    std::unordered_map<FineNormalKey, FineNormalEstimate, FineNormalKeyHash> fine_normal_cache;
    fine_normal_cache.reserve(fine_normal_accumulators.size());
    const auto point_normal = [&](const PointT& point, Eigen::Vector3f& normal, float& confidence) {
        const FineNormalKey key = fine_normal_key(point);
        const auto cached = fine_normal_cache.find(key);
        if (cached != fine_normal_cache.end()) {
            normal = cached->second.normal;
            confidence = cached->second.confidence;
            return;
        }
        FineNormalAccumulator aggregate;
        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dz = -1; dz <= 1; ++dz) {
                    const auto found = fine_normal_accumulators.find(
                        {key.x + dx, key.y + dy, key.z + dz});
                    if (found == fine_normal_accumulators.end()) continue;
                    const FineNormalAccumulator& value = found->second;
                    aggregate.sx += value.sx;
                    aggregate.sy += value.sy;
                    aggregate.sz += value.sz;
                    aggregate.sxx += value.sxx;
                    aggregate.sxy += value.sxy;
                    aggregate.sxz += value.sxz;
                    aggregate.syy += value.syy;
                    aggregate.syz += value.syz;
                    aggregate.szz += value.szz;
                    aggregate.count += value.count;
                }
            }
        }
        FineNormalEstimate estimate;
        if (aggregate.count >= 6) {
            const double inverse_count = 1.0 / static_cast<double>(aggregate.count);
            const Eigen::Vector3d mean(
                aggregate.sx * inverse_count,
                aggregate.sy * inverse_count,
                aggregate.sz * inverse_count);
            Eigen::Matrix3d covariance;
            covariance <<
                aggregate.sxx * inverse_count, aggregate.sxy * inverse_count, aggregate.sxz * inverse_count,
                aggregate.sxy * inverse_count, aggregate.syy * inverse_count, aggregate.syz * inverse_count,
                aggregate.sxz * inverse_count, aggregate.syz * inverse_count, aggregate.szz * inverse_count;
            covariance -= mean * mean.transpose();
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
            if (solver.info() == Eigen::Success) {
                const Eigen::Vector3d eigenvalues = solver.eigenvalues().cwiseMax(0.0);
                const double largest = std::max(eigenvalues[2], 1e-12);
                const double line_support = eigenvalues[1] / largest;
                const double planarity = (eigenvalues[1] - eigenvalues[0]) / largest;
                if (line_support >= 0.05 && planarity > 0.0) {
                    estimate.normal = solver.eigenvectors().col(0).cast<float>().normalized();
                    estimate.confidence = static_cast<float>(std::clamp(planarity, 0.0, 1.0));
                }
            }
        }
        fine_normal_cache.emplace(key, estimate);
        normal = estimate.normal;
        confidence = estimate.confidence;
    };
    const auto validation_normal = [&](
            const PointT& point, Eigen::Vector3f& normal, float& confidence) {
        if (cfg.region_normal_estimator != "pca-vcm-consensus") {
            normal.setZero();
            confidence = 0.0f;
            return;
        }
        const auto found = vcm_normal_by_surfel.find(
            point_key(point, bounds, cfg.surfel_size));
        if (found == vcm_normal_by_surfel.end()) {
            normal.setZero();
            confidence = 0.0f;
            return;
        }
        normal = found->second.normal;
        confidence = found->second.confidence;
    };
    const auto finite_energy = [&](std::size_t plane_id, const Eigen::Vector3f& q,
            const Eigen::Vector3f& normal, float normal_confidence, float& lateral) {
        lateral = support_distance(plane_id, q);
        if (!std::isfinite(lateral)) return std::numeric_limits<float>::infinity();
        const float residual = std::abs(patches[plane_id].normal.dot(q) + patches[plane_id].d);
        const float residual_term = residual / domains[plane_id].point_sigma;
        const float lateral_term = lateral / domains[plane_id].point_sigma;
        float normal_term = 0.0f;
        if (normal_confidence > 0.0f) {
            const float disagreement = 1.0f - std::abs(normal.dot(patches[plane_id].normal));
            normal_term = 16.0f * normal_confidence * disagreement * disagreement;
        }
        return residual_term * residual_term + lateral_term * lateral_term + normal_term;
    };

    std::vector<RegionNode> nodes;
    std::unordered_map<FineKey, int, FineKeyHash> node_by_key;
    const float origin_x = bounds.min_pt.x;
    const float origin_y = bounds.min_pt.y;
    const float origin_z = bounds.min_pt.z;
    for (std::size_t owner = 0; owner < patches.size(); ++owner) {
        const CloudT& output = *patches[owner].output;
        // Keep sub-threshold candidates available as destinations so that
        // independent support can rescue a real small face. Their outgoing
        // points remain under the existing finalizer's all-or-nothing
        // absorption rule; consuming them here would conflate two ablations
        // and let a weak candidate manufacture a large region component.
        if (output.size() < static_cast<std::size_t>(std::max(cfg.min_output_points, 1))) {
            continue;
        }
        for (std::size_t point_index = 0; point_index < output.size(); ++point_index) {
            const PointT& point = output.points[point_index];
            const Eigen::Vector3f q(point.x, point.y, point.z);
            const float owner_support = support_distance(owner, q);
            const bool owner_is_ambiguous = domains[owner].ambiguous.find(
                support_key(owner, q)) != domains[owner].ambiguous.end();
            Eigen::Vector3f local_normal;
            float normal_confidence = 0.0f;
            point_normal(point, local_normal, normal_confidence);
            Eigen::Vector3f local_validation_normal;
            float validation_normal_confidence = 0.0f;
            validation_normal(point, local_validation_normal, validation_normal_confidence);
            if (normal_confidence <= 0.10f) ++stats.mixed_surfel_points;
            float unused_lateral = 0.0f;
            const float owner_energy = finite_energy(
                owner, q, local_normal, normal_confidence, unused_lateral);
            const float owner_normal_dot = normal_confidence > 0.0f
                ? std::abs(local_normal.dot(patches[owner].normal)) : 1.0f;
            int best_alternative = -1;
            float best_alternative_energy = std::numeric_limits<float>::infinity();
            bool best_support_inconsistent = false;
            bool best_normal_inconsistent = false;
            for (std::size_t target = 0; target < patches.size(); ++target) {
                if (target == owner || !inside_envelope(target, q)) continue;
                const float residual = std::abs(patches[target].normal.dot(q) + patches[target].d);
                if (residual > cfg.assign_distance) continue;
                float target_support = 0.0f;
                const float target_energy = finite_energy(
                    target, q, local_normal, normal_confidence, target_support);
                if (!std::isfinite(target_energy) || target_support > grid_size) continue;
                const float target_normal_dot = normal_confidence > 0.0f
                    ? std::abs(local_normal.dot(patches[target].normal)) : 1.0f;
                // Grid distances are quantized. A point one full cell away
                // from trusted owner support is already outside the observed
                // finite support core; using a strict greater-than discarded
                // exactly the scan-line bands seen in the two reported ROIs.
                // Lack of unique support inside an ambiguous intersection
                // corridor is inconclusive. Away from that corridor, however,
                // a one-cell gap is independent evidence that the current
                // finite plane does not own this point.
                const bool support_inconsistent = owner_support >= grid_size &&
                    !owner_is_ambiguous;
                bool geometry_inconsistent = normal_confidence >= 0.20f &&
                    owner_normal_dot < edge_cos && target_normal_dot >= smooth_cos;
                if (geometry_inconsistent &&
                    cfg.region_normal_estimator == "pca-vcm-consensus") {
                    const float validation_owner_dot = validation_normal_confidence > 0.0f
                        ? std::abs(local_validation_normal.dot(patches[owner].normal)) : 1.0f;
                    const float validation_target_dot = validation_normal_confidence > 0.0f
                        ? std::abs(local_validation_normal.dot(patches[target].normal)) : 0.0f;
                    geometry_inconsistent = validation_normal_confidence >= 0.20f &&
                        validation_owner_dot < edge_cos &&
                        validation_target_dot >= smooth_cos;
                }
                if (!support_inconsistent && !geometry_inconsistent) {
                    continue;
                }
                // A dimensionless retention prior represents uncertainty, not
                // a geometric distance or half-space expansion. Every graph
                // label must improve the pointwise finite-support explanation;
                // connectivity only stabilizes a set of already plausible
                // moves and cannot drag a clearly worse plane across an edge.
                // No independent target advantage is required at this stage;
                // the component energy makes the final retain/change decision.
                if (target_energy > owner_energy + 1.0f) continue;
                if (target_energy < best_alternative_energy) {
                    best_alternative = static_cast<int>(target);
                    best_alternative_energy = target_energy;
                    best_support_inconsistent = support_inconsistent;
                    best_normal_inconsistent = geometry_inconsistent;
                }
            }
            if (best_alternative < 0) continue;
            ++stats.support_inconsistent_points;
            const int plane_a = std::min(static_cast<int>(owner), best_alternative);
            const int plane_b = std::max(static_cast<int>(owner), best_alternative);
            FineKey key{
                static_cast<int>(std::floor((point.x - origin_x) / grid_size)),
                static_cast<int>(std::floor((point.y - origin_y) / grid_size)),
                static_cast<int>(std::floor((point.z - origin_z) / grid_size)),
                plane_a,
                plane_b
            };
            auto inserted = node_by_key.emplace(key, static_cast<int>(nodes.size()));
            if (inserted.second) {
                RegionNode node;
                node.key = key;
                node.plane_a = plane_a;
                node.plane_b = plane_b;
                node.labels.push_back(-1);  // retain every point's existing owner
                node.labels.push_back(plane_a);
                node.labels.push_back(plane_b);
                nodes.push_back(std::move(node));
            }
            RegionNode& node = nodes[static_cast<std::size_t>(inserted.first->second)];
            PointEvidence evidence;
            evidence.point = point;
            evidence.point_index = point_index;
            evidence.owner = static_cast<int>(owner);
            evidence.labels.push_back(best_alternative);
            evidence.labels.push_back(static_cast<int>(owner));
            evidence.owner_support = owner_support;
            evidence.normal_confidence = normal_confidence;
            evidence.normal = local_normal;
            evidence.validation_normal_confidence = validation_normal_confidence;
            evidence.validation_normal = local_validation_normal;
            evidence.candidate_from_missing_support = best_support_inconsistent;
            evidence.candidate_from_normal_conflict = best_normal_inconsistent;
            evidence.owner_observation_ambiguous = owner_is_ambiguous;
            node.points.push_back(std::move(evidence));
            if (normal_confidence > 0.0f) {
                node.normal += normal_confidence * canonical_normal(local_normal);
            }
            node.normal_confidence += normal_confidence;
        }
    }
    stats.competition_nodes = nodes.size();
    if (nodes.empty()) return stats;

    struct SpatialKey {
        int x = 0;
        int y = 0;
        int z = 0;
        bool operator==(const SpatialKey& other) const {
            return x == other.x && y == other.y && z == other.z;
        }
    };
    struct SpatialKeyHash {
        std::size_t operator()(const SpatialKey& key) const {
            return static_cast<std::size_t>(key.x) * 73856093u ^
                static_cast<std::size_t>(key.y) * 19349663u ^
                static_cast<std::size_t>(key.z) * 83492791u;
        }
    };
    std::unordered_map<SpatialKey, std::vector<int>, SpatialKeyHash> nodes_by_cell;
    nodes_by_cell.reserve(nodes.size());
    for (std::size_t node_id = 0; node_id < nodes.size(); ++node_id) {
        RegionNode& node = nodes[node_id];
        std::sort(node.labels.begin(), node.labels.end());
        node.labels.erase(std::unique(node.labels.begin(), node.labels.end()), node.labels.end());
        std::size_t count_a = 0;
        std::size_t count_b = 0;
        for (const PointEvidence& evidence : node.points) {
            if (evidence.owner == node.plane_a) ++count_a;
            if (evidence.owner == node.plane_b) ++count_b;
        }
        node.majority_owner = count_a >= count_b ? node.plane_a : node.plane_b;
        node.normal_confidence /= static_cast<float>(std::max<std::size_t>(node.points.size(), 1));
        if (node.normal.norm() > 1e-6f) node.normal.normalize();
        nodes_by_cell[{node.key.x, node.key.y, node.key.z}].push_back(static_cast<int>(node_id));
    }
    for (std::size_t node_id = 0; node_id < nodes.size(); ++node_id) {
        RegionNode& node = nodes[node_id];
        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dz = -1; dz <= 1; ++dz) {
                    if (dx == 0 && dy == 0 && dz == 0) continue;
                    const auto found = nodes_by_cell.find(
                        {node.key.x + dx, node.key.y + dy, node.key.z + dz});
                    if (found == nodes_by_cell.end()) continue;
                    for (int neighbor : found->second) {
                        if (neighbor == static_cast<int>(node_id)) continue;
                        const RegionNode& other = nodes[static_cast<std::size_t>(neighbor)];
                        if (node.plane_a != other.plane_a || node.plane_b != other.plane_b) continue;
                        if (node.normal_confidence >= 0.20f &&
                            other.normal_confidence >= 0.20f &&
                            std::abs(node.normal.dot(other.normal)) < edge_cos) {
                            continue;
                        }
                        node.neighbors.push_back(neighbor);
                    }
                }
            }
        }
        std::sort(node.neighbors.begin(), node.neighbors.end());
        node.neighbors.erase(
            std::unique(node.neighbors.begin(), node.neighbors.end()), node.neighbors.end());
        node.unary.reserve(node.labels.size());
        for (int label : node.labels) {
            double energy = 0.0;
            for (const PointEvidence& evidence : node.points) {
                const Eigen::Vector3f q(evidence.point.x, evidence.point.y, evidence.point.z);
                float lateral = 0.0f;
                const int evaluated_label = label < 0 ? evidence.owner : label;
                const bool feasible = label < 0 ||
                    std::find(evidence.labels.begin(), evidence.labels.end(), label) !=
                        evidence.labels.end();
                if (!feasible) {
                    energy = std::numeric_limits<double>::infinity();
                    break;
                }
                energy += finite_energy(
                    static_cast<std::size_t>(evaluated_label), q, evidence.normal,
                    evidence.normal_confidence, lateral);
                if (label >= 0 && label != evidence.owner) energy += 0.25;
            }
            energy /= static_cast<double>(std::max<std::size_t>(node.points.size(), 1));
            node.unary.push_back(static_cast<float>(energy));
        }
    }

    std::vector<int> component_of(nodes.size(), -1);
    std::vector<std::vector<int>> components;
    for (std::size_t seed = 0; seed < nodes.size(); ++seed) {
        if (component_of[seed] >= 0) continue;
        const int component_id = static_cast<int>(components.size());
        components.emplace_back();
        std::queue<int> pending;
        pending.push(static_cast<int>(seed));
        component_of[seed] = component_id;
        while (!pending.empty()) {
            const int node_id = pending.front();
            pending.pop();
            components.back().push_back(node_id);
            for (int neighbor : nodes[static_cast<std::size_t>(node_id)].neighbors) {
                if (component_of[static_cast<std::size_t>(neighbor)] >= 0) continue;
                component_of[static_cast<std::size_t>(neighbor)] = component_id;
                pending.push(neighbor);
            }
        }
    }
    stats.competition_components = components.size();

    std::vector<std::vector<int>> destinations(patches.size());
    std::vector<std::size_t> original_sizes(patches.size(), 0);
    for (std::size_t plane_id = 0; plane_id < patches.size(); ++plane_id) {
        original_sizes[plane_id] = patches[plane_id].output->size();
        destinations[plane_id].assign(patches[plane_id].output->size(), static_cast<int>(plane_id));
    }
    struct PendingDecision {
        FiniteSupportRegionDecision decision;
        std::size_t point_index = 0;
    };
    std::vector<PendingDecision> pending_decisions;
    const auto label_index = [](const RegionNode& node, int label) {
        const auto found = std::lower_bound(node.labels.begin(), node.labels.end(), label);
        return found != node.labels.end() && *found == label
            ? static_cast<int>(std::distance(node.labels.begin(), found)) : -1;
    };
    const auto effective_label = [&](int node_id, int label) {
        return label < 0
            ? nodes[static_cast<std::size_t>(node_id)].majority_owner : label;
    };
    const auto node_cost = [&](int node_id, int label, const std::vector<int>& labeling) {
        const RegionNode& node = nodes[static_cast<std::size_t>(node_id)];
        const int index = label_index(node, label);
        if (index < 0) return std::numeric_limits<float>::infinity();
        float cost = node.unary[static_cast<std::size_t>(index)];
        for (int neighbor : node.neighbors) {
            if (component_of[static_cast<std::size_t>(neighbor)] !=
                component_of[static_cast<std::size_t>(node_id)]) {
                continue;
            }
            if (effective_label(neighbor, labeling[static_cast<std::size_t>(neighbor)]) ==
                effective_label(node_id, label)) {
                continue;
            }
            const RegionNode& other = nodes[static_cast<std::size_t>(neighbor)];
            const float confidence = std::min(node.normal_confidence, other.normal_confidence);
            const float alignment = node.normal.norm() > 0.5f && other.normal.norm() > 0.5f
                ? std::abs(node.normal.dot(other.normal)) : 0.0f;
            cost += 0.35f * confidence * alignment * alignment;
        }
        return cost;
    };
    const auto total_energy = [&](
            const std::vector<int>& component,
            const std::vector<int>& labeling) {
        double energy = 0.0;
        for (int node_id : component) {
            const RegionNode& node = nodes[static_cast<std::size_t>(node_id)];
            const int label = labeling[static_cast<std::size_t>(node_id)];
            const int index = label_index(node, label);
            if (index < 0) return std::numeric_limits<double>::infinity();
            energy += node.unary[static_cast<std::size_t>(index)];
            for (int neighbor : node.neighbors) {
                if (neighbor <= node_id ||
                    component_of[static_cast<std::size_t>(neighbor)] !=
                        component_of[static_cast<std::size_t>(node_id)] ||
                    effective_label(neighbor, labeling[static_cast<std::size_t>(neighbor)]) ==
                        effective_label(node_id, label)) {
                    continue;
                }
                const RegionNode& other = nodes[static_cast<std::size_t>(neighbor)];
                const float confidence = std::min(node.normal_confidence, other.normal_confidence);
                const float alignment = node.normal.norm() > 0.5f && other.normal.norm() > 0.5f
                    ? std::abs(node.normal.dot(other.normal)) : 0.0f;
                energy += 0.35 * confidence * alignment * alignment;
            }
        }
        return energy;
    };

    for (std::size_t component_id = 0; component_id < components.size(); ++component_id) {
        const std::vector<int>& component = components[component_id];
        if (component.size() < 2) {
            ++stats.retained_components;
            continue;
        }
        std::vector<int> labels(nodes.size(), -1);
        const std::vector<int> original = labels;
        for (int iteration = 0; iteration < 8; ++iteration) {
            std::vector<int> next = labels;
            std::size_t changed = 0;
            for (int node_id : component) {
                const RegionNode& node = nodes[static_cast<std::size_t>(node_id)];
                int best_label = labels[static_cast<std::size_t>(node_id)];
                float best_cost = node_cost(node_id, best_label, labels);
                for (int candidate : node.labels) {
                    const float cost = node_cost(node_id, candidate, labels);
                    if (cost + 1e-6f < best_cost) {
                        best_cost = cost;
                        best_label = candidate;
                    }
                }
                next[static_cast<std::size_t>(node_id)] = best_label;
                if (best_label != labels[static_cast<std::size_t>(node_id)]) ++changed;
            }
            if (total_energy(component, next) + 1e-9 >= total_energy(component, labels)) break;
            labels.swap(next);
            if (changed == 0) break;
        }
        const double original_energy = total_energy(component, original);
        const double optimized_energy = total_energy(component, labels);
        std::size_t changed_nodes = 0;
        for (int node_id : component) {
            if (labels[static_cast<std::size_t>(node_id)] !=
                original[static_cast<std::size_t>(node_id)]) {
                ++changed_nodes;
            }
        }
        // Require a connected multi-cell move and an aggregate gain above
        // numerical/model uncertainty; otherwise retain every point's owner.
        const double required_gain = 0.25 * std::sqrt(static_cast<double>(component.size()));
        if (changed_nodes < 2 || optimized_energy + required_gain >= original_energy) {
            ++stats.retained_components;
            continue;
        }
        const std::size_t pending_before = pending_decisions.size();
        for (int node_id : component) {
            RegionNode& node = nodes[static_cast<std::size_t>(node_id)];
            const int destination = labels[static_cast<std::size_t>(node_id)];
            if (destination < 0) continue;
            const int from_index = label_index(node, -1);
            const int to_index = label_index(node, destination);
            // Pairwise coherence may break a near tie, but it must never drag
            // a locally worse fine cell across a sharp finite boundary. This
            // veto is applied after graph construction/optimization, so weak
            // candidates still keep a competition component connected.
            if (to_index < 0 || from_index < 0 ||
                node.unary[static_cast<std::size_t>(to_index)] + 1e-6f >=
                    node.unary[static_cast<std::size_t>(from_index)]) {
                continue;
            }
            for (const PointEvidence& evidence : node.points) {
                if (destination == evidence.owner) continue;
                const Eigen::Vector3f q(evidence.point.x, evidence.point.y, evidence.point.z);
                float owner_support = 0.0f;
                float target_support = 0.0f;
                const float owner_energy = finite_energy(
                    static_cast<std::size_t>(evidence.owner), q, evidence.normal,
                    evidence.normal_confidence, owner_support);
                const float target_energy = finite_energy(
                    static_cast<std::size_t>(destination), q, evidence.normal,
                    evidence.normal_confidence, target_support);
                const float owner_residual = std::abs(
                    patches[static_cast<std::size_t>(evidence.owner)].normal.dot(q) +
                    patches[static_cast<std::size_t>(evidence.owner)].d);
                const float target_residual = std::abs(
                    patches[static_cast<std::size_t>(destination)].normal.dot(q) +
                    patches[static_cast<std::size_t>(destination)].d);
                const float residual_uncertainty = std::hypot(
                    domains[static_cast<std::size_t>(evidence.owner)].point_sigma,
                    domains[static_cast<std::size_t>(destination)].point_sigma);
                const float owner_normal_dot = evidence.normal_confidence > 0.0f
                    ? std::abs(evidence.normal.dot(
                        patches[static_cast<std::size_t>(evidence.owner)].normal)) : 1.0f;
                const float target_normal_dot = evidence.normal_confidence > 0.0f
                    ? std::abs(evidence.normal.dot(
                        patches[static_cast<std::size_t>(destination)].normal)) : 1.0f;
                const bool residual_decisive =
                    target_residual + residual_uncertainty < owner_residual;
                bool normal_decisive = evidence.normal_confidence >= 0.20f &&
                    target_normal_dot >= smooth_cos && owner_normal_dot < edge_cos;
                if (normal_decisive &&
                    cfg.region_normal_estimator == "pca-vcm-consensus") {
                    const float validation_owner_dot = evidence.validation_normal_confidence > 0.0f
                        ? std::abs(evidence.validation_normal.dot(
                            patches[static_cast<std::size_t>(evidence.owner)].normal)) : 1.0f;
                    const float validation_target_dot = evidence.validation_normal_confidence > 0.0f
                        ? std::abs(evidence.validation_normal.dot(
                            patches[static_cast<std::size_t>(destination)].normal)) : 0.0f;
                    normal_decisive = evidence.validation_normal_confidence >= 0.20f &&
                        validation_target_dot >= smooth_cos &&
                        validation_owner_dot < edge_cos;
                }
                if (!residual_decisive && !normal_decisive) {
                    ++stats.local_geometry_rejections;
                    continue;
                }
                destinations[static_cast<std::size_t>(evidence.owner)][evidence.point_index] = destination;
                pending_decisions.push_back({{
                        evidence.point,
                        evidence.owner,
                        destination,
                        static_cast<int>(component_id),
                        owner_energy,
                        target_energy,
                        owner_support,
                        target_support,
                        evidence.normal_confidence,
                        owner_residual,
                        target_residual,
                        owner_normal_dot,
                        target_normal_dot,
                        node.unary[static_cast<std::size_t>(from_index)],
                        node.unary[static_cast<std::size_t>(to_index)],
                        evidence.candidate_from_missing_support,
                        evidence.candidate_from_normal_conflict,
                        evidence.owner_observation_ambiguous,
                        residual_decisive,
                        normal_decisive
                    }, evidence.point_index});
            }
        }
        if (pending_decisions.size() > pending_before) {
            ++stats.optimized_components;
        } else {
            ++stats.retained_components;
        }
    }

    // A postprocessor must not destroy an otherwise deliverable model merely
    // because its boundary is locally contested. Evaluate all moves
    // synchronously, then roll back outgoing moves from an established plane
    // if they would push its final support below the normal output criterion.
    // This is a conservative evidence rule, not a preference for plane count:
    // sub-threshold models remain eligible as destinations and still follow
    // the unchanged finalization policy.
    const std::size_t minimum_points = static_cast<std::size_t>(
        std::max(cfg.min_output_points, 1));
    std::vector<std::size_t> final_sizes = original_sizes;
    for (std::size_t owner = 0; owner < destinations.size(); ++owner) {
        for (int destination : destinations[owner]) {
            if (destination == static_cast<int>(owner)) continue;
            --final_sizes[owner];
            ++final_sizes[static_cast<std::size_t>(destination)];
        }
    }
    bool support_changed = true;
    while (support_changed) {
        support_changed = false;
        for (std::size_t owner = 0; owner < destinations.size(); ++owner) {
            if (original_sizes[owner] < minimum_points || final_sizes[owner] >= minimum_points) {
                continue;
            }
            for (int& destination : destinations[owner]) {
                if (destination == static_cast<int>(owner)) continue;
                --final_sizes[static_cast<std::size_t>(destination)];
                ++final_sizes[owner];
                destination = static_cast<int>(owner);
                ++stats.support_preservation_rejections;
            }
            support_changed = true;
        }
    }
    for (PendingDecision& pending : pending_decisions) {
        const FiniteSupportRegionDecision& decision = pending.decision;
        if (destinations[static_cast<std::size_t>(decision.from_plane)][pending.point_index] !=
            decision.to_plane) {
            continue;
        }
        stats.decisions.push_back(std::move(pending.decision));
        ++stats.corrected_points;
    }

    std::vector<CloudPtr> rebuilt(patches.size());
    for (CloudPtr& output : rebuilt) output.reset(new CloudT);
    for (std::size_t owner = 0; owner < patches.size(); ++owner) {
        for (std::size_t point_id = 0; point_id < patches[owner].output->size(); ++point_id) {
            const int destination = destinations[owner][point_id];
            rebuilt[static_cast<std::size_t>(destination)]->push_back(
                patches[owner].output->points[point_id]);
        }
    }
    for (std::size_t plane_id = 0; plane_id < patches.size(); ++plane_id) {
        rebuilt[plane_id]->width = static_cast<std::uint32_t>(rebuilt[plane_id]->size());
        rebuilt[plane_id]->height = 1;
        rebuilt[plane_id]->is_dense = true;
        patches[plane_id].output = rebuilt[plane_id];
    }
    return stats;
}

static CornerExtractionResult extract_corners_from_adjacency(
    const CloudPtr& cloud,
    const std::vector<PlanePatch>& patches,
    const std::vector<Surfel>& surfels,
    const SurfelNeighborhoods& neighborhoods,
    const Bounds& bounds,
    const Config& cfg,
    const std::vector<int>* final_labels = nullptr) {
    CornerExtractionResult result;
    result.finite_intersections =
        build_finite_plane_intersections(patches, surfels, neighborhoods, cfg, final_labels);
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
    result.corners = merge_corner_candidates_complete_link(std::move(candidates), merge_radius);
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

struct PointCoordinateKey {
    std::uint32_t x = 0;
    std::uint32_t y = 0;
    std::uint32_t z = 0;

    bool operator==(const PointCoordinateKey& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct PointCoordinateKeyHash {
    std::size_t operator()(const PointCoordinateKey& key) const {
        std::size_t value = static_cast<std::size_t>(key.x) * 0x9e3779b1u;
        value ^= static_cast<std::size_t>(key.y) * 0x85ebca77u;
        value ^= static_cast<std::size_t>(key.z) * 0xc2b2ae3du;
        return value;
    }
};

static PointCoordinateKey point_coordinate_key(const PointT& point) {
    PointCoordinateKey key;
    std::memcpy(&key.x, &point.x, sizeof(float));
    std::memcpy(&key.y, &point.y, sizeof(float));
    std::memcpy(&key.z, &point.z, sizeof(float));
    return key;
}

template <typename Decision>
static void resolve_final_planes(
    const std::vector<PlanePatch>& patches,
    std::vector<Decision>& decisions) {
    std::unordered_multimap<PointCoordinateKey, std::size_t, PointCoordinateKeyHash> requested;
    requested.reserve(decisions.size());
    for (std::size_t i = 0; i < decisions.size(); ++i) {
        decisions[i].final_plane = -1;
        requested.emplace(point_coordinate_key(decisions[i].point), i);
    }
    if (requested.empty()) return;
    for (std::size_t plane_id = 0; plane_id < patches.size(); ++plane_id) {
        for (const PointT& point : patches[plane_id].output->points) {
            const auto range = requested.equal_range(point_coordinate_key(point));
            for (auto match = range.first; match != range.second; ++match) {
                decisions[match->second].final_plane = static_cast<int>(plane_id);
            }
        }
    }
}

static void save_plane_patches(
    const fs::path& path,
    const std::vector<PlanePatch>& patches,
    bool binary) {
    std::ofstream out(path, binary ? (std::ios::out | std::ios::binary) : std::ios::out);
    if (!out) throw std::runtime_error("Failed to write PLY: " + path.string());
    std::size_t vertex_count = 0;
    for (const auto& patch : patches) vertex_count += patch.output->size();
    if (binary) {
        out << "ply\n"
            << "format binary_little_endian 1.0\n"
            << "element vertex " << vertex_count << "\n"
            << "property float x\n"
            << "property float y\n"
            << "property float z\n"
            << "property uchar red\n"
            << "property uchar green\n"
            << "property uchar blue\n"
            << "end_header\n";
        constexpr std::size_t packed_size = 3 * sizeof(float) + 3 * sizeof(std::uint8_t);
        constexpr std::size_t points_per_chunk = 65536;
        std::vector<char> packed(points_per_chunk * packed_size);
        std::size_t buffered_points = 0;
        const auto flush = [&]() {
            out.write(
                packed.data(),
                static_cast<std::streamsize>(buffered_points * packed_size));
            buffered_points = 0;
        };
        for (std::size_t i = 0; i < patches.size(); ++i) {
            const auto color = color_for(i);
            for (const auto& point : patches[i].output->points) {
                char* destination = packed.data() + buffered_points * packed_size;
                std::memcpy(destination, &point.x, sizeof(float));
                std::memcpy(destination + sizeof(float), &point.y, sizeof(float));
                std::memcpy(destination + 2 * sizeof(float), &point.z, sizeof(float));
                destination[3 * sizeof(float)] = static_cast<char>(color[0]);
                destination[3 * sizeof(float) + 1] = static_cast<char>(color[1]);
                destination[3 * sizeof(float) + 2] = static_cast<char>(color[2]);
                ++buffered_points;
                if (buffered_points == points_per_chunk) flush();
            }
        }
        if (buffered_points > 0) flush();
        return;
    }
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
        const auto color = correction.final_plane >= 0
            ? color_for(static_cast<std::size_t>(correction.final_plane))
            : std::array<std::uint8_t, 3>{128, 128, 128};
        ply << correction.point.x << ' ' << correction.point.y << ' ' << correction.point.z
            << ' ' << static_cast<int>(color[0])
            << ' ' << static_cast<int>(color[1])
            << ' ' << static_cast<int>(color[2]) << '\n';
    }

    std::ofstream csv(csv_path);
    if (!csv) throw std::runtime_error("Failed to write topology corrections: " + csv_path.string());
    csv << "x,y,z,from_plane,to_plane,final_plane\n";
    csv << std::setprecision(9);
    for (const TopologyCorrection& correction : corrections) {
        csv << correction.point.x << ',' << correction.point.y << ',' << correction.point.z << ','
            << correction.from_plane << ',' << correction.to_plane << ','
            << correction.final_plane << '\n';
    }
}

static void save_finite_support_region_decisions(
    const fs::path& ply_path,
    const fs::path& csv_path,
    const std::vector<FiniteSupportRegionDecision>& decisions) {
    std::ofstream ply(ply_path);
    if (!ply) throw std::runtime_error("Failed to write finite-support decisions: " + ply_path.string());
    write_color_ply_header(ply, decisions.size());
    for (const FiniteSupportRegionDecision& decision : decisions) {
        const auto color = decision.final_plane >= 0
            ? color_for(static_cast<std::size_t>(decision.final_plane))
            : std::array<std::uint8_t, 3>{128, 128, 128};
        ply << decision.point.x << ' ' << decision.point.y << ' ' << decision.point.z
            << ' ' << static_cast<int>(color[0])
            << ' ' << static_cast<int>(color[1])
            << ' ' << static_cast<int>(color[2]) << '\n';
    }
    std::ofstream csv(csv_path);
    if (!csv) throw std::runtime_error("Failed to write finite-support decisions: " + csv_path.string());
    csv << "x,y,z,from_plane,to_plane,final_plane,component,from_energy,to_energy,"
           "from_support_distance,to_support_distance,normal_confidence,"
           "from_residual,to_residual,from_normal_dot,to_normal_dot,"
           "node_retain_energy,node_target_energy,candidate_from_missing_support,"
           "candidate_from_normal_conflict,owner_observation_ambiguous,"
           "accepted_by_residual,accepted_by_normal\n";
    csv << std::setprecision(9);
    for (const FiniteSupportRegionDecision& decision : decisions) {
        csv << decision.point.x << ',' << decision.point.y << ',' << decision.point.z << ','
            << decision.from_plane << ',' << decision.to_plane << ',' << decision.final_plane << ','
            << decision.component << ','
            << decision.from_energy << ',' << decision.to_energy << ','
            << decision.from_support_distance << ',' << decision.to_support_distance << ','
            << decision.normal_confidence << ','
            << decision.from_residual << ',' << decision.to_residual << ','
            << decision.from_normal_dot << ',' << decision.to_normal_dot << ','
            << decision.node_retain_energy << ',' << decision.node_target_energy << ','
            << static_cast<int>(decision.candidate_from_missing_support) << ','
            << static_cast<int>(decision.candidate_from_normal_conflict) << ','
            << static_cast<int>(decision.owner_observation_ambiguous) << ','
            << static_cast<int>(decision.accepted_by_residual) << ','
            << static_cast<int>(decision.accepted_by_normal) << '\n';
    }
}

static void save_plane_parameters_csv(const fs::path& path, const std::vector<PlanePatch>& patches) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Failed to write plane parameters: " + path.string());
    out << "plane_id,pre_finalize_plane_id,surfels,dense_points,nx,ny,nz,d,cx,cy,cz,extent_u,extent_v,rms_distance\n";
    out << std::setprecision(9);
    for (std::size_t i = 0; i < patches.size(); ++i) {
        const PlanePatch& patch = patches[i];
        out << i << ',' << patch.pre_finalize_plane_id << ','
            << patch.surfels.size() << ',' << patch.output->size() << ','
            << patch.normal.x() << ',' << patch.normal.y() << ',' << patch.normal.z() << ',' << patch.d << ','
            << patch.centroid.x() << ',' << patch.centroid.y() << ',' << patch.centroid.z() << ','
            << patch.extent_u << ',' << patch.extent_v << ',' << patch.rms_distance << '\n';
    }
}

static std::pair<float, float> patch_model_diagnostics(
    const PlanePatch& patch,
    const std::vector<Surfel>& surfels) {
    if (patch.surfels.size() < 7) {
        return {std::numeric_limits<float>::infinity(), 0.0f};
    }
    Eigen::Vector3f axis_u;
    Eigen::Vector3f axis_v;
    plane_basis(patch, axis_u, axis_v);
    const Eigen::Index count = static_cast<Eigen::Index>(patch.surfels.size());
    Eigen::MatrixXd design(count, 6);
    Eigen::VectorXd response(count);
    double plane_rss = 0.0;
    double angle_squared = 0.0;
    double total_weight = 0.0;
    for (Eigen::Index row = 0; row < count; ++row) {
        const Surfel& surfel = surfels[static_cast<std::size_t>(patch.surfels[static_cast<std::size_t>(row)])];
        const Eigen::Vector3f centered = surfel.centroid - patch.centroid;
        const double u = static_cast<double>(centered.dot(axis_u));
        const double v = static_cast<double>(centered.dot(axis_v));
        const double height = static_cast<double>(patch.normal.dot(surfel.centroid) + patch.d);
        const double weight = static_cast<double>(std::max(surfel.count, 1));
        const double root_weight = std::sqrt(weight);
        design.row(row) << root_weight * u * u, root_weight * u * v,
            root_weight * v * v, root_weight * u, root_weight * v, root_weight;
        response[row] = root_weight * height;
        plane_rss += weight * height * height;
        const double dot = std::clamp(
            static_cast<double>(std::abs(surfel.normal.dot(patch.normal))), 0.0, 1.0);
        const double angle = std::acos(dot) * 180.0 / 3.14159265358979323846;
        angle_squared += weight * angle * angle;
        total_weight += weight;
    }
    const Eigen::VectorXd coefficients = design.colPivHouseholderQr().solve(response);
    const double quadratic_rss = (design * coefficients - response).squaredNorm();
    const double observations = static_cast<double>(patch.surfels.size());
    const double floor = std::max(1e-18, observations * 1e-18);
    const double plane_bic = observations * std::log(std::max(plane_rss, floor) / observations) +
        3.0 * std::log(observations);
    const double quadratic_bic = observations * std::log(std::max(quadratic_rss, floor) / observations) +
        6.0 * std::log(observations);
    return {
        static_cast<float>(std::sqrt(angle_squared / std::max(total_weight, 1.0))),
        static_cast<float>(plane_bic - quadratic_bic)
    };
}

static void save_proposal_diagnostics(
    const fs::path& path,
    const std::vector<PlanePatch>& patches,
    const std::vector<Surfel>& surfels) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Failed to write proposal diagnostics: " + path.string());
    out << "plane_id,residual_boundary_model,surfels,dense_points,normal_rms_deg,quadratic_bic_gain\n";
    out << std::setprecision(9);
    for (std::size_t patch_id = 0; patch_id < patches.size(); ++patch_id) {
        const auto diagnostics = patch_model_diagnostics(patches[patch_id], surfels);
        out << patch_id << ',' << (patches[patch_id].requires_local_support ? 1 : 0) << ','
            << patches[patch_id].surfels.size() << ',' << patches[patch_id].output->size() << ','
            << diagnostics.first << ',' << diagnostics.second << '\n';
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
        int strict_local_support = 0;
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
                    if (std::abs(dx) <= 1 && std::abs(dy) <= 1 && std::abs(dz) <= 1) {
                        ++candidates[static_cast<std::size_t>(label)].strict_local_support;
                    }
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
                  << ", strict_local_support=" << candidate.strict_local_support
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
    const SurfelNeighborhoods& neighborhoods,
    const std::vector<PlanePatch>& patches,
    const std::vector<int>* final_labels = nullptr) {
    const std::vector<int> labels = final_labels != nullptr
        ? *final_labels : build_surfel_labels(patches, surfels.size());

    const auto is_feature_edge = [&](std::size_t i) {
        if (labels[i] < 0) return false;
        for (int neighbor : neighborhoods[i]) {
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
        const SurfelNeighborhoods neighborhoods = build_surfel_neighborhoods(surfels, index_by_key);
        std::size_t neighborhood_entries = 0;
        std::size_t neighborhood_capacity = 0;
        for (const std::vector<int>& neighbors : neighborhoods) {
            neighborhood_entries += neighbors.size();
            neighborhood_capacity += neighbors.capacity();
        }
        const std::size_t neighborhood_storage_bytes =
            neighborhoods.size() * sizeof(std::vector<int>) + neighborhood_capacity * sizeof(int);
        now = Clock::now();
        timings.surfel_build_ms = elapsed_ms(stage_start, now);
        stage_start = now;
        estimate_surfel_normals(surfels, surfel_cloud, cfg.normal_k);

        std::vector<PlanePatch> patches;
        std::size_t coarse_candidate_count = 0;
        std::size_t local_candidate_count = 0;
        std::size_t coarse_capacity_expansions = 0;
        std::size_t local_capacity_expansions = 0;
        std::size_t residual_boundary_surfels = 0;
        std::size_t residual_candidate_count = 0;
        std::size_t residual_capacity_expansions = 0;
        ResidualNoveltyStats residual_novelty;
        if (cfg.pipeline == "baseline") {
            mark_edge_surfels(surfels, neighborhoods, cfg);
            patches = extract_all_seed_grown_patches(
                surfels, neighborhoods, cfg, coarse_capacity_expansions);
            coarse_candidate_count = patches.size();
        } else {
            std::vector<Surfel> coarse_surfels = surfels;
            std::vector<Surfel> local_surfels = surfels;
            estimate_surfel_normals(local_surfels, surfel_cloud, std::max(6, cfg.normal_k / 2));
            mark_edge_surfels(coarse_surfels, neighborhoods, cfg);
            mark_edge_surfels(local_surfels, neighborhoods, cfg);

            std::vector<PlanePatch> coarse_patches =
                extract_all_seed_grown_patches(
                    coarse_surfels, neighborhoods, cfg, coarse_capacity_expansions);
            coarse_candidate_count = coarse_patches.size();

            surfels = fuse_multiscale_surfels(coarse_surfels, local_surfels);
            Config local_cfg = cfg;
            local_cfg.min_surfels = std::max(8, cfg.min_surfels / 2);
            local_cfg.min_extent = std::max(1.5f, 3.0f * cfg.surfel_size);
            local_cfg.smooth_angle_deg = std::min(10.0f, cfg.smooth_angle_deg + 2.0f);
            local_cfg.max_planes = std::max(cfg.max_planes, 240);
            std::vector<PlanePatch> local_patches =
                extract_all_seed_grown_patches(
                    surfels, neighborhoods, local_cfg, local_capacity_expansions);
            local_candidate_count = local_patches.size();

            patches = std::move(coarse_patches);
            patches.insert(
                patches.end(),
                std::make_move_iterator(local_patches.begin()),
                std::make_move_iterator(local_patches.end()));

            if (cfg.edge_residual_proposals) {
                std::vector<std::uint8_t> eligible(surfels.size(), 0);
                for (std::size_t i = 0; i < surfels.size(); ++i) {
                    eligible[i] = surfels[i].edge_like ? 1 : 0;
                }
                for (const PlanePatch& patch : patches) {
                    for (int surfel_id : patch.surfels) {
                        eligible[static_cast<std::size_t>(surfel_id)] = 0;
                    }
                }
                residual_boundary_surfels = static_cast<std::size_t>(
                    std::count(eligible.begin(), eligible.end(), static_cast<std::uint8_t>(1)));

                std::vector<Surfel> residual_surfels = surfels;
                for (std::size_t i = 0; i < residual_surfels.size(); ++i) {
                    if (eligible[i] != 0) residual_surfels[i].edge_like = false;
                }
                Config residual_cfg = local_cfg;
                std::vector<PlanePatch> residual_patches = extract_all_seed_grown_patches(
                    residual_surfels,
                    neighborhoods,
                    residual_cfg,
                    residual_capacity_expansions,
                    &eligible);
                residual_novelty = retain_novel_residual_proposals(
                    residual_patches,
                    patches,
                    cloud,
                    surfels,
                    index_by_key,
                    bounds,
                    cfg);
                for (PlanePatch& patch : residual_patches) {
                    patch.requires_local_support = true;
                }
                residual_candidate_count = residual_patches.size();
                patches.insert(
                    patches.end(),
                    std::make_move_iterator(residual_patches.begin()),
                    std::make_move_iterator(residual_patches.end()));
            }
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
            ? refine_patch_ownership_joint(patches, surfels, neighborhoods, cfg)
            : refine_patch_ownership(patches, surfels, neighborhoods, cfg);
        now = Clock::now();
        timings.competition_ms = elapsed_ms(stage_start, now);
        stage_start = now;
        const std::size_t competed_patch_count = patches.size();
        global_family_merges += merge_patches_adaptive(patches, surfels, cfg, global_normals);
        now = Clock::now();
        timings.final_merge_ms = elapsed_ms(stage_start, now);
        stage_start = now;
        const RelationRefinementStats relation_refinement =
            refine_parallel_relations(patches, surfels, cfg);
        now = Clock::now();
        timings.relation_refinement_ms = elapsed_ms(stage_start, now);
        stage_start = now;
        cfg.assign_margin = estimate_assign_margin(cloud, patches, cfg);
        now = Clock::now();
        timings.assign_margin_ms = elapsed_ms(stage_start, now);
        stage_start = now;
        DenseAssignmentStats dense_assignment = assign_full_points(
            cloud, patches, surfels, index_by_key, neighborhoods, bounds, cfg, true);
        // Isolated conflicts are often unstable edge normals. Require evidence comparable
        // to a fraction of one deliverable patch and to the cloud size before changing labels.
        const std::size_t minimum_guided_evidence = std::max<std::size_t>(
            static_cast<std::size_t>(std::max(cfg.min_output_points, 1)) / 4,
            cloud->size() / 10000);
        const bool topology_guidance_applied = cfg.pipeline == "topology";
        const bool normal_guidance_applied = topology_guidance_applied ||
            (!cfg.distance_only && dense_assignment.established_guided_points >= minimum_guided_evidence);
        if (!normal_guidance_applied) {
            assign_full_points(
                cloud, patches, surfels, index_by_key, neighborhoods, bounds, cfg, false);
        }
        for (std::size_t patch_id = 0; patch_id < patches.size(); ++patch_id) {
            patches[patch_id].pre_finalize_plane_id = static_cast<int>(patch_id);
        }
        now = Clock::now();
        timings.dense_assignment_ms = elapsed_ms(stage_start, now);
        stage_start = now;
        // Region competition must see every extracted model. Filtering or
        // absorbing a small plane first irreversibly removes a legitimate
        // finite-support alternative and biases the ablation.
        FiniteSupportRegionStats support_regions = cfg.finite_support_region_optimization
            ? optimize_finite_support_regions(
                cloud, patches, surfels, bounds, cfg)
            : FiniteSupportRegionStats{};
        now = Clock::now();
        timings.support_region_ms = elapsed_ms(stage_start, now);
        stage_start = now;
        const FinalizeStats finalize_stats = finalize_dense_patches(patches, cfg);
        now = Clock::now();
        timings.finalize_ms = elapsed_ms(stage_start, now);
        stage_start = now;
        FiniteBoundaryPostprocessStats boundary_postprocess = cfg.finite_boundary_postprocess
            ? correct_points_from_raw_geometry(
                cloud, patches, surfels, neighborhoods, cfg)
            : FiniteBoundaryPostprocessStats{};
        now = Clock::now();
        timings.boundary_postprocess_ms = elapsed_ms(stage_start, now);
        stage_start = now;
        const std::vector<int> final_surfel_labels = cfg.finite_support_region_optimization
            ? build_final_surfel_labels(
                patches, surfels.size(), index_by_key, bounds, cfg.surfel_size)
            : std::vector<int>{};
        const std::vector<int>* final_surfel_labels_ptr =
            cfg.finite_support_region_optimization ? &final_surfel_labels : nullptr;
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
                cloud, patches, surfels, neighborhoods, bounds, cfg, final_surfel_labels_ptr);
        } else {
            corner_result.corners = extract_corners_filtered(cloud, patches, bounds, cfg);
        }
        const std::vector<CornerCandidate>& corners = corner_result.corners;
        now = Clock::now();
        timings.corners_ms = elapsed_ms(stage_start, now);
        stage_start = now;

        resolve_final_planes(patches, dense_assignment.corrections);
        resolve_final_planes(patches, boundary_postprocess.corrections);
        resolve_final_planes(patches, support_regions.decisions);

        fs::path out_dir = cfg.output_dir / cfg.input.stem();
        fs::create_directories(out_dir);
        save_plane_patches(
            out_dir / "edge_aware_plane_patches.ply", patches, cfg.binary_plane_output);
        save_plane_parameters_csv(out_dir / "plane_parameters.csv", patches);
        if (cfg.edge_residual_proposals) {
            save_proposal_diagnostics(out_dir / "proposal_diagnostics.csv", patches, surfels);
        }
        save_feature_edges(
            out_dir / "feature_edges.ply", surfels, neighborhoods, patches,
            final_surfel_labels_ptr);
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
        if (cfg.finite_boundary_postprocess) {
            save_topology_corrections(
                out_dir / "finite_boundary_corrections.ply",
                out_dir / "finite_boundary_corrections.csv",
                boundary_postprocess.corrections);
        }
        if (cfg.finite_support_region_optimization) {
            save_finite_support_region_decisions(
                out_dir / "finite_support_region_corrections.ply",
                out_dir / "finite_support_region_corrections.csv",
                support_regions.decisions);
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
            std::cout << "Cached neighbor references/storage MiB: "
                      << neighborhood_entries << '/'
                      << static_cast<double>(neighborhood_storage_bytes) / (1024.0 * 1024.0) << "\n";
            std::cout << "Global normal families/reliable surfels: "
                      << global_normals.directions.size() << '/' << global_normals.reliable_surfels << "\n";
            std::cout << "Coarse/local candidate patches: "
                      << coarse_candidate_count << '/' << local_candidate_count << "\n";
            std::cout << "Coarse/local proposal capacity expansions: "
                      << coarse_capacity_expansions << '/' << local_capacity_expansions << "\n";
            std::cout << "Residual boundary surfels/candidates/capacity expansions: "
                      << residual_boundary_surfels << '/' << residual_candidate_count << '/'
                      << residual_capacity_expansions << "\n";
            std::cout << "Residual candidates generated/accepted/rejected/max novel estimate: "
                      << residual_novelty.generated << '/' << residual_novelty.accepted << '/'
                      << residual_novelty.rejected << '/'
                      << residual_novelty.maximum_estimated_novel_points << "\n";
            std::cout << "Accepted global-family merges: " << global_family_merges << "\n";
            std::cout << "Parallel relation mode/clusters/planes/max spread deg: "
                      << cfg.parallel_relations << '/' << relation_refinement.clusters << '/'
                      << relation_refinement.constrained_planes << '/'
                      << relation_refinement.maximum_cluster_spread_deg << "\n";
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
                      << ", decisive edge owners: " << dense_assignment.edge_owner_guided_points
                      << ", applied: " << (normal_guidance_applied ? "yes" : "no") << "\n";
            std::cout << "Local-support dense candidates: " << dense_assignment.locally_supported_points << "\n";
            std::cout << "Boundary-side guided dense assignments: "
                      << dense_assignment.topology_guided_points << "\n";
            std::cout << "Boundary-rejected dense candidates: "
                      << dense_assignment.boundary_rejected_candidates << "\n";
            std::cout << "Active plane-side constraints: "
                      << dense_assignment.boundary_constraints << "\n";
            std::cout << "Maximum dense candidates per point: "
                      << dense_assignment.max_candidates_per_point << "\n";
            std::cout << "Occupancy-rejected boundary candidates: "
                      << dense_assignment.occupancy_rejected_candidates << "\n";
            std::cout << "Safely absorbed small patches/points: "
                      << finalize_stats.absorbed_patches << '/' << finalize_stats.absorbed_points << "\n";
            std::cout << "Discarded sparse patch points: " << finalize_stats.discarded_sparse_points << "\n";
            std::cout << "Finite-boundary postprocess intersections/segments/trusted/cells/candidates/competitors/corrections/rejected: "
                      << boundary_postprocess.finite_intersections << '/'
                      << boundary_postprocess.directed_constraints << '/'
                      << boundary_postprocess.trusted_geometry_points << '/'
                      << boundary_postprocess.support_cells << '/'
                      << boundary_postprocess.boundary_candidate_points << '/'
                      << boundary_postprocess.finite_competitors << '/'
                      << boundary_postprocess.corrected_points << '/'
                      << (boundary_postprocess.rejected_without_local_support +
                          boundary_postprocess.margin_rejected_points) << "\n";
            std::cout << "Finite-support regions trusted/occupied/gap-fill/inconsistent/nodes/components/optimized/retained/mixed/local-reject/support-guard/corrections: "
                      << support_regions.trusted_geometry_points << '/'
                      << support_regions.occupied_support_cells << '/'
                      << support_regions.inferred_sampling_gap_cells << '/'
                      << support_regions.support_inconsistent_points << '/'
                      << support_regions.competition_nodes << '/'
                      << support_regions.competition_components << '/'
                      << support_regions.optimized_components << '/'
                      << support_regions.retained_components << '/'
                      << support_regions.mixed_surfel_points << '/'
                      << support_regions.local_geometry_rejections << '/'
                      << support_regions.support_preservation_rejections << '/'
                      << support_regions.corrected_points << "\n";
            std::cout << "Finite-support region normal mode: "
                      << cfg.region_normal_estimator << "\n";
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
            std::cout << "Timing global/merge1/competition/merge2/relations (ms): "
                      << timings.global_relations_ms << '/' << timings.initial_merge_ms << '/'
                      << timings.competition_ms << '/' << timings.final_merge_ms << '/'
                      << timings.relation_refinement_ms << "\n";
            std::cout << "Timing margin/dense/support-region/finalize/boundary/audit/corners/save (ms): "
                      << timings.assign_margin_ms << '/' << timings.dense_assignment_ms << '/'
                      << timings.support_region_ms << '/' << timings.finalize_ms << '/'
                      << timings.boundary_postprocess_ms << '/'
                      << timings.audit_ms << '/' << timings.corners_ms << '/'
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
