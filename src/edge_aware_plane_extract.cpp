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
    float surfel_size = 0.20f;
    int normal_k = 24;
    float smooth_angle_deg = 7.0f;
    float edge_angle_deg = 18.0f;
    float coplanar_distance = 0.18f;
    float merge_angle_deg = 4.0f;
    float merge_distance = 0.28f;
    float merge_gap = 0.60f;
    float assign_distance = 0.24f;
    float assign_margin = 0.45f;
    bool visibility_filter = false;
    float visibility_bin = 0.8f;
    bool local_support_only = false;
    int support_halo = 1;
    float support_normal_halo = 0.60f;
    int min_surfels = 80;
    int min_output_points = 5000;
    float min_extent = 3.0f;
    float min_aspect_ratio = 0.02f;
    int max_planes = 120;
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
    bool visible = true;
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
            << "  --merge-*, --assign-*, --visibility-*, --local-support-only,\n"
            << "  --support-halo, --support-normal-halo, --min-surfels, --min-output-points.\n";
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
    cfg.visibility_filter = has_flag(args, "--visibility-filter");
    if (auto v = value_after(args, "--visibility-bin")) cfg.visibility_bin = std::stof(*v);
    cfg.local_support_only = has_flag(args, "--local-support-only");
    if (auto v = value_after(args, "--support-halo")) cfg.support_halo = std::stoi(*v);
    if (auto v = value_after(args, "--support-normal-halo")) cfg.support_normal_halo = std::stof(*v);
    if (auto v = value_after(args, "--min-surfels")) cfg.min_surfels = std::stoi(*v);
    if (auto v = value_after(args, "--min-output-points")) cfg.min_output_points = std::stoi(*v);
    if (auto v = value_after(args, "--min-extent")) cfg.min_extent = std::stof(*v);
    if (auto v = value_after(args, "--min-aspect-ratio")) cfg.min_aspect_ratio = std::stof(*v);
    if (auto v = value_after(args, "--max-planes")) cfg.max_planes = std::stoi(*v);
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

static void basis_for_direction(const Eigen::Vector3f& direction, Eigen::Vector3f& axis_u, Eigen::Vector3f& axis_v) {
    const Eigen::Vector3f reference =
        std::abs(direction.z()) < 0.9f ? Eigen::Vector3f::UnitZ() : Eigen::Vector3f::UnitY();
    axis_u = direction.cross(reference).normalized();
    axis_v = direction.cross(axis_u).normalized();
}

static std::vector<Eigen::Vector3f> visibility_directions() {
    std::vector<Eigen::Vector3f> directions;
    for (int x = -1; x <= 1; ++x) {
        for (int y = -1; y <= 1; ++y) {
            for (int z = -1; z <= 1; ++z) {
                if (x == 0 && y == 0 && z == 0) continue;
                Eigen::Vector3f d(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z));
                directions.push_back(d.normalized());
            }
        }
    }
    return directions;
}

static void mark_visible_surfels(std::vector<Surfel>& surfels, const Config& cfg) {
    if (!cfg.visibility_filter) {
        for (auto& surfel : surfels) surfel.visible = true;
        return;
    }

    for (auto& surfel : surfels) surfel.visible = false;
    const float bin_size = std::max(cfg.visibility_bin, 1e-4f);
    const auto directions = visibility_directions();

    for (const auto& direction : directions) {
        Eigen::Vector3f axis_u;
        Eigen::Vector3f axis_v;
        basis_for_direction(direction, axis_u, axis_v);

        struct Best {
            float depth = -std::numeric_limits<float>::infinity();
            int index = -1;
        };
        std::unordered_map<BinKey, Best, BinKeyHash> best_by_bin;
        best_by_bin.reserve(surfels.size() / 2);

        for (std::size_t i = 0; i < surfels.size(); ++i) {
            const Eigen::Vector3f p = surfels[i].centroid;
            const int u = static_cast<int>(std::floor(p.dot(axis_u) / bin_size));
            const int v = static_cast<int>(std::floor(p.dot(axis_v) / bin_size));
            const float depth = p.dot(direction);
            auto& best = best_by_bin[BinKey{u, v}];
            if (depth > best.depth) {
                best.depth = depth;
                best.index = static_cast<int>(i);
            }
        }

        for (const auto& [key, best] : best_by_bin) {
            if (best.index >= 0) surfels[static_cast<std::size_t>(best.index)].visible = true;
        }
    }
}

static bool compatible_surfel_pair(const Surfel& a, const Surfel& b, const Config& cfg) {
    const float smooth_cos = std::cos(cfg.smooth_angle_deg / 180.0f * 3.14159265358979323846f);
    if (std::abs(a.normal.dot(b.normal)) < smooth_cos) return false;

    const float da = std::abs(a.normal.dot(b.centroid) + a.d);
    const float db = std::abs(b.normal.dot(a.centroid) + b.d);
    return da <= cfg.coplanar_distance && db <= cfg.coplanar_distance;
}

static void mark_edge_surfels_and_union(
    std::vector<Surfel>& surfels,
    const std::unordered_map<CellKey, int, CellKeyHash>& index_by_key,
    const Config& cfg,
    DisjointSet& dsu) {
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
                    const bool compatible = compatible_surfel_pair(surfels[i], surfels[static_cast<std::size_t>(j)], cfg);
                    if (compatible && !angle_edge) {
                        dsu.unite(static_cast<int>(i), j);
                    } else if (angle_edge) {
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

static std::vector<PlanePatch> components_to_patches(const std::vector<Surfel>& surfels, DisjointSet& dsu, const Config& cfg) {
    std::unordered_map<int, std::vector<int>> components;
    for (std::size_t i = 0; i < surfels.size(); ++i) {
        if (surfels[i].edge_like || !surfels[i].visible) continue;
        components[dsu.find(static_cast<int>(i))].push_back(static_cast<int>(i));
    }

    std::vector<PlanePatch> patches;
    for (auto& [root, ids] : components) {
        if (ids.size() < static_cast<std::size_t>(std::max(cfg.min_surfels, 3))) continue;
        PlanePatch patch = fit_patch_from_surfels(ids, surfels);
        update_patch_envelope(patch, surfels);
        if (valid_shape(patch, cfg)) patches.push_back(std::move(patch));
    }
    std::sort(patches.begin(), patches.end(), [](const PlanePatch& a, const PlanePatch& b) {
        return a.surfels.size() > b.surfels.size();
    });
    if (patches.size() > static_cast<std::size_t>(std::max(cfg.max_planes, 1))) {
        patches.resize(static_cast<std::size_t>(std::max(cfg.max_planes, 1)));
    }
    return patches;
}

static bool patches_can_merge(const PlanePatch& a, const PlanePatch& b, const Config& cfg) {
    const float normal_cos = std::cos(cfg.merge_angle_deg / 180.0f * 3.14159265358979323846f);
    const float dot = a.normal.dot(b.normal);
    if (std::abs(dot) < normal_cos) return false;
    const float b_d = dot >= 0.0f ? b.d : -b.d;
    if (std::abs(a.d - b_d) > cfg.merge_distance) return false;

    Eigen::Vector3f axis_u;
    Eigen::Vector3f axis_v;
    plane_basis(a, axis_u, axis_v);
    auto range_for = [&](const PlanePatch& p) {
        std::array<float, 4> r{
            std::numeric_limits<float>::max(),
            std::numeric_limits<float>::lowest(),
            std::numeric_limits<float>::max(),
            std::numeric_limits<float>::lowest()
        };
        for (const auto corner : {Eigen::Vector2f(p.min_u, p.min_v), Eigen::Vector2f(p.min_u, p.max_v),
                                  Eigen::Vector2f(p.max_u, p.min_v), Eigen::Vector2f(p.max_u, p.max_v)}) {
            (void)corner;
        }
        return r;
    };
    (void)range_for;

    const bool u_touch = !(a.max_u + cfg.merge_gap < b.min_u || b.max_u + cfg.merge_gap < a.min_u);
    const bool v_touch = !(a.max_v + cfg.merge_gap < b.min_v || b.max_v + cfg.merge_gap < a.min_v);
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

using CellSet = std::unordered_set<CellKey, CellKeyHash>;

static std::vector<CellSet> build_patch_support_cells(
    const std::vector<PlanePatch>& patches,
    const std::vector<Surfel>& surfels,
    const Config& cfg) {
    std::vector<CellSet> support(patches.size());
    const int halo = std::max(cfg.support_halo, 0);
    const int kernel = 2 * halo + 1;
    const std::size_t reserve_factor = static_cast<std::size_t>(kernel * kernel * kernel);
    const float normal_limit = std::max(cfg.support_normal_halo, 0.0f) * cfg.surfel_size;
    for (std::size_t patch_id = 0; patch_id < patches.size(); ++patch_id) {
        support[patch_id].reserve(patches[patch_id].surfels.size() * reserve_factor);
        for (const int surfel_id : patches[patch_id].surfels) {
            const CellKey base = surfels[static_cast<std::size_t>(surfel_id)].key;
            for (int dx = -halo; dx <= halo; ++dx) {
                for (int dy = -halo; dy <= halo; ++dy) {
                    for (int dz = -halo; dz <= halo; ++dz) {
                        const Eigen::Vector3f offset(
                            static_cast<float>(dx) * cfg.surfel_size,
                            static_cast<float>(dy) * cfg.surfel_size,
                            static_cast<float>(dz) * cfg.surfel_size);
                        if (std::abs(offset.dot(patches[patch_id].normal)) > normal_limit) continue;
                        support[patch_id].insert({base.x + dx, base.y + dy, base.z + dz});
                    }
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
    const std::vector<CellSet> support = cfg.local_support_only ? build_patch_support_cells(patches, surfels, cfg) : std::vector<CellSet>{};

    for (const auto& p : cloud->points) {
        const Eigen::Vector3f q(p.x, p.y, p.z);
        const CellKey key = cfg.local_support_only ? point_key(p, bounds, cfg.surfel_size) : CellKey{};
        int best = -1;
        float best_distance = std::numeric_limits<float>::max();
        for (std::size_t i = 0; i < patches.size(); ++i) {
            if (cfg.local_support_only && support[i].find(key) == support[i].end()) continue;
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

int main(int argc, char** argv) {
    try {
        const Config cfg = parse_args(argc, argv);
        CloudPtr cloud = load_cloud(cfg.input, cfg.input_scale);
        const Bounds bounds = compute_bounds(cloud);

        std::unordered_map<CellKey, int, CellKeyHash> index_by_key;
        CloudPtr surfel_cloud;
        std::vector<Surfel> surfels = build_surfels(cloud, bounds, cfg, index_by_key, surfel_cloud);
        estimate_surfel_normals(surfels, surfel_cloud, cfg);
        mark_visible_surfels(surfels, cfg);

        DisjointSet dsu(surfels.size());
        mark_edge_surfels_and_union(surfels, index_by_key, cfg, dsu);
        std::vector<PlanePatch> patches = components_to_patches(surfels, dsu, cfg);
        merge_patches(patches, surfels, cfg);
        assign_full_points(cloud, bounds, surfels, patches, cfg);

        fs::path out_dir = cfg.output_dir / cfg.input.stem();
        fs::create_directories(out_dir);
        pcl::io::savePLYFileASCII((out_dir / "edge_aware_plane_patches.ply").string(), *make_colored_patch_cloud(patches));
        pcl::io::savePLYFileASCII((out_dir / "edge_surfels.ply").string(), *make_edge_cloud(surfels));
        save_report(out_dir / "edge_aware_plane_report.csv", patches);

        if (cfg.verbose) {
            std::size_t output_points = 0;
            for (const auto& patch : patches) output_points += patch.output->size();
            std::cout << "Input points: " << cloud->size() << "\n";
            std::cout << "Surfels: " << surfels.size() << "\n";
            if (cfg.visibility_filter) {
                const auto visible_count = std::count_if(surfels.begin(), surfels.end(), [](const Surfel& surfel) {
                    return surfel.visible;
                });
                std::cout << "Visible surfels: " << visible_count << "\n";
            }
            std::cout << "Plane patches: " << patches.size() << "\n";
            std::cout << "Output points: " << output_points << "\n";
            std::cout << "Saved to " << fs::absolute(out_dir).string() << "\n";
        }
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}
