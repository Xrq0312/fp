# 点云面片与角点提取算法说明报告

## 1. 项目目标

本项目面向规则多面体、冲头、工件局部结构等 DLP 点云数据，目标是从输入点云中自动提取：

- 平面面片：表示物体表面上稳定、可解释的平面区域。
- 多面体角点：表示多个平面交汇处或面片边界折角处的特征点。

代码当前封装入口为 `CornerFeatureDetector`。外部可以通过 `detect(cloud)` 直接输入内存中的 PCL 点云，也可以通过 `detectFile(path)` 输入 PLY 文件。检测结果由 `CornerDetectionResult` 返回，包含最终 `plane_patches` 和 `corners`。

命令行程序只负责参数解析、调用检测器、保存结果，不再承担核心算法逻辑。

## 2. 输入输出

### 输入

输入为 PLY 格式点云，内部转换为 `pcl::PointCloud<pcl::PointXYZ>`，仅使用 XYZ 坐标。

可选预处理包括：

- 坐标裁剪：通过 `--min-x/--max-x/--min-y/--max-y/--min-z/--max-z` 限定检测区域。
- 随机限采样：通过 `--load-limit` 控制最大加载点数。
- 体素降采样：通过 `--voxel-leaf` 控制。
- 统计滤波：通过 `--sor-mean-k` 和 `--sor-stddev` 控制。
- 半径离群滤波：通过 `--ror-radius-scale` 和 `--ror-min-neighbors` 控制。

### 输出

每个输入点云会在输出目录下生成：

- `plane_patches.ply`：检测到的平面面片，按面片随机着色。
- `polyhedron_corners.ply`：检测到的角点，统一以红色点保存。

推荐脚本当前结果为：

- `top_left_cut.ply`：26 个面片，30 个角点。
- `top_down.ply`：17 个面片，8 个角点。

## 3. 总体流程

整体算法分为 5 个阶段：

1. 点云预处理与分辨率估计。
2. 多尺度局部平面原子提取。
3. 平面面片扩展、合并与过滤。
4. 候选角点生成。
5. 角点合并与边缘辅助过滤。

流程主线在 `CornerFeatureDetector::detect_prepared()` 中：

```text
输入点云
  -> prepare_cloud
  -> extract_plane_patches_atoms
  -> expand_patches_per_plane
  -> trim_support_cloud_conflicts
  -> trim_core_cloud_conflicts
  -> filter_weak_visible_patches
  -> extract_corners
  -> 输出面片和角点
```

## 4. 点云预处理与分辨率估计

预处理函数为 `prepare_cloud()`。

首先根据配置执行可选裁剪、采样和滤波，得到最终参与检测的点云。随后调用 `estimate_model_resolution()` 估计模型分辨率。

分辨率估计方法：

1. 从点云中随机采样最多 5000 个点。
2. 建立 KD-Tree。
3. 对每个采样点查找最近邻。
4. 统计最近邻距离均值，作为模型分辨率。

后续大部分阈值都使用“分辨率倍数”定义，例如网格尺度、平面距离阈值、角点支持半径等。这样同一套参数可以适应不同密度或不同尺寸的点云。

## 5. 多尺度局部平面原子提取

面片提取不是直接对全局点云做 RANSAC 或 region growing，而是采用“多尺度局部平面原子”的方式。

核心函数：

- `extract_plane_patches_atoms()`
- `extract_plane_patches_atoms_one_scale()`

### 5.1 网格划分

对每个尺度 `cell_scale`，代码根据：

```text
cell_size = cell_scale * model_resolution
```

把点云划分到三维规则网格中。每个网格单元中如果点数超过 `atom_min_points`，就尝试拟合一个局部平面。

### 5.2 局部平面拟合

每个网格单元使用 PCA 拟合平面：

1. 计算点集质心。
2. 计算协方差矩阵。
3. 对协方差矩阵做特征值分解。
4. 最小特征值对应的特征向量作为平面法向。
5. 平面方程为：

```text
n dot x + d = 0
```

为了降低噪声和边缘混合点的影响，`fit_plane_patch()` 会进行鲁棒残差修剪：

1. 初始 PCA 拟合。
2. 计算每个点到平面的残差。
3. 保留残差较小的前 `robust_fit_keep_ratio` 比例点。
4. 重新 PCA 拟合。
5. 重复 `robust_fit_iterations` 次。

局部平面会经过两个质量检查：

- `atom_max_planarity_error`：基于最小特征值占比的平面性误差。
- `atom_max_rms_scale`：点到平面的 RMS 残差上限。

通过检查的局部平面称为平面原子。

### 5.3 邻接原子合并

平面原子之间只在相邻网格范围内尝试合并。合并条件包括：

- 法向夹角足够小。
- 平面距离足够近。
- 原子所在网格相邻。

代码使用并查集 `DisjointSet` 把兼容原子连成连通分量。每个连通分量合并为一个初始平面候选。

多尺度结果最后会再次调用 `merge_coplanar_patches()` 合并共面面片。

## 6. 平面面片扩展、合并与过滤

局部原子只能覆盖平面的一部分，因此需要把种子平面扩展到完整点云。

核心函数：

- `make_plane_envelope()`
- `point_matches_envelope()`
- `expand_patches_per_plane()`
- `finalize_plane_patches()`

### 6.1 平面包络构建

对每个种子面片，先构建局部二维坐标系：

- `axis_u`：平面内第一个方向。
- `axis_v`：平面内第二个方向。
- 法向 `normal`：平面的垂直方向。

随后把种子点投影到 `u-v` 平面，记录二维包围盒：

```text
min_u, max_u, min_v, max_v
```

### 6.2 全点云扩展

对完整点云中的每个点，调用 `point_matches_envelope()` 判断是否属于该平面。

判断条件有两个：

1. 点到平面的距离小于 `expand_plane_distance_scale * resolution`。
2. 点投影到平面二维坐标后，落在种子面片二维包围盒加 margin 的范围内。

这里的 margin 由 `expand_plane_margin_scale` 控制。它允许种子面片向周围扩展，但不会无限吸收同一平面上距离很远、实际不相邻的区域。

### 6.3 严格重拟合

扩展得到候选点集后，调用 `fit_strict_plane_patch()`：

1. 先对候选点拟合平面。
2. 使用 `strict_plane_distance_scale * resolution` 过滤严格内点。
3. 对严格内点重新拟合。
4. 原候选点作为 `support_cloud` 保存，用于后续可见支持和输出。

这样可以让平面参数由核心内点决定，同时保留更完整的可视化支持区域。

### 6.4 共面合并

扩展后再次进行共面合并。合并条件为：

- 法向接近。
- 平面距离接近。
- 二维投影包围盒相邻或重叠。

`merge_plane_adjacency_scale` 用来避免把空间上分离但数学上共面的两个平面错误合成一个。

### 6.5 形状过滤

`is_valid_plane_patch_shape()` 会过滤过窄的边缘条带。判断依据：

- 投影短边长度必须大于 `min_plane_extent_scale * resolution`。
- 面片长宽比必须大于 `min_plane_aspect_ratio`。

这一步主要用于去除边缘附近形成的伪平面。

### 6.6 冲突点修剪

平面之间在边缘、角点、遮挡处容易互相抢点。代码做了两级修剪：

- `trim_support_cloud_conflicts()`：修剪显示/支持点云中更接近其他非平行平面的点。
- `trim_core_cloud_conflicts()`：修剪严格核心点，并在必要时重新拟合平面。

如果一个点到另一个非平行平面的距离明显更近，或两个平面的距离非常接近，则认为它是边界竞争点，不应该强行归入当前平面。

### 6.7 弱面片过滤

`filter_weak_visible_patches()` 会基于场景自适应统计删除弱面片。

它会统计所有面片的：

- 可见支持点数。
- 投影面积上的点密度。

然后使用中位数构造自适应阈值。若某个面片可见支持过少，且相对自身核心或场景密度都很弱，则删除。

这一步用于减少噪声、遮挡边界或薄片结构产生的假平面。

## 7. 角点提取

角点提取由两类候选组成：

1. 三平面交点候选。
2. 单个面片边界凸包角候选。

核心函数：

- `extract_corners()`
- `intersection_of_three_planes()`
- `extract_patch_boundary_corners()`
- `merge_candidates()`
- `CornerEdgeScorer`

### 7.1 三平面交点

对所有非平行的三平面组合，求解线性方程：

```text
n1 dot x + d1 = 0
n2 dot x + d2 = 0
n3 dot x + d3 = 0
```

得到候选交点后进行检查：

- 三个平面不能近似共线或奇异。
- 交点必须落在点云整体包围盒附近。
- 交点到三个平面的距离不能过大。
- 交点附近必须有足够点云支持。
- 每个参与平面在交点附近必须有足够局部支持。

通过后生成角点候选，记录其对应的平面编号。

### 7.2 面片边界凸包角

对于一些外轮廓角点，可能不是由三个稳定平面交汇得到，而是来自可见面片边界。因此代码额外提取面片二维凸包角。

步骤：

1. 将面片点投影到该平面的二维坐标系。
2. 计算二维凸包。
3. 遍历凸包顶点，检查两侧边长度和夹角。
4. 检查点云支持数量。
5. 检查附近有多少个面片提供局部支持。

满足条件后，将凸包折角作为候选角点加入候选集。

### 7.3 候选合并

`merge_candidates()` 会把距离很近的候选角点合并。合并半径为：

```text
intersection_merge_scale * resolution
```

合并时保留支持点数更高的位置，并合并平面编号集合。

随后根据 `min_corner_planes` 删除平面支持数量不足的候选。

### 7.4 边缘辅助过滤

`CornerEdgeScorer` 使用局部协方差特征值对候选角点进行过滤。

在候选点附近建立局部邻域，计算协方差矩阵特征值：

```text
lambda_min / (lambda1 + lambda2 + lambda3)
```

该值反映局部点云是否表现为明显的边缘/角点/散乱结构。代码支持：

- `min_corner_edge_score`：过滤过于平坦、不像角点的候选。
- `max_corner_edge_score`：过滤过于散乱或疑似遮挡边界的候选。
- `max_corner_edge_growth_ratio`：过滤大尺度分数相对中尺度增长过快的候选。

这一步相当于借鉴边缘提取思路，但只作用于最终候选角点，而不是对全点云做昂贵的全局边缘检测。

## 8. 关键设计思路

### 8.1 为什么不用单一邻域协方差直接找角点

点云密度高、边界遮挡和局部噪声都会让纯局部协方差方法不稳定。当前算法先提取平面，再由平面结构反推角点。这样角点位置由全局几何关系约束，不完全依赖局部点云纹理。

### 8.2 为什么使用多尺度 atom

单一网格尺度容易出现两个问题：

- 网格太小：每个局部平面点数不足，噪声敏感。
- 网格太大：跨越边缘或多个平面，拟合结果混杂。

多尺度 atom 能同时覆盖细小面片和大面片，提高不同点云密度下的稳定性。

### 8.3 为什么要做投影邻接判断

仅凭法向和平面距离判断共面，容易把空间上分离但数学上共面的区域错误合并。通过二维投影包围盒邻接判断，可以保留共面但不连续结构之间的分界。

### 8.4 为什么要修剪平面冲突点

角点和棱边附近的点天然靠近多个平面。如果这些点被强行归入某个平面，会导致：

- 平面法向被拉偏。
- 平面边界扩张过度。
- 角点交点位置漂移。

因此代码对支持点和核心点分别做冲突修剪，提升面片纯净度。

## 9. 主要参数说明

| 参数 | 作用 |
| --- | --- |
| `atom_cell_scales` | 多尺度网格大小，单位为模型分辨率倍数 |
| `atom_min_points` | 一个网格内形成平面原子的最少点数 |
| `atom_max_planarity_error` | 局部原子的平面性误差上限 |
| `atom_max_rms_scale` | 局部原子的 RMS 残差上限 |
| `merge_plane_angle_deg` | 共面合并时允许的法向角度差 |
| `merge_plane_distance_scale` | 共面合并时允许的平面距离差 |
| `merge_plane_adjacency_scale` | 共面合并时投影邻接 margin |
| `expand_plane_distance_scale` | 平面扩展时点到平面的距离阈值 |
| `expand_plane_margin_scale` | 平面扩展时投影包围盒外扩 margin |
| `strict_plane_distance_scale` | 严格重拟合时的内点距离阈值 |
| `support_conflict_scale` | 支持点冲突修剪阈值 |
| `core_conflict_scale` | 核心点冲突修剪阈值 |
| `min_visible_support_*` | 自适应弱面片过滤参数 |
| `intersection_merge_scale` | 角点候选合并半径 |
| `corner_support_radius_scale` | 角点附近点云支持搜索半径 |
| `min_corner_support` | 候选角点附近最少点数 |
| `min_plane_local_support` | 每个参与平面在角点附近的最少支持点数 |
| `min_corner_planes` | 合并后角点至少需要多少个平面支持 |
| `corner_edge_*` | 基于局部协方差的边缘辅助过滤参数 |

## 10. 当前算法特点

### 优点

- 以平面结构为主导，角点结果可解释。
- 参数大多基于模型分辨率缩放，适应不同密度点云。
- 多尺度局部 atom 对大面、小面都有一定鲁棒性。
- 共面合并引入投影邻接，减少错误跨区域合并。
- 角点同时考虑三平面交点和边界凸包角，覆盖更完整。
- 边缘协方差只用于最终候选过滤，计算量比全点云边缘检测小。

### 局限

- 如果真实表面不是平面，或者面片严重弯曲，算法会倾向于分裂或误判。
- 如果某个角点附近点云缺失严重，支持点检查可能会删除该角点。
- 如果点云存在大量共面但断开的结构，需要仔细设置 `merge_plane_adjacency_scale`。
- 对非常密集的数据，局部协方差和支持搜索半径需要适当增大，否则角点过滤可能过于敏感。
- 目前平面扩展是逐平面对全点云扫描，点数很大时计算量较高。

## 11. 推荐结果与现有配置

当前推荐脚本为：

```powershell
.\scripts\run_recommended.ps1
```

它分别处理：

- `data01\data01\top_left_cut.ply`
- `data02\data02\top_down.ply`

当前验证结果：

| 数据 | 面片数 | 角点数 |
| --- | ---: | ---: |
| `top_left_cut.ply` | 26 | 30 |
| `top_down.ply` | 17 | 8 |

其中 `top_down` 使用了更小的 atom cell scale 和更严格的边缘辅助角点过滤，适合该数据中点云较密、局部面片更容易碎裂的情况。

## 12. 总结

本算法的核心思想是：

```text
先稳定提取平面，再由平面几何关系定位角点，最后用局部边缘证据过滤伪角点。
```

相比直接在全点云上做角点检测，这种方法更适合具有明显平面结构的工件点云。它把问题拆成“面片稳定性”和“角点几何一致性”两个阶段，使最终结果更可解释，也方便针对不同数据通过少量参数进行调节。
