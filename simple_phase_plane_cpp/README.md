# Simple phase-plane unwrap

这是从调试代码中整理出的独立最小版本，只包含：

1. 读取单张 8 位包裹相位图，或从目录中的 `1..4` 四步图解码相位；
2. 在指定 ROI 内用 32 像素基线的包裹相位差自动估计载频方向；
3. 自动推断候选点的整数周期 K，并按 TXT 的 `|phase| < 0.6*pi` 规则选择拟合点；
4. 用归一化坐标拟合一次相位平面；
5. 用平面逐像素取整 K，输出解相位、残差和报告。

没有保留旧调试程序中的手工 anchor 接口、二次曲面、质量引导、连通域合并、周期误差校正、配置解析和 GUI。

## Build

```powershell
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=D:/vcpkg/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build build --config Release
```

## Run

```powershell
.\build\Release\simple_phase_plane_unwrap.exe <phase-image-or-raw-dir> <output-dir> [x0 y0 x1 y1]
```

不提供 ROI 参数时默认使用半开区间 `(100,1200)-(4701,3648)`。RAW 目录使用当前 data2 的相位约定：`-atan2(s,c)+pi/2`。

批量测试：

```powershell
.\run_tests.ps1
```
