# 如何运行

所有命令从仓库根目录执行；以下使用 Windows PowerShell。科学计算依赖见 `requirements.txt`。

## 1. 只用随库小数据

```powershell
python -m pip install -r requirements.txt
python tools\check_publication.py
python -m unittest discover -s tests -v
python tools\plot_summaries.py
```

没有网络下载，也不需要 `pyfefi2`。重联摘要图使用逐候选 CSV；日侧边界图使用已经提取的一维时间序列。

## 2. 使用已有私有原始切面

```powershell
$env:MAG_RAW_DIR = "D:\private_simulation\data"
```

该目录应包含：

```text
data/
  eq/
    coords.npz
    f00001.npz
    ...
  mer/
    coords.npz
    f00001.npz
    ...
```

这里只示意布局；仓库不包含这些文件。路径可指向原工作目录的 `data`，无需复制原始数据到仓库。下载器和 `magviz.Plane` 都使用该环境变量；未设置时默认仓库内被忽略的 `data/`。

## 3. 如需重新下载

先确认拥有数据访问权限、服务可用及足够本地空间；此前两面切面约 9 GiB。**下列命令会下载原始数据，不属于默认示例。**

```powershell
python -m pip install -r requirements-download.txt
$env:MAG_RAW_DIR = "D:\private_simulation\data"
python download.py eq 2 4
python download.py mer 4 4
```

参数依次是切面、输出帧步长、并发下载数。帧范围与所取变量见脚本。下载目录永远不应被加入 Git。

## 4. 完整分析的依赖顺序

```powershell
python analysis.py
python snapshots.py eq
python snapshots.py mer
python anatomy.py
python keogram_eq.py
python probes.py
python repro_cache.py
python reproduce.py
python figure_diagnostics.py
python recon_redo.py
python recon_slide_figures.py
python recon_schematic.py
python location_insets.py
```

- `analysis.py` 生成 `output/diagnostics.json`、`output/recon.json`、一维时间序列、时空数组及初始图。
- `repro_cache.py` 从原始输入生成 `output/cache/avg_*.npz` 等本地缓存；`reproduce.py` 依赖它们以及 `analysis.py` 的输出。
- `figure_diagnostics.py` 进一步依赖 `reproduce.py` 的输出。
- `recon_redo.py` 生成原始输入派生的局地缓存及探索性诊断；更新的展示图由 `recon_slide_figures.py` 生成，不能将旧率代理当标准率，也不以未核实电场尺度检查作结论。
- `location_insets.py` 依赖平均场缓存；`recon_schematic.py` 为理论示意图。
- 这些脚本可能覆盖本地同名自绘图和结果。先备份个人修改；生成的原始场缓存保持被忽略。

完整分析未在公开副本中重新下载/计算；归档验证只运行无需原始数据的摘要示例、单元测试和脚本语法检查。

## 5. 动画

安装系统 FFmpeg 并加入 PATH，或设置可执行文件路径：

```powershell
$env:FFMPEG = "C:\Tools\ffmpeg\bin\ffmpeg.exe"
python make_movies.py eq 4
python make_movies.py mer 4
```

动画由 `frames/` 编码到 `output/`；帧和 MP4/GIF 均被忽略，不提交。`make_ppt.py` 等依赖个人模板与 Office 的历史组装脚本没有进入公开副本；手工编辑后的 PPT 请使用 README 的现有下载入口。
