# Open · 磁层形成与演化

全球混合模拟 **ear-Reynolds-S** 的赤道面与正午—子夜子午面分析：磁层形成、弓激波、磁鞘、侧翼 KHI、磁尾高速流及局地磁场重联。

**这是经过筛选的公开归档，不是原工作目录的整体上传。严禁将原始数据提交到本仓库。** 原工作目录及现有 PPT 不作移动或覆盖。

## 阅读入口

- [逐图报告：F01–F48](output/report.md)：保留文字与自绘图；9 张含教材截图的对照拼图不重新分发，图号和分析保留。
- [22 页文字讲稿](output/slides/script.md)：面向转专业新生；仓库版已补正 F10 灰带及第 14 页线型解释。
- [完整报告网页](http://zsyq.hxlab.tech:7654/view/magnetosphere-hybrid-v10) · [带截图的 PPT 讲稿及下载入口](http://zsyq.hxlab.tech:7654/view/magnetosphere-hybrid-v10/script.html)：完整 PPT 和视频留在现有托管站，不纳入 Git。仓库中的文字修订尚未同步回该历史网页。
- [小数据说明](processed/README.md) · [复现步骤](docs/REPRODUCING.md) · [文件与数据边界](docs/DATA_POLICY.md) · [来源与限制](docs/PROVENANCE.md)。

## 目录

| 路径 | 内容 |
|---|---|
| 根目录 `*.py` | 数据读取、下载、分析、自绘图、动画编码与诊断脚本；保持原模块导入关系 |
| `processed/` | 3 个一维缩减 NPZ、3 个诊断 JSON、90 行候选统计 CSV 与重联摘要 |
| `output/report.md`、`output/figs/` | 逐图说明、自绘结果及位置示意图 |
| `output/slides/script.md` | 22 页文字讲稿；PPT 本体不入库 |
| `tools/` | 无原始数据的摘要画图、提交前检查与清单更新 |
| `tests/` | 数据边界和小数据统计的一致性检查 |
| `publication_manifest.json` | 允许公开的文件清单、分类、字节数和 SHA-256 |

不包含：原始切面 `data/`、逐帧图像 `frames/`、二维平均图缓存、pickle、教材/论文 PDF、PPT、下载凭据、浏览器状态或临时构建目录。排除规则之外，提交前检查还使用**精确文件白名单和三个 NPZ 的键/形状校验**；不能只依赖扩展名判断。

## 不下载原始数据即可运行

在仓库根目录执行，Python 3.10+：

```powershell
python -m pip install -r requirements.txt
python tools\check_publication.py
python -m unittest discover -s tests -v
python tools\plot_summaries.py
```

最后一条命令仅使用 `processed/`，生成被 Git 忽略的 `output/figs/summary_boundaries.png` 与 `summary_hall.png`。**上述命令均不下载原始数据。**

需要重新处理完整切面时，先阅读[复现步骤](docs/REPRODUCING.md)。多数分析脚本会在运行时读取本地原始数据，不可将“脚本语法检查通过”理解为完整模拟复现已运行。

## 主要证据及边界

- 晚期日下点磁层顶约 11.6、弓激波约 14.2 个地球半径；边界判据和时间窗口见报告。
- Hall 四象限符号：独立按喷流等条件筛选的 **28 帧中 21 帧**满足；全部 **90 帧候选中 27 帧**满足。输出时刻相关，不能当作独立事件发生率，更不能当作整个磁层的比例。
- 二维切面、局地符号及喷流证据不能独立证明三维拓扑或电子扩散区。
- 旧输出中的 `Ey/(B_L V_out)` 不是标准的 `Ey/(B_L V_A)`，不得用它直接比较通常所说的 0.1 重联率。本仓库小数据省略该代理及未独立核实的 Hall 电场尺度检查。

## 提交前

```powershell
python tools\check_publication.py --write-manifest
git add -- .gitignore README.md requirements*.txt *.py docs processed output tools tests publication_manifest.json
python tools\check_publication.py --staged
git diff --cached --stat
```

清单更新只接受白名单路径；新增文件需显式修改检查器中的公开边界。原始数据即使被强制 `git add -f`，暂存检查也必须拒绝。Git 历史一旦包含原始数据，后续删除文件不能消除泄露，禁止先提交再清理。
