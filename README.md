# MrGap reproduction and manifold-fitting benchmark

这个仓库以 Dunson and Wu 的 **Manifold Reconstruction via Gaussian
Processes (MrGap)** 官方 MATLAB 仓库为基础，增加了 Python 复现、与 Yao
等人的 **Manifold Fitting** 的统一比较、诊断实验以及 Beamer 汇报。

本 README 的首要目的，是明确区分上游原始材料和本仓库新增内容。除非特别
标明，新增 Python 代码不是两篇论文作者发布的官方实现。

## 代码与数据来源

| 路径 | 来源 | 说明 |
|---|---|---|
| `Mfit1.m` | **Dunson--Wu 原始代码** | MrGap Algorithm 1 的局部 GP 去噪步骤；`A,rho,sig` 由调用者传入 |
| `Mfit2.m` | **Dunson--Wu 原始代码** | MrGap Algorithm 2 的插值步骤 |
| `Cassini oval.mat` | **Dunson--Wu 原始数据** | Cassini noisy sample `X` 与评价参考集 `M11` |
| `RP3.mat` | **Dunson--Wu 原始数据/输出** | noisy data `X`、两轮去噪结果 `M0/M1` 与插值 `M2` |
| `Vocal1.mat`, `Vocal2.mat` | **Dunson--Wu 原始数据/输出** | bird-vocalization spectrogram examples |
| `half torus bounded.mat` | **Dunson--Wu 原始数据** | 带边界、非各向同性噪声的 half-torus example |
| `experiments/` | **本仓库新增** | Cassini 的透明 Python 复现及 noise/sample-size sensitivity |
| `benchmark/` | **本仓库新增** | MrGap、oracle-tangent MrGap 与 Manifold Fitting 的统一 benchmark |
| `results/` | **本仓库新增结果** | CSV、metadata、诊断报告与可复现图表 |
| `Manifold estimation via gp/` | **本仓库新增汇报** | Beamer 源文件、图和编译后的汇报 PDF |

原始 MrGap 仓库：
[wunan3/Manifold-reconstruction-via-Gaussian-processes](https://github.com/wunan3/Manifold-reconstruction-via-Gaussian-processes)
（本仓库当前 Git 历史中的 MATLAB 文件和 `.mat` 文件来自该代码发布）。

## Manifold Fitting 在哪里？

Yao 等人的官方 MATLAB 实现位于：

> [zhigang-yao/manifold-fitting — Manifold Fitting/Matlab](https://github.com/zhigang-yao/manifold-fitting/tree/master/Manifold%20Fitting/Matlab)

本仓库没有把对方的整个 MATLAB 目录复制进来。`benchmark/manifold_benchmark.py`
中的 `manifold_fitting` 是根据官方 `manfit_ours.m` 写成的 Python 行为移植，
包括 cylindrical neighborhood averaging、少样本时的 nearest-neighbor fallback
及可选 final averaging。它用于统一数据生成、计时和误差评估，不应被称为 Yao
等人发布的官方 Python 包。

## 新增目录

```text
.
├── Mfit1.m, Mfit2.m, *.mat        # Dunson--Wu 上游原始文件
├── experiments/
│   └── cassini_sensitivity.py     # Cassini 小型复现
├── benchmark/
│   ├── manifold_benchmark.py      # 统一模拟与方法比较
│   └── README.md                  # 参数、公平性和运行说明
├── results/
│   ├── cassini/                   # Cassini sensitivity 输出
│   └── benchmark/                 # 统一 benchmark 输出与讨论
└── Manifold estimation via gp/
    ├── mrgap_presentation.tex
    ├── mrgap_presentation.pdf
    └── figures/
```

## Benchmark 覆盖范围

统一 benchmark 包括：

- Cassini oval、`RP(3)`、torus 和 half-torus；
- ambient Gaussian noise
  `sigma = [0, .01, .02, .04, .06, .08, .12]`；
- 独立生成的样本量
  `n = [50, 100, 250, 500, 1000, 5000]`；
- MrGap 第 1--5 轮、Manifold Fitting、oracle-tangent MrGap；
- noise、sample size、local-neighborhood/bandwidth 和 runtime sensitivity；
- geometric RMSE、paired RMSE、局部样本量、运行时间与近似内存诊断。

Pilot 结果见 [`results/benchmark/REPORT.md`](results/benchmark/REPORT.md)，解释与
限制见 [`results/benchmark/DISCUSSION.md`](results/benchmark/DISCUSSION.md)。

### 当前最重要的限制

MrGap 论文要求每轮通过所有局部回归的 summed log marginal likelihood 重新估计
`(A,rho,sigma)`，并在相邻两轮的 `sigma` 变化足够小时停止。然而公开的
`Mfit1.m` 只接收这些参数，并未包含 MLE optimizer 或完整 experiment driver。

因此当前 iteration scan：

- 第 1 轮使用保存的 first-round tuple；
- 第 2--5 轮复用保存的 final-round tuple；
- 衡量的是 fixed-parameter denoising map 被重复应用时的敏感性；
- **不是** 对论文 MLE stopping rule 的复现。

Cassini 和 torus 的 tuple 与论文正文报告值一致；RP(3) 和 half-torus 在 benchmark
中谨慎标记为 reproduction settings。详细说明见
[`benchmark/README.md`](benchmark/README.md)。

另一个不对称是 Manifold Fitting 按其官方公式接收 simulation 的真实 `sigma`，
而 MrGap 使用固定 example parameters。比较应作为 failure-mode diagnosis，而不是
简单的排行榜。

## 快速运行

需要 Python 3.10+，以及：

```text
numpy
scipy
matplotlib
```

从仓库根目录运行 Cassini 实验：

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache python \
  experiments/cassini_sensitivity.py --repeats 10
```

运行统一 pilot benchmark：

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache python \
  benchmark/manifold_benchmark.py --profile pilot
```

完整模式默认使用三个 replicates，并尝试 exact MrGap `n=5000`，计算成本明显更高：

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache python \
  benchmark/manifold_benchmark.py --profile full
```

可用 `--manifolds cassini torus` 只运行部分 manifold。所有输出都写入
`results/`，不会从固定的 100000 点 observation cloud 中取 prefix；不同 `n`
对应独立生成的观察样本。

## 已提交的结果

仓库保留轻量、便于审阅的结果：

- `results/benchmark/benchmark_rows.csv`：pilot 的逐行诊断数据；
- `results/benchmark/*.png`：noise、sample size、local count、bandwidth、oracle
  tangent、iteration 和 runtime 图；
- `REPORT.md`、`DISCUSSION.md` 与 `metadata.json`；
- Cassini sensitivity 的 CSV/图；
- 汇报的 `.tex`、所需图片和编译后的 PDF。

Python cache、macOS metadata、LaTeX 辅助文件及本地参考论文 PDF 不纳入版本控制。

## 引用

- David B. Dunson and Nan Wu. *Inferring manifolds using Gaussian processes*.
  Biometrika, 2026. Earlier version: [arXiv:2110.07478](https://arxiv.org/abs/2110.07478).
- Zhigang Yao, Jiaji Su, Bingjie Li, and Shing-Tung Yau. *Manifold Fitting*.
  [arXiv:2304.07680](https://arxiv.org/abs/2304.07680), 2023.

使用或再发布上游代码与数据时，请同时检查并遵守相应原始仓库及论文的许可与
引用要求。本 README 的 provenance 标记不改变任何上游文件的版权归属。
