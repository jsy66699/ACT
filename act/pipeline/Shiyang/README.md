# 模糊测试 / 反例多样性实验

这个分支（`slim-experiments`）只保留跑实验需要的东西：ACT 本体、fuzzing 引擎、
实验 driver 与分析脚本、实验日志、参数说明。**不含任何结果数据和 benchmark 二进制。**

| 文档 | 内容 |
|---|---|
| [`docs/CHANGES_VS_MAIN.md`](docs/CHANGES_VS_MAIN.md) | **与上游 `main` 的差异说明**：58 个改动文件逐个说明为什么改；默认参数下只有 1 个文件会改变上游行为 |
| [`docs/FILE_REFERENCE.md`](docs/FILE_REFERENCE.md) | **文件逐个说明**：修改过的上游文件改了什么/为什么，新增的每个文件是干什么的 |
| [`docs/PARAMETERS.md`](docs/PARAMETERS.md) | **参数说明**：44 个命令行参数的默认值/取值/作用，以及实验里真正用过的 arm 配方 |
| [`docs/EXPERIMENT_LOG_20260909_smooth_state.md`](docs/EXPERIMENT_LOG_20260909_smooth_state.md) | 光滑激活（Sigmoid/Tanh）上的 state 准入与三段状态码，两轮实验 |
| [`docs/EXPERIMENT_LOG_20260827_28.md`](docs/EXPERIMENT_LOG_20260827_28.md) | 五个 benchmark 的调度线实验，18 节 |
| [`docs/CE_DIVERSITY_METRIC.md`](docs/CE_DIVERSITY_METRIC.md) | (D, S) 两个多样性指标的推导 |

---

## 准备 benchmark

`data/` 不在版本控制里。ERAN Sigmoid/Tanh 这套（光滑激活实验用的）可以本地重建：

```bash
# 需要先从 HuggingFace 数据集 zhouxingshi/GenBaB 取 eran/* 的 6 个 ONNX，
# 放到 data/vnnlib/eran_sigmoid_tanh_mlp/onnx/，然后：
python data/vnnlib/eran_sigmoid_tanh_mlp/generate_specs.py      # 写 vnnlib/ + instances.csv
python data/vnnlib/eran_sigmoid_tanh_mlp/fold_normalization.py  # 写 onnx_folded/
mv data/vnnlib/eran_sigmoid_tanh_mlp/instances.csv data/vnnlib/eran_sigmoid_tanh_mlp/instances_original.csv
mv data/vnnlib/eran_sigmoid_tanh_mlp/instances_folded.csv data/vnnlib/eran_sigmoid_tanh_mlp/instances.csv
```

细节见 `data/vnnlib/eran_sigmoid_tanh_mlp/README.md` —— 特别是**为什么必须用
`onnx_folded/`**：原图开头的 `Sub(0.1307) → Div(0.3081)` 是标量对 784 的广播，
`torch2act` 会报 `NotImplementedError: Var-var 'sub' size mismatch (784 vs 1)`。

VNN-COMP 的其它 category 用 `tools/batch_download_vnnlib_categories.py` 下载。

---

## 一个最小的完整流程

```bash
# 1) 跑一个 arm，5 次重复，60 秒一组
python -m act.pipeline.Shiyang.pipeline.paper_cifar100_batch_ani \
    --category eran_sigmoid_tanh_mlp --instance-indices 0-98 \
    --admission-mode state --hpgd-weight 0.5 \
    --timeout 60 --repeat 5 --device cpu --no-save \
    --output results/_smooth_ab/statehpgd

# 2) 汇总 distinct / violations
python -m act.pipeline.Shiyang.pipeline.summarize_state_ab _smooth_ab

# 3) 反例多样性：dump 那一轮要写到 ce_diversity 约定的目录下
#    （eransig -> results/div_eran_sig6x100/<arm>/，见 ce_diversity.py 的 BENCH 表）
python -m act.pipeline.Shiyang.pipeline.paper_cifar100_batch_ani \
    --category eran_sigmoid_tanh_mlp --instance-indices 0-98 \
    --admission-mode state --hpgd-weight 0.5 --dump-founder-clusters 250 \
    --timeout 60 --device cpu --no-save \
    --output results/div_eran_sig6x100/statehpgd
python -m act.pipeline.Shiyang.pipeline.ce_diversity eransig --arms portfolio
```

arm 的完整配方（baseline / statebase / statehpgd / always / state_always /
2-bin vs 3-bin）见 [`docs/PARAMETERS.md` 第 1 节](docs/PARAMETERS.md)。

---

## 脚本索引

`pipeline/` 下每个脚本的完整说明在它自己的模块 docstring 里（通常 20–60 行，含实测数字）。
这里只给一句话和它回答的问题。

### 主 driver

| 脚本 | 是什么 |
|---|---|
| `paper_cifar100_batch_ani.py` | **所有 campaign 的唯一入口**。一个 arm = 一组它的命令行参数。原本是论文 Cifar100 Batch-Ani 实验的复现，现在承载全部消融。 |
| `common.py` | 共享工具（初始种子构造等），被下面多数脚本 import。 |

### 汇总与分析

| 脚本 | 回答的问题 |
|---|---|
| `summarize_state_ab.py` | 一次 A/B 的 distinct 和 violations，逐 arm 的均值 ± sd。 |
| `ce_diversity.py` | **(D, S) 多样性分析**。D = 攻破的不同实例数（inter-instance），S = 反例约束方向的平均两两余弦距离（intra-instance）。读 `--dump-founder-clusters` 写出的 `ce_sample.npz`。 |
| `distinct_ceiling.py` | 一组实例里到底有多少个是**可证伪的**？distinct=1 在天花板是 1 还是 15 的情况下含义完全不同。每条 lane 攻击自己的 spec，无语料库无调度，给出公平的上界。 |

### 状态码 / 光滑激活

| 脚本 | 回答的问题 |
|---|---|
| `smooth_state_probe.py` | 在 Sigmoid/Tanh 上「状态」还能指什么？逐层报告哪些候选定义在验证盒里真的会**动**。 |
| `bin_transfer_probe.py` | HPGD 能不能把**指定的**神经元推进**指定的**段？逐 (源段→目标段) 报告命中率。 |
| `two_segment_reachability.py` | 跨两个段的转移在**真实观察到的**状态里存在吗？（`bin_transfer_probe` 的目标是伪造的，0% 不可归因 —— 这个脚本补上真实目标的对照。） |
| `unstable_mask_scope.py` | 不稳定神经元掩码是单条 lane 的正确候选集吗？（默认 `row0` 只来自第一个实例的盒子，实测与一条 lane 自己的集合只重叠 9–12%。） |

### HPGD

| 脚本 | 回答的问题 |
|---|---|
| `hpgd_real_target_hitrate.py` | HPGD 该瞄哪里 —— 以及「落点」本身是不是目标？配平位移下，真实可达目标命中 ~100%，伪造目标 ~6–12%。 |
| `hpgd_ce_guided_flips.py` | 按「反例判别性」挑神经元翻转，比随机翻转强吗？ |
| `hpgd_then_pgd.py` | HPGD 把搜索挪进反例区域后，PGD 的起点是不是更好？（在流水线里 HPGD 不产生反例，它**重定位**搜索 —— 单独给 HPGD 记分是在考它不做的事。） |

---

## 两条方法学纪律

这两条是被实验反复咬过之后写下来的，脚本的 docstring 里也反复引用：

1. **配平位移再比较。** 比较两个机制的结果率之前，先把位移/努力量级配平，否则比较没有意义。
   HPGD 的固定 `flip_count=10` 就栽在这里（见 `--hpgd-flip-frac`）。
2. **目标必须是真实可达的。** 问「X 能不能到达状态 B」时，B 必须是真实观察到/可达的状态，
   不能是凭空构造的 —— 否则失败只说明 B 不可行，什么也没证明。
