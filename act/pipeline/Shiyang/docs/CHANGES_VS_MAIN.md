# 与 `main` 的差异说明

这个分支相对上游 `main` 改了什么、为什么改。表格由
`python act/pipeline/Shiyang/docs/gen_changes_doc.py` 从**真实的 diff** 生成，
改动了却没写理由的文件会让脚本报错退出，所以这张表不会漏。

> 表里 `CHANGES_VS_MAIN.md` 自己那一行的行数会比实际少几行 —— 它统计的是**生成之前**的自己，这是自指文档的固定点问题，不影响其它任何数字。

> 这次实验（状态码 / state 准入 / HPGD / 光滑激活）的实现细节 —— 哪个类、谁调用它、怎么完成 —— 看 [`IMPLEMENTATION_state_hpgd_smooth.md`](IMPLEMENTATION_state_hpgd_smooth.md)。

```bash
# 重新生成（默认对比 origin/main）
python act/pipeline/Shiyang/docs/gen_changes_doc.py
python act/pipeline/Shiyang/docs/gen_changes_doc.py --base upstream/main --head slim-experiments
```

<!-- BEGIN SUMMARY -->
| | |
|---|---|
| 对比基准 | `origin/main` at `e9ed992` |
| 相差 commit | 23 |
| 改动文件 | 61（新增 48 · 修改 12 · 删除 1） |
| 行数 | +15,457 / −216 |
| 改动的**上游**文件 | 12，其中在默认参数下就生效的只有 1 个（`torch2act.py`，见下） |
<!-- END SUMMARY -->

---

## 一句话结论

> **默认配置下，这个分支的 fuzzer 行为和 `main` 一致。** 所有新能力（state 准入、HPGD、
> 稀疏调度、BI/GCE、能量分层、CE-parent 替换、3 段状态码）默认全部关闭，
> `admission_mode: coverage` + `scheduling_mode: energy` 就是原版。

「上游行为修正」那组有 5 个文件，但**在默认参数下真正生效的只有一个**：

1. **`torch2act.py` —— 唯一一个默认就生效的改动，而且是个真 bug 修复。** flatten 层在 B>1 时
   `out_vars` 带 batch 维、`output_shape` 不带，区间传播两边都查，没有任何 batch 能同时满足。
   于是每个纯 MLP benchmark（mnist_fc、acasxu、tllverifybench、cora…）在 B>1 时**静默丢失
   unstable mask**，退化成「所有神经元都是候选」—— 日志里看不出任何异常。不修的话，所有基于
   「不稳定神经元」的东西（状态码、HPGD 候选集、准入打分）在 MLP 上都是在对全部神经元工作。
   **B=1 的结果逐字节不变**，而验证器自己建的图全都是 B=1。
2. 其余 4 个（`onnx_converter` / `spec_creator_base` / `create_specs` / `data_model_loader`）
   加的都是**默认关闭**的参数：`batch_size=1`、`model_batch_size=1`、`batch_conversion=False`、
   `instance_indices=None`。不传就走原来的路径。

所以：**想确认「我的改动没有动上游」，只需要审 `torch2act.py` 那 32 行**，其余上游文件的
改动都在默认关闭的开关后面，实验代码和文档则是纯新增。

怎么自己验证：

```bash
# 只看会改变上游默认行为的那几个文件
git diff origin/main -- act/pipeline/verification/torch2act.py \
    act/front_end/vnnlib_loader/onnx_converter.py \
    act/front_end/spec_creator_base.py \
    act/front_end/vnnlib_loader/data_model_loader.py

# 确认引擎的新字段都有默认值，且默认值 = 原行为
git diff origin/main -- act/pipeline/fuzzing/actfuzzer.py | grep -E "^\+ +[a-z_]+: .+ = "
```

---

## 按文件

<!-- BEGIN TABLE -->

### 上游行为修正　　<sub>5 个文件 · +157 / −30</sub>

| 文件 | | 行数 | 为什么 |
|---|---|---|---|
| `act/front_end/vnnlib_loader/create_specs.py` | 修改 | +69 / −10 | 两个新参数，都默认关：`instance_indices` 让调用方直接点名要 instances.csv 的哪几行，而不必为了拿到第 199 行先把前 199 行全部 ONNX 转换一遍；`batch_conversion` 把上一条的 batch 钉法接出来。默认路径与原来完全相同。 |
| `act/pipeline/verification/torch2act.py` | 修改 | +32 / −3 | **真 bug 修复**：flatten 层在 B>1 时 out_vars 带 batch 维而 output_shape 不带，两边都被区间传播检查，没有任何 batch 能同时满足 —— 于是每个纯 MLP benchmark 在 B>1 时**静默丢失 unstable mask**，退化成「所有神经元都是候选」。新增 `_per_sample_forward`；B=1 的结果逐字节不变。 |
| `act/front_end/spec_creator_base.py` | 修改 | +24 / −8 | 配合上一条：校验模型 I/O 形状时把 forward 加宽到 `model_batch_size`，但**报告回来的形状保持 per-sample** —— 否则 per-instance 的 spec 会因为「只有一行而不是 B 行」被拒。 |
| `act/front_end/vnnlib_loader/onnx_converter.py` | 修改 | +23 / −6 | `convert_onnx_to_pytorch(batch_size=)`：把 ONNX 的**符号 batch 维**钉到调用方真正要用的 lane 数。注意力图（concat CLS + 按计算出的 shape Reshape）被 onnxsim 按这个值常量折叠，钉成 1 就只能在 B=1 跑，vit_2023 在 B=4 报 `size of tensor a (401) must match tensor b (5)`。默认仍是 1，形状不依赖 batch 的图完全不受影响。 |
| `act/front_end/vnnlib_loader/data_model_loader.py` | 修改 | +9 / −3 | 把 `model_batch_size` 从加载入口透传到转换器，并加了 ONNX 转换缓存。 |

### fuzzing 引擎（改上游文件）　　<sub>4 个文件 · +2,554 / −59</sub>

| 文件 | | 行数 | 为什么 |
|---|---|---|---|
| `act/pipeline/fuzzing/actfuzzer.py` | 修改 | +1,118 / −14 | `FuzzingConfig` 从 16 个字段扩到 60+，覆盖 state 准入、HPGD、调度与能量的全部开关，**默认值全部等于原版行为**。新增 `_init_state_manager` / `_observe_state` / `_gce_iteration` / `_propose_hpgd_targets`，以及反例先验和反例模式的 dump。 |
| `act/pipeline/fuzzing/mutations.py` | 修改 | +869 / −5 | 三个新变异策略 `HPGDMutation` / `HPGDCoverageMutation` / `HPGDPullbackMutation`，以及批量激活模式提取 `_activation_sign_pattern_batched` —— 后者是让状态码能跑在 Sigmoid/Tanh 上的前提。 |
| `act/pipeline/fuzzing/corpus.py` | 修改 | +462 / −28 | `SeedCorpus` 加上诊断「能量垄断」所需的全部仪表：逐实例抽取计数 `draws_by_instance`、丢弃统计 `drop_stats`、CE 血统/founder 追踪、能量分层，以及 `select()` 的无放回与逐实例轮询两种取样方式和 CE-parent 替换。 |
| `act/pipeline/fuzzing/coverage.py` | 修改 | +105 / −12 | `has_observations()` 区分「还没建 mask」和「已全覆盖」（原来 `get_uncovered_neurons()` 两种情况返回一样）；`GlobalCov(per_instance=)` 让每个实例有自己的覆盖行，而不是全 batch 共用一个单调并集 —— 并集下 A 实例点亮一个神经元会把门槛抬给其余 99 个。 |

### fuzzing 引擎（新文件）　　<sub>9 个文件 · +3,620 / −0</sub>

| 文件 | | 行数 | 为什么 |
|---|---|---|---|
| `act/pipeline/fuzzing/state_manager.py` | 新增 | +1,266 / −0 | `PatternStateManager`：Bloom filter + BK-tree 的全局状态登记表。准入、能量、稀疏调度都从它读，是 `--admission-mode state` 背后的东西。 |
| `act/pipeline/fuzzing/pattern_search_pgd.py` | 新增 | +755 / −0 | 模式空间两阶段 PGD 的独立 runner。 |
| `act/pipeline/fuzzing/auto_attack_pgd.py` | 新增 | +389 / −0 | 独立攻击 runner：随机重启 PGD / AutoAttack。 |
| `act/pipeline/fuzzing/random_start_pgd.py` | 新增 | +324 / −0 | 独立攻击 runner：随机重启 PGD / AutoAttack。 |
| `act/pipeline/fuzzing/auto_pgd_batched.py` | 新增 | +237 / −0 | AutoPGD 的批量实现。 |
| `act/pipeline/fuzzing/bi_gce_fuzz.py` | 新增 | +205 / −0 | BI（稀疏性引导的探索）/ GCE（在已知反例附近做锚点拉回）双任务循环，共用一个 PatternStateManager。 |
| `act/pipeline/fuzzing/state_bins.py` | 新增 | +202 / −0 | 状态码的编码：2 段（`sign(z)`，ReLU 划分）或 3 段（`±τ` 两堵墙），3 段时每个神经元两个 ±1 坐标，所以 BK-tree / Bloom / 指纹打包都不用改。 |
| `act/pipeline/fuzzing/bi_threads.py` | 新增 | +136 / −0 | BI/GCE 的生产者-消费者线程，每个线程一份模型副本（共享模型会竞争）。 |
| `act/pipeline/fuzzing/test_observe_batch.py` | 新增 | +106 / −0 | 批量状态观测的测试。 |

### 配置　　<sub>5 个文件 · +299 / −0</sub>

| 文件 | | 行数 | 为什么 |
|---|---|---|---|
| `act/config/pattern_search_pgd.yaml` | 新增 | +104 / −0 | 对应独立 runner 的配置文件。 |
| `act/config/random_start_pgd.yaml` | 新增 | +82 / −0 | 对应独立 runner 的配置文件。 |
| `act/config/auto_attack_pgd.yaml` | 新增 | +39 / −0 | 对应独立 runner 的配置文件。 |
| `act/config/bi_gce_fuzz.yaml` | 新增 | +38 / −0 | 对应独立 runner 的配置文件。 |
| `act/config/pipeline.yaml` | 修改 | +36 / −0 | 上面那些开关的默认值和注释。**默认全关**：`admission_mode: coverage` + `scheduling_mode: energy` 就是原版 fuzzer。 |

### 实验代码（纯新增）　　<sub>17 个文件 · +4,254 / −0</sub>

| 文件 | | 行数 | 为什么 |
|---|---|---|---|
| `act/pipeline/Shiyang/pipeline/paper_cifar100_batch_ani.py` | 新增 | +993 / −0 | **所有 campaign 的唯一 driver**。一个 arm = 一组它的命令行参数。 |
| `act/pipeline/Shiyang/pipeline/ce_diversity.py` | 新增 | +801 / −0 | (D, S) 反例多样性分析。 |
| `act/pipeline/Shiyang/pipeline/smooth_state_probe.py` | 新增 | +595 / −0 | 分析/探针脚本，每个的完整说明在它自己的模块 docstring 里。 |
| `act/pipeline/Shiyang/pipeline/hpgd_real_target_hitrate.py` | 新增 | +416 / −0 | 分析/探针脚本，每个的完整说明在它自己的模块 docstring 里。 |
| `act/pipeline/Shiyang/pipeline/hpgd_ce_guided_flips.py` | 新增 | +320 / −0 | 分析/探针脚本，每个的完整说明在它自己的模块 docstring 里。 |
| `act/pipeline/Shiyang/pipeline/hpgd_then_pgd.py` | 新增 | +207 / −0 | 分析/探针脚本，每个的完整说明在它自己的模块 docstring 里。 |
| `act/pipeline/Shiyang/pipeline/unstable_mask_scope.py` | 新增 | +175 / −0 | 分析/探针脚本，每个的完整说明在它自己的模块 docstring 里。 |
| `act/pipeline/Shiyang/pipeline/bin_transfer_probe.py` | 新增 | +151 / −0 | 分析/探针脚本，每个的完整说明在它自己的模块 docstring 里。 |
| `act/pipeline/Shiyang/pipeline/two_segment_reachability.py` | 新增 | +140 / −0 | 分析/探针脚本，每个的完整说明在它自己的模块 docstring 里。 |
| `act/pipeline/Shiyang/pipeline/distinct_ceiling.py` | 新增 | +97 / −0 | 分析/探针脚本，每个的完整说明在它自己的模块 docstring 里。 |
| `act/pipeline/Shiyang/results/verify_safenlp_10x3.sh` | 新增 | +66 / −0 | 启动 campaign 的 shell 脚本（`results/` 其余内容不入库）。 |
| `act/pipeline/Shiyang/results/verify_safenlp_30x4.sh` | 新增 | +52 / −0 | 启动 campaign 的 shell 脚本（`results/` 其余内容不入库）。 |
| `act/pipeline/Shiyang/results/verify_safenlp_4arm.sh` | 新增 | +51 / −0 | 启动 campaign 的 shell 脚本（`results/` 其余内容不入库）。 |
| `act/pipeline/Shiyang/pipeline/common.py` | 新增 | +49 / −0 | 分析/探针脚本，每个的完整说明在它自己的模块 docstring 里。 |
| `act/pipeline/Shiyang/results/verify_cifar_30x4.sh` | 新增 | +49 / −0 | 启动 campaign 的 shell 脚本（`results/` 其余内容不入库）。 |
| `act/pipeline/Shiyang/results/verify_safenlp_isolate.sh` | 新增 | +48 / −0 | 启动 campaign 的 shell 脚本（`results/` 其余内容不入库）。 |
| `act/pipeline/Shiyang/pipeline/summarize_state_ab.py` | 新增 | +44 / −0 | 分析/探针脚本，每个的完整说明在它自己的模块 docstring 里。 |

### 文档（纯新增）　　<sub>10 个文件 · +3,976 / −0</sub>

| 文件 | | 行数 | 为什么 |
|---|---|---|---|
| `act/pipeline/Shiyang/docs/EXPERIMENT_LOG_20260827_28.md` | 新增 | +1,847 / −0 | 实验入口、实现说明、参数说明、与 main 的差异、实验日志、(D,S) 指标推导。 |
| `act/pipeline/Shiyang/docs/IMPLEMENTATION_state_hpgd_smooth.md` | 新增 | +463 / −0 | 实验入口、实现说明、参数说明、与 main 的差异、实验日志、(D,S) 指标推导。 |
| `act/pipeline/Shiyang/docs/CE_DIVERSITY_METRIC.md` | 新增 | +343 / −0 | 实验入口、实现说明、参数说明、与 main 的差异、实验日志、(D,S) 指标推导。 |
| `act/pipeline/Shiyang/docs/EXPERIMENT_LOG_20260909_smooth_state.md` | 新增 | +242 / −0 | 实验入口、实现说明、参数说明、与 main 的差异、实验日志、(D,S) 指标推导。 |
| `act/pipeline/Shiyang/docs/gen_changes_doc.py` | 新增 | +217 / −0 | 实验入口、实现说明、参数说明、与 main 的差异、实验日志、(D,S) 指标推导。 |
| `act/pipeline/Shiyang/docs/CHANGES_VS_MAIN.md` | 新增 | +209 / −0 | 实验入口、实现说明、参数说明、与 main 的差异、实验日志、(D,S) 指标推导。 |
| `act/pipeline/Shiyang/docs/PARAMETERS.md` | 新增 | +199 / −0 | 实验入口、实现说明、参数说明、与 main 的差异、实验日志、(D,S) 指标推导。 |
| `ACT_PIPELINE_HELP.md` | 新增 | +194 / −0 | `python -m act.pipeline --help` 的 Markdown 版。 |
| `act/pipeline/Shiyang/docs/gen_parameters_doc.py` | 新增 | +150 / −0 | 实验入口、实现说明、参数说明、与 main 的差异、实验日志、(D,S) 指标推导。 |
| `act/pipeline/Shiyang/README.md` | 新增 | +112 / −0 | 实验入口、实现说明、参数说明、与 main 的差异、实验日志、(D,S) 指标推导。 |

### benchmark 工具（纯新增）　　<sub>8 个文件 · +583 / −0</sub>

| 文件 | | 行数 | 为什么 |
|---|---|---|---|
| `data/vnnlib/eran_sigmoid_tanh_mlp/fold_normalization.py` | 新增 | +124 / −0 | ERAN Sigmoid/Tanh benchmark 的生成脚本与说明（ONNX 和 spec 本身不入库）。 |
| `data/vnnlib/eran_sigmoid_tanh_mlp/generate_specs.py` | 新增 | +103 / −0 | ERAN Sigmoid/Tanh benchmark 的生成脚本与说明（ONNX 和 spec 本身不入库）。 |
| `tools/convert_vnnlib_1_to_2.py` | 新增 | +98 / −0 | VNN-COMP category 下载、VNNLIB 1.0/2.0 互转、给 α,β-CROWN 备料。 |
| `tools/batch_download_vnnlib_categories.py` | 新增 | +79 / −0 | VNN-COMP category 下载、VNNLIB 1.0/2.0 互转、给 α,β-CROWN 备料。 |
| `tools/convert_vnnlib_2_to_1.py` | 新增 | +60 / −0 | VNN-COMP category 下载、VNNLIB 1.0/2.0 互转、给 α,β-CROWN 备料。 |
| `tools/stage_vnnlib_1_for_abcrown.py` | 新增 | +56 / −0 | VNN-COMP category 下载、VNNLIB 1.0/2.0 互转、给 α,β-CROWN 备料。 |
| `data/vnnlib/eran_sigmoid_tanh_mlp/README.md` | 新增 | +55 / −0 | ERAN Sigmoid/Tanh benchmark 的生成脚本与说明（ONNX 和 spec 本身不入库）。 |
| `data/vnnlib/eran_sigmoid_tanh_mlp/info.json` | 新增 | +8 / −0 | ERAN Sigmoid/Tanh benchmark 的生成脚本与说明（ONNX 和 spec 本身不入库）。 |

### 杂项　　<sub>3 个文件 · +14 / −127</sub>

| 文件 | | 行数 | 为什么 |
|---|---|---|---|
| `.gitignore` | 修改 | +14 / −0 | 忽略结果数据、嵌套的 results 副本、`.idea/`。 |
| `act/pipeline/__main__.py` | 修改 | +0 / −1 | docstring 里少了一个空行，无行为变化。 |
| `act/pipeline/log/pipeline_tests.log` | 删除 | +0 / −126 | 删掉一个被跟踪的测试日志。 |
<!-- END TABLE -->

---

## commit 列表

<!-- BEGIN COMMITS -->
| commit | 日期 | 标题 |
|---|---|---|
| `a617c5e` | 2026-07-16 | feat(fuzzing): add PatternSearchPGD two-phase pattern-space attack |
| `0b3f3f2` | 2026-07-16 | feat(fuzzing): add round2/round3 to PatternSearchPGD for parity with the original design |
| `8474a0e` | 2026-07-16 | fix(fuzzing): give an accurate error when a downloaded VNNLIB category has no parseable instances |
| `c81a69c` | 2026-07-16 | feat(fuzzing): add HPGD mutation strategy (pattern-space hinge-loss PGD) |
| `dc0a1a1` | 2026-07-16 | feat(fuzzing): global PatternStateManager (Bloom filter + BK-tree) with config-switchable admission/scheduling, plus BI/GCE two-task loop |
| `1a61da6` | 2026-07-19 | Merge remote-tracking branch 'origin/main' |
| `ee6690f` | 2026-07-25 | Merge remote-tracking branch 'origin/main' |
| `9359dd8` | 2026-08-25 | Merge remote-tracking branch 'origin/main' |
| `807f08f` | 2026-08-25 | recover: restore hpgd_cov, instance_indices and the lost FuzzingConfig fields |
| `0a54186` | 2026-08-25 | fix(fuzzing): stop coverage-steered mutation degrading silently |
| `6c5e235` | 2026-08-25 | feat(batch-ani): add --hpgd-weight and pair each guided strategy with its own admission mode |
| `73b068d` | 2026-08-25 | perf(fuzzing): stop paying for a tree walk to answer an exact-match question |
| `e395908` | 2026-08-25 | test(batch-ani): isolation run separating state admission from the hpgd strategy |
| `b5803a4` | 2026-08-25 | perf+feat(fuzzing): fingerprint-based batch admission, per-instance state space |
| `ab3abc7` | 2026-08-25 | test(batch-ani): 4-arm safenlp campaign on one pinned code version |
| `850bcbd` | 2026-08-26 | feat(fuzzing): make HPGD actually steer -- close its feedback loop, score only what it aims at, and target sparse regions |
| `4998904` | 2026-08-26 | test(batch-ani): 30x4 safenlp campaign on 66280e9 |
| `0acf1c2` | 2026-08-26 | feat(batch-ani): --repeat, and the cifar100 counterpart of the safenlp campaign |
| `44f1c7e` | 2026-09-11 | Merge origin/main into shiyang-fuzzing-research |
| `b7f02ae` | 2026-09-14 | chore: slim experiment branch -- code, scripts and docs only |
| `9b6b598` | 2026-09-14 | docs: explain every difference from upstream main, and generate it from the diff |
| `0c2bb82` | 2026-09-14 | docs: file-by-file reference in Chinese for every change against main |
| `652b7db` | 2026-09-14 | docs: replace the broad file reference with a deep dive on this experiment |
<!-- END COMMITS -->

---

## 相关文档

* [`README.md`](../README.md) —— 实验入口：怎么准备 benchmark、最小流程、脚本索引
* [`PARAMETERS.md`](PARAMETERS.md) —— 44 个参数的说明，以及实验里用过的 arm 配方
* [`../../../../PROVENANCE.md`](../../../../PROVENANCE.md) —— 仅 `slim-standalone` 分支有：
  快照来源与删改清单
