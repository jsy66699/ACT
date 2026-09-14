# 参数说明

`paper_cifar100_batch_ani.py` 是所有 campaign 的唯一 driver，一个 arm = 一组它的命令行参数。
本文件分三部分：**arm 配方**（实验里真正用过的组合）、**参数总表**（从 argparse 自动生成）、
**仅代码可改的字段**。

> 表格由 `python act/pipeline/Shiyang/docs/gen_parameters_doc.py` 从
> `paper_cifar100_batch_ani.py` 的 argparse 和 `FuzzingConfig` 重新生成，默认值和取值范围
> 不会与代码脱节。散文部分不会被覆盖。每个参数的完整解释在代码的 `help=` 里，本表只截前两句。

---

## 1. arm 配方

下面每一行都是实验日志里真实跑过的 arm。除了列出的参数，其余全部保持默认。

### 光滑激活两轮实验（`EXPERIMENT_LOG_20260909_smooth_state.md`）

benchmark `eran_sigmoid_tanh_mlp`，60 s/组，n=5。sigmoid 6×100 是 `--instance-indices 0-98`，
tanh 6×100 是 `--instance-indices 297-393`。

| arm | 参数 | 问的是什么 |
|---|---|---|
| `baseline` | `--admission-mode coverage` | 论文原版（GlobalCov 准入 + 其能量公式） |
| `statebase` | `--admission-mode state` | 换成状态新颖性准入 + 密度感知能量 |
| `statehpgd` | `--admission-mode state --hpgd-weight 0.5` | 再加上 HPGD 变异策略 |
| `always` | `--admission-mode always` | **隔离臂**：闸门全开，保留 coverage 的能量公式 |
| `state_always` | `--admission-mode state_always` | **隔离臂**：闸门全开，保留 state 的能量公式 |
| `*_b2` / `*_b3` | 上面各臂再加 `--state-bins 2` / `--state-bins 3` | 状态码从 `sign(z)` 换成 `±τ` 两堵墙 |

`always` / `state_always` 这一对是关键：它们把「准入闸门」和「能量公式」拆开。结论是
`--admission-mode state` 的收益全部来自它顺带换掉的能量公式，不是来自闸门。**不要把
`state` 的效果写成「状态新颖性准入提升了 distinct」。**

### 反例多样性 (D, S) dump

在上面任一 arm 后追加 `--dump-founder-clusters 250`，每组会多写一个 `ce_sample.npz`；
再用 `ce_diversity.py` 读它。

### 调度线（`EXPERIMENT_LOG_20260827_28.md`）

| arm | 参数 |
|---|---|
| `ce1_cerepl` | `--ce-energy 1 --ce-parent-replacement` |
| `ce1_cerepl_state` | 上面再加 `--admission-mode state` |
| `hpgd_ce1_cerepl` | 上面再加 `--hpgd-weight 0.5` |

⚠️ 这条线上的 `state` / `hpgd` 臂是**叠在** `ce1_cerepl` 上的，不能读成「state 准入值多少钱」。
要隔离准入和变异组合，用上面那三个 `portfolio` 臂。

### 几条踩过的坑

* `--hpgd-weight` 需要 `--admission-mode state`：它的候选翻转集就是准入所打分的那个不稳定子空间。
* `--hpgd-flip-frac` 而不是固定的 `hpgd_flip_count`：10 次翻转在 safenlp 的 93 个不稳定神经元里
  是 10.8%，在 cifar100 ResNet 的 3225 个里是 0.31% —— 固定计数的跨 benchmark 比较不是配平的比较。
* `--ce-parent-energy-threshold` 必须 ≤ `--ce-energy`，否则 `--ce-parent-replacement` 悄悄不生效。
* `--unstable-mask row0`（默认）取的是**第一个实例**的盒子并套用到全部 B 条 lane；实测一条 lane
  自己的不稳定集与它只重叠 9–12%。要逐实例的掩码用 `--unstable-mask per_instance`。
* `--ce-prior` 是 **oracle**：分数来自上一轮已经找到的反例。它衡量「这个信息值不值钱」，
  不是一个能在线运行的技术。

---

## 2. 参数总表

<!-- BEGIN CLI -->

### 选哪些实例、跑多久

| 参数 | 默认 | 取值 | 作用 |
|---|---|---|---|
| `--category` | `cifar100_2024` | — | — |
| `--max-instances` | `200` | — | Total VNNLIB instances to load across the whole category (all model groups combined; cifar100_2024 has 200 on disk). Ignored when --instance-indices is given. |
| `--instance-indices` | `None` | — | Restrict to specific 0-based rows of the category's instances.csv instead of a 'first --max-instances' prefix -- e.g. '100-199' or '100,101,105' to load only one model group (cifar100_2024: rows 0-99=resnet_medium, 100-199=resnet_large) for faster repeated-trial verification runs. |
| `--timeout` | `60.0` | — | Per-model-group wall-clock budget in seconds (paper: t_max=60s). |
| `--max-iterations` | `10_000_000` | — | Iteration cap per group (effectively unbounded; --timeout governs stopping). |
| `--repeat` | `1` | — | Run this many independent trials in ONE process, writing each to <output>/rep<N>/. The VNNLIB load and model synthesis are paid once instead of per trial -- on cifar100_2024 that setup is 109s against 120s of fuzzing, so per-trial re-invocation spends nearly half a campaign re-reading the same ONNX files. |
| `--device` | `cuda` | `cpu`, `cuda`, `gpu` | — |
| `--dtype` | `float32` | `float32`, `float64` | — |
| `--batch-conversion` | （开关） | — | Pin each ONNX's SYMBOLIC batch dimension to the number of instances sharing it, instead of to 1. Only graphs whose shapes depend on the batch need it -- attention graphs, which concat a CLS token and Reshape by a computed shape, otherwise convert to a module usable at B=1 alone and the whole group cannot be batched (vit_2023 fails with 'size of tensor a (401) must match tensor b (5)'). |

### 变异策略组合（谁来产生下一个样本）

| 参数 | 默认 | 取值 | 作用 |
|---|---|---|---|
| `--perturb-scale` | `0.1` | — | Anisotropic scale factor s (Eq. 13). |
| `--pgd-only` | （开关） | — | Portfolio becomes {pgd 1.0}: boundary, random and any guided strategy are dropped. The ablation that asks whether the portfolio contributes anything at all, given that every guided strategy measured here has been at or below the arm without it. |
| `--hpgd-weight` | `0.0` | — | Not part of the paper baseline (default 0 = off). >0 adds the pattern-space 'hpgd' mutation strategy, taking the SAME displaced share --hpgd-cov-weight would (33.3% at 0.5) so the two are matched. |
| `--hpgd-cov-weight` | `0.0` | — | Not part of the paper baseline (default 0 = off). >0 adds the coverage-targeted 'hpgd_cov' mutation strategy to the portfolio -- each sample chases its own randomly-drawn never-activated neuron. |
| `--hpgd-absorb-non-pgd` | （开关） | — | Portfolio becomes exactly {pgd 50%, hpgd 50%}: boundary and random are dropped and their share goes to hpgd, while pgd keeps its baseline share. Requires --hpgd-weight > 0. |

### HPGD 的瞄准方式

| 参数 | 默认 | 取值 | 作用 |
|---|---|---|---|
| `--hpgd-target-mode` | `random_flip` | `random_flip`, `interp_real` | Only meaningful with --hpgd-weight > 0. 'random_flip' (default, unchanged) flips k bits of the seed's own pattern independently, which names a sign assignment nothing guarantees is satisfiable: measured on cifar100_2024, ~15% of the asked flips land, 0% exactly, and the projection ends FARTHER from its own target than it started. |
| `--hpgd-flip-frac` | `None` | — | Size HPGD's flip budget as a fraction of the unstable candidate set instead of the fixed hpgd_flip_count. The fixed count is not comparable across networks -- 10 flips is 10.8% of safenlp's 93 unstable neurons but 0.31% of a cifar100 ResNet's 3225 -- so matching this fraction is what makes a cross-benchmark HPGD comparison a matched one. |
| `--hpgd-schedule` | `off` | `off`, `coarse_to_fine` | Only meaningful with --hpgd-weight > 0. Gives each INSTANCE a two-phase curriculum instead of one fixed targeting rule: COARSE large random flips while that instance's state frontier is still growing (imprecise, but the displacement is what pushes the boundary out and stocks the registry with distant real states), then FINE interpolated targets once it stops growing (reached exactly ~94-99%), with rho annealed down so precision rises. |
| `--hpgd-expand-frac` | `0.05` | — | Coarse-phase flip budget as a share of the candidate set (default 5%: ~48 of medium's 951, ~161 of large's 3225, against hpgd_flip_count=10 in the fine phase). |
| `--hpgd-expand-patience` | `20` | — | Admissions an instance may make without growing its frontier radius before its lanes switch to the fine phase. |
| `--hpgd-cov-nearest-margin` | （开关） | — | Only meaningful when --hpgd-cov-weight > 0. Picks the SAME hpgd_cov_target_count targets/sample, but chooses the ones closest to activation_threshold from below (easiest to fire) instead of drawing them uniformly from the whole uncovered pool. |

### 状态码：一个神经元如何变成一个坐标

| 参数 | 默认 | 取值 | 作用 |
|---|---|---|---|
| `--admission-mode` | `coverage` | `coverage`, `state`, `always`, `state_always` | Not part of the paper baseline (default 'coverage' = original GlobalCov-driven admission/energy). 'state' switches ALL strategies' admission+energy to PatternStateManager's BK-tree pattern-novelty check instead (act/pipeline/fuzzing/state_manager.py). |
| `--state-bins` | `2` | `2`, `3` | How a pre-activation becomes a state coordinate. 2 (default) is sign(z) -- the ReLU partition, one bit per neuron. |
| `--state-bin-tau` | `1.0` | — | Wall position for --state-bins 3. Fixed, not per-neuron: on layers whose \|z\| runs to 20 a fixed tau leaves most neurons with all three segments collapsed into one, which is a real limitation of this first version. |
| `--state-dims` | `0.0` | — | Thin each instance's state vector to this many dimensions (0 = all; <1 = a fraction of that instance's own unstable set). Needs --unstable-mask per_instance. |
| `--state-dim-select` | `random` | `random`, `flip_freq`, `flip_rare`, `margin_grad`, `ce_prior` | Which dimensions to keep. 'random' is the control that isolates the dimension COUNT from the choice of dimensions. |
| `--unstable-mask` | `row0` | `row0`, `union`, `per_instance` | Which spec rows the unstable-neuron mask comes from. 'row0' (default, and what every earlier arm ran) derives it from the FIRST instance's box alone and applies it to all B lanes -- measured on cifar100_2024, a lane's own unstable set overlaps it by only 9-12%, and just 6.6-7.7% of the sign flips that actually occur land inside it. |
| `--unstable-mask-source` | `ibp` | `ibp`, `gradient` | How the candidate set is built. 'ibp' (default) propagates intervals through the back_end. |
| `--unstable-mask-grad-threshold` | `1.0` | — | With --unstable-mask-source gradient: keep a coordinate when \|z0 - wall\| / budget < this. 1.0 is 'the box can just reach it'; raising it admits near-misses (vit_2023: 74 at 1.0, 145 at 2.0, 315 at 5.0). |

### 语料库调度：下一个种子从哪来

| 参数 | 默认 | 取值 | 作用 |
|---|---|---|---|
| `--scheduling-mode` | `energy` | `energy`, `sparse` | Not part of the paper baseline. 'sparse' draws seeds from PatternStateManager's BK-tree registry instead of SeedCorpus's energy-weighted select(). |
| `--ce-energy` | `100.0` | — | Energy a counterexample seed carries in the corpus, against an admitted seed's 10. The shipped 100 makes the CE tier own the draw (measured on safenlp: 100% of the live corpus and of the sampling mass). |
| `--energy-tiers` | `None` | — | Four explicit seed energies 'plain,admitted,ce,ce_and_admitted', e.g. '1,5,10,15'. |
| `--ce-parent-replacement` | （开关） | — | Not part of the paper baseline (default off). Stops the counterexample tier from monopolising seed selection: a CE child whose parent is already in that tier takes the parent's corpus slot instead of being appended beside it (one for one per parent). |
| `--ce-parent-energy-threshold` | `100.0` | — | Only meaningful with --ce-parent-replacement. Energy at or above which a parent counts as already in the CE tier. |
| `--select-per-instance` | （开关） | — | Not part of the paper baseline (default off). Round-robin over INSTANCES, energy-weighted only within each, so every instance contributes a lane before any instance contributes a second. |
| `--select-without-replacement` | （开关） | — | Not part of the paper baseline (default off = the paper's draw). The paper draws the B lanes WITH replacement (Algorithm 3 line 10, 'high-energy seeds may repeat'), which is standard exploitation when the corpus serves one program under test. |
| `--coverage-per-instance` | （开关） | — | Give each verification instance its own GlobalCov 'already covered' set instead of one union over the batch. Under the union (default, and what every earlier arm ran) a sample counts as interesting only if it fires a neuron NO instance has ever fired, so instance A raises the bar for the other 99 and the bar keeps rising -- a harsher admission rule purely for being run in a bigger batch. |

### 输出与探针

| 参数 | 默认 | 取值 | 作用 |
|---|---|---|---|
| `--output` | `str(DEFAULT_OUTPUT_DIR.relative_to(SHIYANG_ROOT` | — | Output dir, relative to act/pipeline/Shiyang/ unless absolute. |
| `--no-save` | （开关） | — | — |
| `--report-interval` | `2000` | — | — |
| `--verbose` | `1` | — | — |
| `--dump-founder-clusters` | `0` | — | Per group, write founder_clusters.npz: the input tensor of each of up to N founders for the instance with the most of them, plus one sibling per founder. Off (0) by default. |
| `--dump-ce-prior` | `None` | — | After each group, append per-instance counterexample suspiciousness (\|P(sign=+\|CE) - P(sign=+\|rest)\|) to this .pt file. Merges across runs and groups, so several harvest runs sharpen the same prior. |
| `--dump-ce-patterns` | `None` | — | Dump the raw per-instance sign patterns (counterexample and not) to this .pt, for analysing the STRUCTURE of an instance's counterexamples -- how many distinct linear regions they occupy, how far apart they are, which neurons are constant across them. The prior dump only keeps per-neuron marginals, which cannot answer any of that. |
| `--ce-prior` | `None` | — | Read state-dimension scores from a file written by --dump-ce-prior. Only used with --state-dim-select ce_prior. |
<!-- END CLI -->

---

## 3. 仅代码可改的字段

`FuzzingConfig`（`act/pipeline/fuzzing/actfuzzer.py`）里没有对应命令行开关的字段。改它们要么编辑
dataclass 默认值，要么在 driver 里构造 config 时传入。

<!-- BEGIN CONFIG_ONLY -->
| 字段 | 默认 | 类型 |
|---|---|---|
| `stop_on_first_violation` | `False` | `bool` |
| `enable_bi_gce` | `False` | `bool` |
| `bi_batch_size` | `0   # 0 = use the normal (model-synthesis) batch size` | `int` |
| `gce_batch_size` | `0  # 0 = same as bi_batch_size` | `int` |
| `state_diversity_threshold` | `1` | `int` |
| `state_bloom_bits` | `1 << 20` | `int` |
| `state_bloom_hashes` | `4` | `int` |
| `state_local_bias_high` | `10.0` | `float` |
| `state_local_bias_low` | `0.1` | `float` |
| `ce_prior_path` | `""` | `str` |
| `ce_prior_key` | `""` | `str` |
| `dump_ce_prior_path` | `""` | `str` |
| `hpgd_flip_count` | `10` | `int` |
| `hpgd_num_steps` | `10` | `int` |
| `hpgd_margin` | `0.01` | `float` |
| `hpgd_sequential_flip` | `False` | `bool` |
| `hpgd_loss_scope` | `"target_only"` | `str` |
| `hpgd_momentum` | `0.0` | `float` |
| `hpgd_step_decay` | `False` | `bool` |
| `hpgd_hold_still_weight` | `1.0` | `float` |
| `hpgd_normalize_by_scale` | `False` | `bool` |
| `hpgd_sparse_targets` | `True` | `bool` |
| `hpgd_interp_rhos` | `"0.25,0.5,0.75"` | `str` |
| `hpgd_interp_proposals` | `2` | `int` |
| `hpgd_interp_pool` | `32` | `int` |
| `hpgd_refine_rho_start` | `0.75` | `float` |
| `hpgd_refine_rho_end` | `0.25` | `float` |
| `hpgd_interp_min_d` | `4` | `int` |
| `hpgd_cov_target_count` | `3` | `int` |
| `hpgd_cov_num_steps` | `10` | `int` |
| `hpgd_cov_momentum` | `0.0` | `float` |
| `hpgd_cov_step_decay` | `False` | `bool` |
| `gce_num_steps` | `10` | `int` |
| `gce_margin` | `0.01` | `float` |
| `gce_hamming_radius` | `3` | `int` |
| `gce_noise_scale` | `0.05` | `float` |
| `gce_step_decay` | `False` | `bool` |
| `bi_attack_strategy` | `None` | `Optional[str]` |
| `bi_attack_pgd_steps` | `50` | `int` |
| `bi_attack_apgd_t_target_classes` | `5` | `int` |
| `bi_random_restart_prob` | `0.0` | `float` |
| `bi_random_restart_cooling_rate` | `0.98` | `float` |
| `bi_threaded` | `False` | `bool` |
| `bi_queue_size` | `64` | `int` |
| `pgd_restarts` | `1` | `int` |
| `pgd_restarts_binarized` | `40` | `int` |
<!-- END CONFIG_ONLY -->
