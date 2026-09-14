# 文件逐个说明

这份文档把这个分支相对上游 `main` 的**每一个**改动讲清楚：修改过的上游文件改了什么、
为什么改、会不会影响别人；新增的每个文件是干什么的。

统计数字和「一句话为什么」看 [`CHANGES_VS_MAIN.md`](CHANGES_VS_MAIN.md)（那份是脚本生成的）；
这份是展开的散文版。

**读之前先记住一件事**：所有新能力默认都是关的。`admission_mode: coverage` +
`scheduling_mode: energy` 就是原版 fuzzer，逐字节一致。下面「修改的上游文件」里，
**只有 `torch2act.py` 在默认参数下就生效**，而且它修的是上游本来就坏掉的东西。

---

# 第一部分 · 修改过的上游文件（12 个）

## 1.1 会改变默认行为的：只有一个

### `act/pipeline/verification/torch2act.py`　`+32 / −3`

**改了什么**：新增 `_LayerGraphBuilder._per_sample_forward()`，让 flatten 类层按**单个样本**
分配输出变量，而不是沿用 `_same_size_forward` 把 batch 维一起算进去。

**为什么**：这是个真 bug。输入变量是按 `_prod(self.input_shape)` 分配的，**含 batch**；
而下游每个真实层都是按单样本分配（`_convert_conv2d` 拿 `out_c*out_h*out_w`，
`_convert_linear` 拿 `out_features`）。CNN 在第一个卷积就把 batch 甩掉了，从此自洽；
但**纯 MLP 的第一个变形层是 flatten**，它走 `_same_size_forward`，于是 FLATTEN 的
`out_vars` 是 `B*N`，而它记录的 `output_shape` 是单样本的。区间传播两边都检查
（`tf_flatten`，`act/back_end/interval_tf/tf_cnn.py:313` 和 `:323`），**没有任何 batch 能同时满足**：
传 B 行则 out_vars 长了 B 倍，把 batch 折成一行则 output_shape 对不上。

后果是：`mnist_fc`、`acasxu`、`tllverifybench`、`cora` 这类纯 MLP benchmark 在 B>1 时
**静默丢失 unstable mask**，退化成「所有神经元都是候选」，而日志里看不出任何异常。
所有依赖「不稳定神经元」的机制——状态码、HPGD 的翻转候选集、准入打分——在 MLP 上
其实都在对全部神经元工作，测出来的数一个都不能引用。

**对别人的影响**：`B=1` 时 `_per_sample_forward` 返回的东西和 `_same_size_forward` **完全一样**，
而验证器自己建的图全都是 B=1。所以只有批量 fuzzing 路径受影响，而那条路径原本就是坏的。

**怎么验**：`git diff origin/main -- act/pipeline/verification/torch2act.py`，32 行，注释里
写了完整推理。

## 1.2 加了参数但默认关闭的（4 个前端文件）

这四个是一件事的四个环节：**把 ONNX 的符号 batch 维钉到调用方真正要用的 lane 数**。

### `act/front_end/vnnlib_loader/onnx_converter.py`　`+23 / −6`

`convert_onnx_to_pytorch(..., batch_size=1)` 新增参数，`_preprocess_onnx_for_onnx2torch`
把图里第一个 `dim_value=0`（符号 batch）设成它。

**为什么**：这个值是被**烤进图里**的，不只是声明——onnxsim 会拿它做常量折叠。形状依赖
batch 的图（所有注意力图：先 concat 一个 CLS token，再按计算出来的 shape 做 Reshape）
从 onnx2torch 出来之后**只能在那个 batch 跑**。`vit_2023` 钉成 1 之后在 B=4 报
`size of tensor a (401) must match tensor b (5)` —— 401 = 100×4+1，token 轴把 batch 折进去了。

**默认 1**，所以形状不依赖 batch 的图（绝大多数）转换结果和以前一模一样。

### `act/front_end/spec_creator_base.py`　`+24 / −8`

`_get_model_io_shapes(..., model_batch_size=1)`：校验模型 I/O 形状时，如果模型是按 B 钉过的，
就把测试输入 `expand` 到 B 再 forward；但**报告回去的输出形状仍然砍回单样本**。

**为什么**：被校验的 spec 是 per-instance 的，如果按 B 行的形状去比，spec 会因为
「只有一行而不是 B 行」被拒。

### `act/front_end/vnnlib_loader/create_specs.py`　`+69 / −10`

两个新参数：

* `instance_indices`：直接点名要 instances.csv 的**哪几行**，而不是「前 max_instances 行」
  的前缀。想要第 199 行，原来必须把 0..199 行全部加载并 ONNX 转换一遍才能走到；现在
  `instance_indices=[199]` 只转那一行。返回顺序跟随你给的顺序，越界直接 `ValueError`
  （想要旧的「静默跳过」就自己先过滤）。
* `batch_conversion`：把上面那条 batch 钉法接到命令行（driver 的 `--batch-conversion`）。

**默认 `None` / `False`**，不传就是原来的路径。

### `act/front_end/vnnlib_loader/data_model_loader.py`　`+9 / −3`

把 `model_batch_size` 从加载入口透传到 `convert_onnx_to_pytorch`，另外加了一层 ONNX 转换
缓存（同一个 ONNX 被多个 spec 引用时不重复转）。

## 1.3 fuzzing 引擎（改动大，但全在默认关闭的开关后面）

### `act/pipeline/fuzzing/actfuzzer.py`　`+1,118 / −14`

**`FuzzingConfig` 从 16 个字段扩到 60 多个**，覆盖：状态准入（`admission_mode`、
`state_bins`、`state_bin_tau`、`state_dims`、`state_dim_select`）、候选集来源
（`unstable_mask_source`、`unstable_mask_scope`、`unstable_mask_grad_threshold`）、
调度与能量（`scheduling_mode`、`select_with_replacement`、`select_per_instance`、
`ce_energy_bonus`、`energy_tiers`、`ce_parent_replacement`、`coverage_per_instance`）、
HPGD（十几个 `hpgd_*`）、BI/GCE（`enable_bi_gce`、`bi_batch_size`、`gce_batch_size`）。
**每一个的默认值都等于原版行为。**

新增方法：

| 方法 | 干什么 |
|---|---|
| `_init_state_manager()` | 建 `PatternStateManager`，算不稳定掩码 |
| `_observe_state()` | 每个 batch 把状态码喂给登记表，拿回「是否新颖」 |
| `_gce_iteration()` | BI/GCE 双任务循环里 GCE 那一半 |
| `_propose_hpgd_targets()` | 给 HPGD 生成目标状态码（随机翻转 / 真实态插值） |
| `_ce_add_kwargs()` | 反例入库时带上血统信息 |
| `dump_ce_patterns()` / `dump_ce_prior()` | 导出反例的原始符号模式 / 逐神经元先验 |

### `act/pipeline/fuzzing/mutations.py`　`+869 / −5`

三个新变异策略：

* **`HPGDMutation`** —— 模式空间的铰链损失 PGD。给定目标状态码，对每个要翻转的神经元
  施加 `target * z` 的铰链，把预激活推过墙。这是实验里 `--hpgd-weight` 打开的那个。
* **`HPGDCoverageMutation`** —— 同样的机制，但目标是「从没被点亮过的神经元」，
  服务于覆盖率而不是状态多样性。
* **`HPGDPullbackMutation`** —— 朝已确认反例的锚点模式拉回，BI/GCE 里 GCE 那一半用。

以及 `_activation_preactivations_batched` / `_activation_sign_pattern_batched`：用 forward hook
批量取任意激活层的预激活。**这是状态码能跑在 Sigmoid/Tanh 上的前提**——原来的实现只认 ReLU。

### `act/pipeline/fuzzing/corpus.py`　`+462 / −28`

`SeedCorpus` 加上了诊断「能量垄断」所需要的全部仪表，以及两个修它的开关：

| 新增 | 干什么 |
|---|---|
| `draws_by_instance()` | 每个实例被抽中过多少次 —— 发现「一个实例吃掉 95% 抽样质量」靠的就是它 |
| `drop_stats()` | offered / gate / dedup / ce_sibling / added 的分解 |
| `select(replace=, per_instance=)` | 无放回抽取；按实例轮询（每个实例先出一条 lane，再轮第二条） |
| `ce_lineage_by_instance()` / `ce_founder_rows()` | 反例的血统树与 founder，`--dump-founder-clusters` 的数据来源 |
| `count_above()` / `ce_mass()` / `energy_mass_above()` | 能量分层的质量分布 |
| CE-parent 替换 | 反例子代**顶替**父代的语料库槽位，而不是并排追加 |

### `act/pipeline/fuzzing/coverage.py`　`+105 / −12`

* `has_observations()` —— 区分「还没建任何 mask」和「已经全覆盖」。原来
  `get_uncovered_neurons()` 两种情况返回一样，调用方分不出来。
* `GlobalCov(per_instance=N)` —— 每个验证实例一行覆盖记录，而不是全 batch 共用一个
  单调并集。**为什么这是个真选择而不是细节**：并集之下，一个样本要「点亮从没有任何实例
  点亮过的神经元」才算 interesting，于是 A 实例在第 3 次迭代点亮一个神经元，就把门槛
  永久抬给了其余 99 个，而且门槛只升不降 —— 同一个 benchmark 只因为跑在更大的 batch 里
  就得到更严格的准入规则。B=1 时两种模式等价，所以这个问题只在有批处理之后才浮现。
  实测：coverage 臂的语料库停在约 273 个活种子，而状态准入是约 3010 个。

## 1.4 杂项

### `act/config/pipeline.yaml`　`+36 / −0`

上面那些开关的默认值和注释。开头就写明：**默认全关**，`admission_mode: "coverage"` +
`scheduling_mode: "energy"` 是逐字节的原版 fuzzer（CoverageTracker + SeedCorpus energy）。

### `act/pipeline/__main__.py`　`+0 / −1`

docstring 里少了一个空行。无行为变化。

### `.gitignore`　`+11`

忽略 `act/pipeline/Shiyang/results/*`（但保留里面的 `.sh`）、跑错 cwd 产生的嵌套 results 副本、
`.idea/`。

### `act/pipeline/log/pipeline_tests.log`　**删除** `−126`

一个被误提交进版本库的测试日志。

---

# 第二部分 · 新增文件

## 2.1 fuzzing 引擎（9 个）

### `state_manager.py`　`1,266 行` —— 整条状态线的地基

`PatternStateManager`：全局的 ReLU 符号模式登记表，**被所有变异策略共用**，不只是 HPGD。
它替换掉两样东西：CoverageTracker 驱动的语料库准入（换成真正的状态去重/多样性检查），
和 SeedCorpus 纯能量的调度（换成密度感知的）。两者都只在实例的**不稳定神经元**上做
—— 稳定神经元对盒内任何输入都不会变号，对去重和调度都零信息量。

三个部件，各解决一个问题：

| 部件 | 解决 |
|---|---|
| `_BloomFilter` | 定长内存的近似成员检查，做「这个模式见过吗」的快速预筛 |
| `_StateBKTree` | Hamming 距离上的 BK-tree，做「有没有见过距离 ≤ d 的模式」 |
| `compute_unstable_mask` | 用现有后端的区间传播算候选集 |

其它重要函数：

* `compute_gradient_budget_masks()` —— 用盒子的一阶梯度预算算候选集，替代区间传播。
  注意力图上区间传播的误差会**相乘**放大（`vit_2023` 返回 960/960 全不稳定，宽度 1e12
  而真值 0.10），这个函数返回 74/960。**不 sound**，但这个掩码只用来选攻击目标。
* `compute_per_instance_masks()` —— 每行是该实例**自己的**不稳定集（修 `row0` 那个坑）。
* `flip_balance_scores()` / `margin_gradient_scores()` / `ce_prior_from_registry()` /
  `select_state_dims()` —— 状态维度筛选的四种打分方式（`--state-dim-select`）。

### `state_bins.py`　`202 行` —— 预激活 → 状态码

ReLU 的状态是 `sign(z)`，每个神经元一位，在拐点处切开。原样搬到 Sigmoid/Tanh 上，
这一位在拐点（0）处切开，而实测 ERAN 网的预激活落在 `|z| ≈ 3.5–22`，**几乎没有神经元
能在盒内够到 0**，「状态」几乎恒定。

三段划分在 `z = -τ, +τ` 切两刀，给这些神经元一个够得着的边界。实测（一堵墙落在盒子
一阶梯度预算之内才算数）：sigmoid 24/600 → 54/600，tanh 20/600 → 40/600，**可达墙多 2.0–2.3 倍**。

`StateBinning` 把三段编码成**每个神经元两个 ±1 坐标**（`sign(z+τ)`, `sign(z−τ)`），
所以 BK-tree、Bloom、指纹打包、边缘分布全都不用改；相邻两个坐标属于同一个神经元，
移动一格恰好是 Hamming 距离 1。`(-1,+1)` 不可达，`repair()` 把目标投影回可达集合。

### `pattern_search_pgd.py`　`755 行`

模式空间 PGD，采集/攻击分离，最多三轮。第 1 轮（广域探测）：在实例输入盒里采一个随机点，
翻转它自身模式的一个随机子集当目标，用 margin loss 做投影梯度行走；每个投影点只有在
**实际达成的符号模式**与池中所有模式至少相差 `--min-diversity-hamming` 个神经元时才入池，
所以池子覆盖的是很多不同的激活区域而不是同一个区域的近似副本。每轮采集之后立刻攻击：
从每个新入池的点做违反性最大化的 PGD，用实例自己的 `OutputSpec` 检查是不是真反例。

### `random_start_pgd.py`　`324 行` —— 上一个的消融基线

每次重启完全独立：在输入盒里采一个均匀随机点（**没有**模式定向、**没有**多样性池、
**没有**两阶段），然后跑和 PatternSearchPGD 第二阶段一模一样的攻击 PGD。
用来隔离「PatternSearchPGD 的产出有多少来自模式空间的多样性搜索，多少只是攻击 PGD 本身」
—— 同样的损失、同样的步数、同样的加载与标签解析，**唯一的差别是起点分布**。

### `auto_attack_pgd.py`　`389 行` —— 更强的消融基线

AutoAttack 四阶段集成（APGD-CE → APGD-T → 简化 FAB-T → 简化 Square）的**从头重写**。

**为什么不直接包官方的 `autoattack` 包**：官方实现把有效像素域硬编码成
`clamp(0., 1.)` 加一个标量 `eps`。本项目的 VNNLIB 实例（比如 `cifar100_2024`）的攻击盒
定义在**归一化后的**输入空间里，lb/ub 直接来自解析出的 VNNLIB 边界，不是 `x ± eps` 再裁到
`[0,1]`。喂给官方包会**静默地攻击错误的可行域**。这里每个攻击都投影回实例自己的
`(lb, ub)` 盒 —— 和 PatternSearchPGD / RandomStartPGD 用的、已经对着 VNNLIB spec 验证过的
同一个盒。

### `auto_pgd_batched.py`　`237 行`

Auto-PGD 系列的批量版（每行独立），给 ACTFuzzer 的 BI 两段式变异用。是
`pattern_search_pgd.py` 里 batch=1 版本的直接向量化：每个样本有自己独立的步长 / 最优损失 /
目标类，而不是共用一个标量。`_attack_pgd` / `_violation_score` 原样复用——它们本来就写成了
batch 泛型（纯 gather/scatter/max，一个共享的步长浮点数）；而 DLR/CW 损失和 `_auto_pgd_core`
假定了 B=1（`outputs[0]`、python 标量记账），是真重写的。

### `bi_gce_fuzz.py`　`205 行`

让 ACTFuzzer 只针对**单个** VNNLIB 实例跑，选实例的方式和三个 PGD runner 一致
（`--category` / `--max-instances` / `--instance-index` / `--model-index`，CLI > 本脚本的
`--config` YAML > 代码默认）。这和 `python -m act.pipeline --fuzz` 不同，后者会把最多
`--max-instances` 个实例批在**一次** fuzzing 里。用来隔离一个难实例（比如 PatternSearchPGD
啃不动的那个），把 BI/GCE 的状态驱动搜索专门指过去。BI/GCE 本身的配置仍然来自
`act/config/pipeline.yaml` 的 `fuzzing:` 段。

### `bi_threads.py`　`136 行`

BI 的生产者/消费者线程：一个只跑 HPGD 的线程持续往有界队列里灌候选，一个只跑 PGD 的
线程持续从队列取候选跑（昂贵的）攻击。只在 `bi_threaded=True` 时启用。

**为什么**：HPGD 便宜（`hpgd_num_steps` 步梯度）而且是真正在发现新的、多样的状态；
PGD 昂贵（`bi_attack_pgd_steps`，比如 50 步）而且负责把候选变成真违反。把两者 1:1 绑死，
会让候选发现的吞吐被攻击吞吐卡住，尽管两件事逻辑上互相独立。每个线程一份模型副本
（共享模型会竞争）。

### `test_observe_batch.py`　`106 行`

指纹快路径对暴力参考实现的正确性测试。两个独立命题分开测，好把**语义**改动和**实现**
改动隔离开：(1) `observe_batch(per_instance=False)` 必须逐字复现原来的准入判定
（「当且仅当这个精确模式已被接纳过才拒绝，按 lane 顺序扫描，所以同 batch 中靠前的 lane 算数」）；
(2) `per_instance=True` 必须把同一条规则**按实例**施加。参考实现故意写成暴力的
（和每个已存模式逐个比，不用树不用指纹），这样它不可能和被测实现共享同一个 bug。

## 2.2 配置（4 个新 yaml）

| 文件 | 干什么 |
|---|---|
| `act/config/pattern_search_pgd.yaml` | PatternSearchPGD 默认跑哪个 benchmark/实例，以及各阶段参数 |
| `act/config/random_start_pgd.yaml` | 上者的消融基线的配置 |
| `act/config/auto_attack_pgd.yaml` | AutoAttack 四阶段基线的配置 |
| `act/config/bi_gce_fuzz.yaml` | BiGceFuzz 针对哪个单实例、什么预算 |

四个都遵循同一个优先级：**命令行 > 这个 YAML > 代码默认**。

## 2.3 实验 driver 与脚本（12 个）

### `pipeline/paper_cifar100_batch_ani.py`　`993 行` —— 唯一的入口

原本是论文 Cifar100 Batch-Ani（各向异性）实验的复现，现在承载**全部**消融：
一个 arm = 一组它的命令行参数，44 个参数的完整说明见 [`PARAMETERS.md`](PARAMETERS.md)。
它负责加载 category、合成批处理模型、构造 `FuzzingConfig`、跑 N 次重复、
把每组的 `group_summary.json` 写到 `--output` 下。

### `pipeline/common.py`　`49 行` —— 一个不起眼但关键的修补

`initial_seeds_from_wrapped_model()`：从已经批过的 wrapped model 里取出逐实例的种子。

**为什么需要它**：VNNLIB 来源的实例**从来不填** `InputLayer.labeled_input.label`，
真实类别实际上住在 `OutputSpecLayer` 自己的 spec 上（`OutputSpec.y_true`，从性质的约束
推出来）。退回 `label=None`（在 `FuzzingSeed` 里变成 `-1`）会让 PGDMutation / HPGDMutation
**静默地**掉到它们的无标签目标函数（「最大化输出方差」）而不是朝 `y_true` 的定向 CW margin
损失，也会让 PropertyChecker 分不清是哪个类。

### 汇总与分析

| 脚本 | 行数 | 回答什么 |
|---|---|---|
| `ce_diversity.py` | 801 | **(D, S) 反例多样性分析**。D = 攻破的不同实例数（inter-instance），S = 反例的仿射约束方向 `d severity / d x` 的平均两两余弦距离（intra-instance）。读 `--dump-founder-clusters` 写出的 `ce_sample.npz`。约束方向取自各组自己的 `OutputSpec.severity` 而不是手写的 TOP1 margin —— safenlp 的 spec 是 `UNSAFE_LINEAR`，压根没有真实类别。S 是**精确**的：所有对、不做抽样稀释。 |
| `summarize_state_ab.py` | 44 | 读一个结果目录下所有 `<arm>/rep*/<group>/group_summary.json`，打印逐 arm 的 violations 和 distinct 的均值 ± sd。 |
| `distinct_ceiling.py` | 97 | **一组实例里到底有多少个是可证伪的？** `distinct_instances_hit` 混淆了两件事：搜索够到了多少实例，和有多少实例本来就能破。distinct=1 在天花板是 1 还是 15 的情况下含义完全不同。这里用对每个实例都公平的方式测上界：每条 lane 每轮都攻击**自己的** spec，全新随机重启，对 spec 自己的 severity 做朴素梯度上升——没有语料库、没有调度、没有 lane 竞争。 |

### 状态码 / 光滑激活

| 脚本 | 行数 | 回答什么 |
|---|---|---|
| `smooth_state_probe.py` | 595 | **在 Sigmoid/Tanh 上「状态」还能指什么？** ReLU 的状态是不稳定神经元上的 `sign(z)`，不稳定 = 盒子没有把神经元钉在拐点的哪一侧。光滑激活没有拐点，所以在设计替代划分之前，得先知道哪些候选定义在验证盒里**真的会动**。逐激活层报告：跨零神经元（区间传播给的 sound 但松的计数）、以及其它几种口径。 |
| `bin_transfer_probe.py` | 151 | **HPGD 能不能把「指定的」神经元推进「指定的」段？** 表达这个移动是解决了的（`StateBinning.write_bin` 能精确命名六种 (源, 目标) 段对），但**够得着**是另一回事：高→中跨一堵墙，高→低跨两堵，后者需要盒子把预激活挪动超过 `2τ`。跑的是**生产环境的**铰链投影，每条 lane 一个目标神经元，逐 (源段→目标段) 报告命中率。这是最有利的设定——单目标、没有竞争的保持项、没有联合约束。 |
| `two_segment_reachability.py` | 140 | **跨两段的转移在真实观察到的状态里存在吗？** 上一个脚本的目标是 `write_bin` 伪造的，所以 sigmoid 8% / tanh 0% 这个数**不可归因**：「投影太弱」和「盒子根本实现不了那个神经元的那一段」会给出同一个 0（这就是「目标必须真实可达」那条纪律）。把钳制盒放大到 4 倍能把它抬到 38%，这是第二种解释的证据。所以在为「跨两段」做任何工程之前，先问这件事到底发不发生：像 fuzzer 一样把登记表填满**真实观察到的**状态，再逐实例地看。 |
| `unstable_mask_scope.py` | 175 | **不稳定神经元掩码是单条 lane 的正确候选集吗？** `ACTFuzzer._init_state_manager` 给整组算**一个**掩码（`compute_unstable_mask(model, lb, ub)`，lb/ub 带着全部 B 行），于是一个神经元只要在 batch 里**某处**跨零就算候选。下游全部作用域都被这个集合限死：HPGD 只朝它里面瞄，PatternStateManager 只在它里面算新颖性，粗到细的调度也只在它里面测前沿。这个脚本量化它在两个相反方向上出错的程度。 |

### HPGD

| 脚本 | 行数 | 回答什么 |
|---|---|---|
| `hpgd_real_target_hitrate.py` | 416 | **HPGD 该瞄哪里 —— 以及「落点」本身是不是目标？** HPGD 靠**独立地**翻转种子自身 ReLU 符号模式的 k 位来造目标，没有任何东西检查结果是联合可满足的，所以低命中率**不可归因**：「投影弱」和「那里本来就没东西可到」产生同一个数字。这个脚本第一版就把这件事结了：**在配平位移下**，某个样本真实到达过的目标命中率约 100% 且精确，同距离伪造的目标约 6–12% 且从不精确。**优化器没问题，目标有问题。**而且落点**不是**目标本身，把它当目标优化是错的。 |
| `hpgd_ce_guided_flips.py` | 320 | **按「反例判别性」挑神经元翻转，比随机翻转强吗？** 此前所有神经元挑选准则要么按「动得多少」排（盒内随机点下的符号均衡度），要么根本不排（均匀）。修复侧故障定位真正在用的那个准则——对比**失败**输入和**通过**输入的激活——一直用不上，因为还没有反例的实例提供不了失败侧，而别的实例的反例不迁移（不同的盒、不同的不稳定集；实测跨实例目标的落点和随机目标一样）。这个脚本在数据**确实存在**的地方问这个问题：在已经被攻破的实例上。 |
| `hpgd_then_pgd.py` | 207 | **HPGD 把搜索挪进反例区域后，PGD 的起点是不是更好？** 此前每次测量都在给 HPGD 自己的输出打分，那是在考它不做的事：在流水线里 HPGD 不产生反例，它**重定位**搜索，让 PGD 从别处发起攻击。「一步 HPGD 然后查违反」测的是 HPGD 不承担的工作，于是到处返回 0%。这个脚本直接测两段式的东西，从干净输入出发，两臂在**总梯度步数**上配平：`pgd_only` 用 N+shift 步 PGD 从原始输入出发作为对照。 |

### campaign 启动脚本（`results/*.sh`，5 个）

`results/` 其余内容全部不入库，只留这 5 个启动脚本，因为它们的注释头**记录了每个 campaign
的设计与当时的实测理由**：

| 脚本 | 内容 |
|---|---|
| `verify_safenlp_30x4.sh` | safenlp_2024 全部 1080 个实例，4 臂 × 30 次 × 60 s（约 4.5 小时）。注释里列了每一臂隔离的是什么变量。 |
| `verify_safenlp_10x3.sh` | 同 benchmark 的 3 策略 × 10 次版本，并记录了一个**按组不同**的坑（要读每组打印的 "HPGD-Cov calls:" 那行，不能一概而论）。 |
| `verify_safenlp_4arm.sh` | 在同一个代码版本 `6696a18` 上重跑全部四臂——该 commit 改了状态准入，所以更早的 state 结果都不可比；让一部分臂跑旧代码等于给同样墙钟时间下不同的迭代预算。 |
| `verify_safenlp_isolate.sh` | 隔离 `statehpgd` 那 4.4 倍 distinct 增益到底来自哪：那一臂同时动了两个变量（加 hpgd 策略 **且** 把 admission 从 coverage 换成 state）。 |
| `verify_cifar_30x4.sh` | cifar100_2024 的对照版本，网络不同因而 safenlp 的结论未必迁移：这里覆盖率**不**饱和（59.9% / 65.0%，1761 / 1315 个从未激活的神经元），所以 hpgd_cov 真的能引导。 |

## 2.4 文档（新增）

| 文件 | 内容 |
|---|---|
| `act/pipeline/Shiyang/README.md` | 实验入口：怎么准备 benchmark、一个最小的端到端流程、脚本索引、两条方法学纪律 |
| `docs/PARAMETERS.md` | 44 个参数的默认值/取值/作用 + 实验里用过的 arm 配方 + 踩过的坑 |
| `docs/gen_parameters_doc.py` | 从 driver 的 argparse 重新生成上面的表；代码里加了参数而文档没加就报错 |
| `docs/CHANGES_VS_MAIN.md` | 与上游 `main` 的差异总表（脚本生成） |
| `docs/gen_changes_doc.py` | 从真实 `git diff` 生成上表；改了却没写理由的文件会让它报错 |
| `docs/FILE_REFERENCE.md` | **本文件** |
| `docs/EXPERIMENT_LOG_20260909_smooth_state.md` | 光滑激活两轮实验的完整记录 |
| `docs/EXPERIMENT_LOG_20260827_28.md` | 五个 benchmark 的调度线实验，18 节 |
| `docs/CE_DIVERSITY_METRIC.md` | (D, S) 两个指标的推导 |
| `ACT_PIPELINE_HELP.md` | `python -m act.pipeline --help` 的 Markdown 版 |

## 2.5 benchmark 工具

### `data/vnnlib/eran_sigmoid_tanh_mlp/`（4 个文件，**不含**任何 ONNX 或 spec）

| 文件 | 内容 |
|---|---|
| `README.md` | benchmark 的来源、六个网络的 eps、两个 ONNX 变体的区别、重建步骤 |
| `generate_specs.py` | 按 GenBaB / α,β-CROWN 的同一套配方生成 VNNLIB 2.0 spec 和 instances.csv：MNIST 测试集前 100 张，逐模型的 L∞ epsilon，只保留模型分类正确的图。规约写在原始 [0,1] 像素上，和 `mnist_fc` 一致。 |
| `fold_normalization.py` | 把 ERAN 网内嵌的 MNIST 归一化折进第一个 Gemm：`W((x−m)/s)+b == (W/s)x + (b−Wm/s)`。原图开头是 `Sub(mean) → Div(std) → Flatten → Gemm`，`torch2act` 建不了那个前置广播（`Var-var 'sub' size mismatch (784 vs 1)`）。输入仍是原始 [0,1] 像素，所以 spec 不用改。折完对着原图在前 100 张测试图上验等价。 |
| `info.json` | benchmark 元数据 |

### `tools/`（4 个）

| 文件 | 内容 |
|---|---|
| `batch_download_vnnlib_categories.py` | 批量下载 VNN-COMP 2026 的若干 category（2.0 格式）到 `data/vnnlib/<category>/`，用的是 ACT 自己的下载器。下载器本身已经做了「解压 + 按 instances.csv 命名」，所以不需要额外的重命名步骤。 |
| `convert_vnnlib_1_to_2.py` | 把 VNNLIB 1.0 扁平格式转成 ACT 唯一能读的 2.0。ACT 只支持 2.0，1.0 的 category（`mnist_fc` 在内）加载时会失败。两种格式的差别是机械的，不是语义的。 |
| `convert_vnnlib_2_to_1.py` | 反向转换——官方发布的 α,β-CROWN 的 `read_vnnlib.py` 只认 1.0（实测它在我们的 2.0 文件上硬失败：`AssertionError: failed parsing line: (vnnlib-version <2.0>)`）。复用 ACT 自己已经正确的 2.0 括号变量 → 扁平名的 ravel 逻辑，而不是重新实现索引运算。 |
| `stage_vnnlib_1_for_abcrown.py` | 把一整个 2.0 category 转成 1.0 的副本给 α,β-CROWN 用：`onnx/` 和 `instances.csv` 原样拷，每个 `vnnlib/*.vnnlib` 走上面的转换。 |

---

## 附：两条贯穿全部脚本的方法学纪律

这两条被反复咬过之后写进了多个脚本的 docstring，读那些脚本时会一直看到：

1. **配平位移再比较。** 比较两个机制的结果率之前，先把位移/努力量级配平，否则比较没有意义。
   HPGD 固定 `flip_count=10` 就栽在这里：10 次翻转是 safenlp 93 个不稳定神经元的 10.8%，
   却只是 cifar100 ResNet 3225 个的 0.31% —— 同一个名义设置在要求相差两个数量级的位移。
   修复是 `--hpgd-flip-frac`。
2. **目标必须是真实可达的。** 问「X 能不能到达状态 B」时，B 必须是真实观察到/可达的状态。
   如果 B 是凭空构造的，失败只说明 B 不可行，关于 X 什么也没证明。
   `hpgd_real_target_hitrate.py` 和 `two_segment_reachability.py` 都是为了补上这个对照才存在的。
