# 实现说明：状态码 · state 准入 · HPGD · 光滑激活

这份文档只讲这次实验涉及的三块改动，讲到「哪个类、谁调用它、怎么完成」的粒度。
其余改动（AutoAttack 基线、benchmark 下载工具、cora/safenlp 的调度线等）不在这里。

行号对应本分支当前代码，格式 `文件:行`。

---

## 0. 三块东西的关系

```
                      ┌──────────────────────────────────────┐
                      │  StateBinning  (state_bins.py)       │
                      │  预激活 z ──► 状态码 code ∈ {−1,+1}^M │
                      └───────┬──────────────────────┬───────┘
                              │                      │
              ┌───────────────▼──────────┐   ┌───────▼───────────────────┐
              │ PatternStateManager      │   │ HPGDMutation              │
              │ (state_manager.py)       │   │ (mutations.py)            │
              │ 记「这个状态见过吗」     │   │ 把种子推向一个目标状态码  │
              │ → 准入 / 能量 / 调度     │   │ → 产生新的子样本          │
              └───────────────┬──────────┘   └───────┬───────────────────┘
                              │                      │
                      ┌───────▼──────────────────────▼───────┐
                      │  ACTFuzzer  (actfuzzer.py)           │
                      │  _init_state_manager  装配           │
                      │  _observe_state       每轮喂状态     │
                      │  fuzz() step 6        准入+能量      │
                      └──────────────────────────────────────┘
```

三者共用**同一个坐标空间**。这是整套东西能对上的前提：`StateBinning` 决定一个神经元
变成几个坐标，掩码、BK-tree 宽度、HPGD 的候选集和目标全都活在这个坐标空间里
（`actfuzzer.py:770-777` 有一段注释专门强调这件事）。

**开关**：`--admission-mode state` 打开准入那条线，`--hpgd-weight 0.5` 打开 HPGD 那条线。
HPGD **要求** state 准入同时打开，因为它的翻转候选集就是准入所打分的那个子空间
（`paper_cifar100_batch_ani.py` 的 `build_mutation_weights` 的 docstring 里写明了这个配对规则）。

---

## 1. 状态码：`StateBinning`

**文件** `act/pipeline/fuzzing/state_bins.py`（202 行，本次新增）

### 1.1 它解决什么

ReLU 的状态天然是 `sign(z)`：一个神经元一位，在拐点（0）处切开。原样搬到 Sigmoid/Tanh
上，这一位仍然在 0 处切开，而实测 ERAN 网的预激活落在 `|z| ≈ 3.5–22` —— **盒子内几乎没有
神经元够得着 0**，「状态」几乎恒定，准入永远说「见过」。

三段划分在 `z = −τ, +τ` 各切一刀，给这些神经元一个够得着的边界。按「一堵墙落在盒子
一阶梯度预算之内才算可达」来数：sigmoid 24/600 → **54/600**，tanh 20/600 → **40/600**。

### 1.2 编码：三段 = 两堵墙 = 每神经元两个坐标

```python
code = sign(z − threshold)        # thresholds 按坐标排列
# bins=2:  thresholds = [0, 0, ..., 0]                  M = N
# bins=3:  thresholds = [−τ, +τ, −τ, +τ, ...]           M = 2N
```

相邻两个坐标属于同一个神经元。这么编码的好处是**字母表还是 ±1**，所以 BK-tree、
Bloom filter、指纹打包、边缘分布统统不用改；而且移动一个段恰好等于 Hamming 距离 1。

三个段的编码：

| 段 | `(sign(z+τ), sign(z−τ))` | `bin_of` |
|---|---|---|
| 低 `z < −τ` | `(−1, −1)` | 0 |
| 中 `−τ ≤ z ≤ +τ` | `(+1, −1)` | 1 |
| 高 `z > +τ` | `(+1, +1)` | 2 |
| **不可达** | `(−1, +1)` | — 要求 `z < −τ` 且 `z > +τ` |

### 1.3 方法一览与调用方

| 方法 | 干什么 | 谁调用 |
|---|---|---|
| `build(num_neurons, bins, tau, device)` | 造 thresholds 表 | `ACTFuzzer._init_state_manager` `actfuzzer.py:768` |
| `num_coords` | M（bins=3 时是 2N） | 同上 `:771`，下游一切宽度都用它 |
| `code(z)` | `[B,N]` 预激活 → `[B,M]` 状态码 | `_activation_sign_pattern_batched` |
| `shifted(z)` | `[B,M]` 的 `z[神经元] − threshold`，**保留 autograd 图** | `HPGDMutation.mutate` 的铰链损失 |
| `expand_bounds(lb, ub)` | 哪些坐标被盒子跨过（`lb < threshold < ub`） | `compute_unstable_mask` 系列 |
| `repair(code)` | 把 `(−1,+1)` 投影回可达集合 | `HPGDMutation.mutate` 造完目标之后 |
| `bin_of(code)` | `[...,M]` → `[...,N]` 段号 ∈ {0,1,2} | HPGD 的段级目标生成 |
| `write_bin(code, rows, neurons, dest)` | 原地把某些神经元写成指定段 | 同上 |
| `neuron_index_of_coord(idx)` | 坐标下标 → 神经元下标 | 同上（从抽到的坐标回推神经元） |

`bins=2` 时 `build` 返回的对象让每个方法都退化成恒等：`code` 就是 `sign(z)`，
`shifted` 直接返回 `z`，`repair` 原样返回。**所以 `--state-bins 2` 和改动前逐字节一致**
（实测：mask 在新旧代码下都报 355/600）。

### 1.4 一个真实的限制

τ 是**全局固定**的，不是逐神经元的。有些层的 `|z|` 跑到 20，固定 τ=1 会让那些神经元的
三个段塌成一个。这是第一版的已知短板，写在 `--state-bin-tau` 的 help 里。

---

## 2. 光滑激活：怎么把 Sigmoid/Tanh 的预激活取出来

**文件** `act/pipeline/fuzzing/mutations.py`（三个模块级函数，本次新增）

### 2.1 `_is_activation_module(module)` —— 一个非踩不可的坑

```python
if isinstance(module, (nn.ReLU, nn.Sigmoid, nn.Tanh)):
    return True
fn = getattr(module, "function", None)          # onnx2torch 的 OnnxFunction 包装
return callable(fn) and getattr(fn, "__name__", "") in _ACTIVATION_FN_NAMES
```

**为什么需要第二段**：`onnx2torch` 对 ONNX 的 `Tanh` **不发射 `nn.Tanh`**，它发射一个
持有 `torch.tanh` 的 `OnnxFunction` 包装。只做 `isinstance` 判断的话，在所有 ONNX 来源的
tanh 图上（每一个 ERAN tanh 网都是）会**静默地一个激活层都找不到**，调用方看到的是
「模型没有激活层」，或者更糟——一个空的 pattern。

### 2.2 `_activation_preactivations_batched(model, x) -> [B, N]`

用 `register_forward_pre_hook` 在每个激活层上挂钩子，跑一次 `model(x)`，把每层的**输入**
（也就是预激活）收集起来 `flatten(start_dim=1)` 后 `cat`。顺序是模块注册顺序。

**保留 autograd 图** —— HPGD 的铰链损失要对它求梯度。

### 2.3 `_activation_sign_pattern_batched(model, x, binning=None) -> [B, M]`

`no_grad` 下取预激活，然后：`binning` 为 `None` 就是老的 `sign(z)`（M = N，所有老调用方
不受影响）；给了 binning 就走 `binning.code(z)`。

文件末尾两行别名保证了向后兼容：

```python
_relu_preactivations_batched  = _activation_preactivations_batched
_relu_sign_pattern_batched    = _activation_sign_pattern_batched
```

所以 `actfuzzer.py` 里那些 `_relu_*` 的调用点一行没改，但现在它们认 Sigmoid/Tanh。

---

## 3. 候选集：哪些坐标算「可翻转」

**文件** `act/pipeline/fuzzing/state_manager.py`（模块级函数）

准入和 HPGD 都只在**候选集**上工作。稳定神经元对盒内任何输入都不会变号，对去重和调度
都零信息量。三种造法，由 `--unstable-mask-source` 和 `--unstable-mask` 选：

| 函数 | 判据 | 对应参数 |
|---|---|---|
| `compute_unstable_mask(model, lb, ub, scope, binning)` | 区间传播（IBP）：`binning.expand_bounds(lb,ub)`，即盒子是否跨过该坐标的墙 | 默认 |
| `compute_per_instance_masks(...)` | 同上，但**每行是该实例自己的集合**，不做并集 | `--unstable-mask per_instance` |
| `compute_gradient_budget_masks(..., threshold)` | 盒子的一阶梯度预算 `ε·‖dz/dx‖₁` 够不够到墙 | `--unstable-mask-source gradient` |

`scope` 三档（`--unstable-mask`）：

* `row0`（默认，也是 2026-08-27 之前所有实验跑的）—— 只用**第一个实例**的盒子，套给全部 B 条 lane。
  实测一条 lane 自己的不稳定集和它只重叠 9–12%，真正发生的符号翻转只有 6.6–7.7% 落在里面。
* `union` —— 所有行的盒子取或，是每条 lane 自己集合的真超集。
* `per_instance` —— 每条 lane 一行。

梯度预算那一版**不 sound**，但这个掩码只用来选攻击目标。它存在的理由是区间传播在注意力图
上误差会相乘放大（`vit_2023` 返回 960/960 全不稳定、宽度 1e12 而真值 0.10，梯度预算返回 74/960）。
在 ERAN 上它也把 IBP 的 355/600 收紧到 24/600（2-bin）/ 54/600（3-bin）。

---

## 4. state 准入：`PatternStateManager`

**文件** `act/pipeline/fuzzing/state_manager.py`（1266 行，本次新增）

### 4.1 三个部件

| 部件 | 解决 | 说明 |
|---|---|---|
| `_BloomFilter` | 「这个精确模式见过吗」的**快速预筛** | 定长内存（`--state-bloom-bits`，默认 2²⁰ 位，4 个哈希） |
| `_StateBKTree` | 「见过距离 ≤ d 的模式吗」 | Hamming 距离上的 BK-tree，限制在候选集上 |
| `restrict()` / `fingerprint()` | 把 `[B,M]` 全码砍到候选集、再打包成指纹 | 指纹让 exact-match 能**批量**回答 |

`diversity_threshold`（`--state-diversity-threshold`，默认 1）决定走哪条路：
**等于 1 时是纯 exact-match**，可以走指纹的批量快路径；大于 1 是 Hamming 球查询，
指纹表达不了，退回逐样本的 BK-tree 路径。这个分支在 `ACTFuzzer._observe_state` 开头
（`actfuzzer.py:944-948`）。

### 4.2 装配：`ACTFuzzer._init_state_manager(initial_seeds)`

`actfuzzer.py:740` 起，只在 `admission_mode ∈ {state, state_always}` 或 `enable_bi_gce`
时才建（`:683`）。步骤：

1. **探针 forward 数神经元**。
   ```python
   sample = torch.cat([s.tensor for s in initial_seeds], dim=0)   # 必须是全部 B 条 lane
   total_neurons = _relu_preactivations_batched(self.model, sample).shape[1]
   ```
   注释里记着一个真事故：这里走的是 **wrapped model**，它的 forward 会顺带评估输出规约；
   `TOP1_ROBUST` 的 `y_true` 是按行索引的，喂一条 lane 会抛
   `"TOP1_ROBUST: y_true carries B spec rows but the batch has 1 lanes"`，
   而三行之后一个裸的 `except Exception` 会把它吞掉，变成**静默丢失的 unstable mask**。
   safenlp 的 `UNSAFE_LINEAR` 不是按行索引的，所以早期所有 state 实验都没发现这个问题。

2. **建 binning**，把神经元数换算成坐标数 `total_coords`（`:768-777`）。

3. **算候选掩码**，按 `unstable_mask_source` / `unstable_mask_scope` 分派到第 3 节那三个函数
   （`:788-830`）。任何异常都被捕获成 `mask_reason`，并**明确打印**
   `"unstable mask UNAVAILABLE (...) -- falling back to all N coordinates"`
   —— 是上一条事故之后加的，就是为了不再静默退化。

4. **造 `PatternStateManager`**，把掩码和 `total_coords` 交给它。

5. **把参数装到 `hpgd` 策略上**（`:904-915`，见第 5 节）。

### 4.3 每轮观测：`ACTFuzzer._observe_state(...)`

`actfuzzer.py:932`。输入：父种子、子输入、变异前的 `natural_pattern`、变异后的
`achieved_pattern`、违反掩码。输出 `[B]` 的 bool「是否被接纳」。

快路径（`diversity_threshold == 1`）：

```python
admitted = self.state_manager.observe_batch(
    seed_tensors=child_inputs, patterns_full=achieved_pattern,
    labels=..., original_tensors=..., original_indices=...,
    is_ce_mask=violation_mask)
```

`observe_batch` 的语义（`test_observe_batch.py` 对着暴力实现验过）：
**当且仅当这个精确模式已经被接纳过才拒绝，按 lane 顺序扫描**，所以同一个 batch 里靠前的
lane 会影响靠后的。`per_instance=True` 时同一条规则**按实例**独立施加。

被接纳的样本还会更新 `local_bias`：

```python
flipped = (restrict(natural)[b] != restrict(achieved)[b]).nonzero()
self.state_manager.update_local_bias(child_inputs[b:b+1], flipped)
```

也就是「刚刚从这个种子成功翻动过的坐标」会在下次被 HPGD 优先抽到（见 5.2 的 `flip_weights`）。

### 4.4 准入与能量：`fuzz()` step 6

`actfuzzer.py:1229-1287`。四个分支，这是整个实验的核心开关：

```python
if admission_mode == "always":                       # :1237
    interesting_mask = 全 True
    energies = cov_interesting*10 + violation*ce_energy_bonus      # ← coverage 的能量

elif admission_mode in ("state", "state_always"):    # :1245
    achieved_pattern = _relu_sign_pattern_batched(model, inputs, binning=self.binning)
    admitted = self._observe_state(seeds, inputs, natural_pattern,
                                   achieved_pattern, violation_mask)
    interesting_mask = (全 True) if state_always else (violation_mask | admitted)
    energies = admitted*10 + violation*ce_energy_bonus             # ← state 的能量

else:                                                # :1273 默认
    interesting_mask = violation_mask | cov_interesting
    energies = cov_interesting*10 + violation*ce_energy_bonus
```

然后 `energies = clamp(energies, min=0.1)`，可选地被 `--energy-tiers` 整个换掉
（`:1275-1286`，从加法能量里反解出两个 flag 再查表）。最后

```python
self.seed_corpus.add(child_seeds, interesting_mask, **self._ce_add_kwargs(violation_mask))
```

**这四个分支为什么长这样**：`always` 和 `state_always` 是**隔离臂**。
`always` 把闸门开到底但保留 coverage 的能量公式，`state_always` 把闸门开到底但保留 state
的能量公式。两者一对比就把「闸门」和「能量公式」拆开了 —— 实验结论是**收益全在能量公式**，
闸门整个拿掉都不损失（sigmoid statebase 4.6 vs state_always 4.8，无法区分）。

另外注意 `violation_mask | admitted`：**反例无论状态新不新颖都进语料库**。
所以 `state_novelty.admitted/observed`（0.4–5%）**不是拒绝率**，它只统计有多少样本被判为新颖。

---

## 5. HPGD：`HPGDMutation`

**文件** `act/pipeline/fuzzing/mutations.py`

### 5.1 怎么进入变异组合

driver 的 `build_mutation_weights`（`paper_cifar100_batch_ani.py`）。关键规则是
**「displacing 而不是 diluting」**：

`MutationEngine` 拿到权重字典会先归一化，然后**每轮只抽一个**策略。所以直接
`weights["hpgd"] = 0.5` 会把分母从 1.0 变成 1.5，**每一项都缩水 1/3，pgd 也一样**
（50% → 33.3%）。HPGD 的目标函数里没有任何反例项，那些被挪走的迭代对找反例零贡献，
于是产出下降和「pgd 预算少了 38%」混在一起分不开——`verify_large_trials_v2` 就栽在这。

现在的做法：**pgd 保持基线份额**，hpgd 拿走它在旧加法方案下的同一份额（w=0.5 时 33.3%），
差额全部从其余非 pgd 策略里按基线比例扣。这样 `statehpgd` 和 `baseline` 的
**反例搜索预算是配平的**。

### 5.2 ACTFuzzer 把什么装到它身上

`actfuzzer.py:904-915`，紧跟在 `PatternStateManager` 造好之后：

```python
hpgd = self.mutation_engine.strategies.get("hpgd")
if hpgd is not None:
    hpgd.candidate_indices  = self.state_manager.candidate_indices   # 候选坐标
    hpgd.binning            = self.binning if state_bins != 2 else None
    hpgd.flip_count         = config.hpgd_flip_count                 # 固定预算
    hpgd.flip_frac          = config.hpgd_flip_frac                  # 或按比例
    hpgd.num_steps          = config.hpgd_num_steps
    hpgd.margin             = config.hpgd_margin
    hpgd.loss_scope         = config.hpgd_loss_scope
    hpgd.hold_still_weight  = config.hpgd_hold_still_weight
```

`bins == 2` 时故意传 `None`，让 HPGD 走原来那条路。

`flip_weights` 另外由 `local_bias` 逐轮写入（`:616-620`，只在 hpgd 权重 > 0 时才接线）。

### 5.3 `mutate(input_tensor, model, ...)` 的六步

```python
x0 = input_tensor.detach()
natural_pattern = _relu_sign_pattern_batched(model, x0, binning=binning)   # ① 当前状态码
candidates = self.candidate_indices or arange(M)                            #    候选坐标
```

**① 目标从哪来**，三条路，优先级从高到低：

* `target_proposer(natural_pattern)` —— 回调，**在 mutate 内部调用**。必须在内部，
  因为 `MutationEngine` 每轮只抽一个策略，外面不知道这轮会不会轮到 hpgd。
  `ACTFuzzer._propose_hpgd_targets` 就挂在这里，`--hpgd-target-mode interp_real` 时用。
* `target_override` + `target_override_mask` —— 逐 lane 指定。batch 维对不上就**整个丢弃**
  （防止上一批的陈旧目标悄悄操纵这一批）。
* 否则 —— 翻转自己模式的 k 位。

**② 翻转预算 k**：
```python
k = round(flip_frac * C) if flip_frac is not None else min(flip_count, C)
```
`flip_frac` 存在的理由：固定 10 在 safenlp 的 93 个候选里是 10.8%，在 cifar100 ResNet 的
3225 个里是 0.31%——**同一个名义设置在要求相差两个数量级的位移**。跨 benchmark 比较必须用比例。

**③ 抽坐标**：有 `flip_weights` 就 `multinomial`（局部偏置），否则 `randperm`。

**④ 造目标**，这里 2-bin 和 3-bin 分岔：

*2-bin* —— 直接取反：`target_pattern[b, candidates[idx]] *= -1`。

*3-bin* —— **不能**按坐标翻转。三段码的单坐标翻转有六种可能，其中两种是空操作或两段跳跃，
而且「高→低」根本无法表达。所以退回**神经元粒度**：
```python
neurons = unique(binning.neuron_index_of_coord(candidates[local_idx]))  # 坐标→神经元
cur     = binning.bin_of(target_pattern[b])[neurons]                    # 当前段
# bin_move == "adjacent": ±1 段，clamp；卡在端点就往反方向送，保证真的动
# bin_move == "any":      均匀落在它不在的另外两段之一
binning.write_bin(target_pattern, rows=b, neurons, dest)
```

**⑤ 修复**：`target_pattern = binning.repair(target_pattern)`。独立的逐坐标翻转可能造出
`(−1,+1)`，那要求一个神经元同时「低于 −τ」且「高于 +τ」。**瞄一个不可行的目标正是投影
落不到地方的已知原因**，所以先投影回可达集合。

**⑥ 铰链投影**，`num_steps` 步符号梯度下降：

```python
asked = (target_pattern != natural_pattern)
term_weight = asked                     if loss_scope == "target_only"   # 默认
              else asked + (~asked) * hold_still_weight

for _ in range(num_steps):
    z_all    = _relu_preactivations_batched(model, x_req)     # 带梯度
    z_coords = binning.shifted(z_all) if binning else z_all   # ← 3-bin 在这里变成 z ∓ τ
    violation = (margin - target_pattern * z_coords).clamp(min=0)   # 铰链
    loss = (violation * term_weight).sum()
    grad = autograd.grad(loss, x_req)[0]
    x = x - step_size * sign(grad)
    x = clamp(x, x0 - perturb_size, x0 + perturb_size)        # 局部盒
```

**3-bin 对投影本身零改动**：唯一的差别是 `binning.shifted` 把坐标从 `z` 换成 `z − threshold`，
于是同一个铰链把神经元推过 `±τ` 墙，而不是推过 0。

**`loss_scope` 为什么默认 `target_only`**：`"all"` 会让 k 个目标项和 N−k 个「保持不动」项
竞争，而 `sign(grad)` 下每个维度不管想动多少都走满一步，占多数的保持项能把翻转整个抵消。
实测 safenlp：要求翻 10 个，落了 1.5 个，同时有 10–14 个**没被要求的**神经元自己翻了。

返回前记三样东西供调用方使用：`last_natural_pattern`、`last_achieved_pattern`、
`last_target_pattern`（**瞄哪** vs **落哪**，盒子或步数不够时两者不同）。

### 5.4 目标该怎么定：一个必须知道的结论

`random_flip` 是**独立地**翻 k 位，没有任何东西检查结果是联合可满足的。
在配平位移下实测（`hpgd_real_target_hitrate.py`）：

* 某个样本**真实到达过**的目标 —— 命中约 **100%**，且精确；
* 同距离**伪造**的目标 —— 命中约 **6–12%**，且从不精确；生产设置 k=10 下，投影结束时
  离自己的目标**比出发时还远**。

**优化器没问题，目标有问题。** 这也是 `--hpgd-target-mode interp_real` 存在的原因：
瞄种子和某个真实观察到的状态**之间**的一个格子。

---

## 6. 一次迭代的完整时序

```
ACTFuzzer.fuzz()  每轮：
  1. seed_corpus.select(B)                     取 B 个种子
  2. mutation_engine.mutate(...)               ← 只抽中一个策略
       若抽中 "hpgd":
          HPGDMutation.mutate()
            ├ _relu_sign_pattern_batched(x0, binning)   当前状态码
            ├ 造目标（proposer / override / 随机翻转→write_bin→repair）
            └ num_steps × 铰链符号梯度，clamp 回局部盒
       其它策略（pgd / boundary / random）照旧
  3. PropertyChecker                           找违反 → violation_mask
  4. CoverageTracker.update                    照旧（state 模式下结果不被采纳）
  5. step 6 准入 + 能量
       admission_mode == "state":
          achieved = _relu_sign_pattern_batched(x, binning)
          admitted = _observe_state(...)
                       └ PatternStateManager.observe_batch
                            ├ restrict → fingerprint
                            ├ Bloom 预筛 / BK-tree
                            └ 接纳者 update_local_bias(翻动过的坐标)
          interesting = violation | admitted
          energy      = admitted*10 + violation*ce_energy_bonus
  6. seed_corpus.add(children, interesting, ...)
  7. 记账：state_novelty{observed, admitted, ce, ce_and_novel}，并按策略分桶
```

第 7 步那份分策略统计就是实验里「HPGD 产生新状态码的频率比 PGD 高 11–12 倍」那张表的来源
（`group_summary.json` 的 `state_novelty_by_strategy`）。

---

## 7. 相关参数速查

只列这三块用到的，完整 44 个见 [`PARAMETERS.md`](PARAMETERS.md)。

| 参数 | 默认 | 作用 |
|---|---|---|
| `--admission-mode` | `coverage` | `state` 换成状态新颖性准入+密度感知能量；`always` / `state_always` 是把闸门开到底的隔离臂 |
| `--state-bins` | `2` | 2 = `sign(z)`；3 = `±τ` 两堵墙，每神经元两个坐标 |
| `--state-bin-tau` | `1.0` | 三段划分的墙位置，全局固定 |
| `--unstable-mask` | `row0` | 候选集来自第一个实例 / 全部取或 / 逐实例 |
| `--unstable-mask-source` | `ibp` | 区间传播 or 一阶梯度预算 |
| `--hpgd-weight` | `0.0` | > 0 打开 HPGD，占 33.3%（w=0.5），pgd 份额不变 |
| `--hpgd-flip-frac` | `None` | 翻转预算按候选集比例，跨网络才可比 |
| `--hpgd-target-mode` | `random_flip` | `interp_real` 瞄真实可达的中间态 |
| `--state-diversity-threshold` | `1` | 1 = exact-match（走指纹快路径）；> 1 = Hamming 球 |
| `--dump-founder-clusters` | `0` | 写 `ce_sample.npz`，给 `ce_diversity.py` 算 (D, S) |

---

## 8. 已知限制

1. **IBP 才是瓶颈，不是 τ。** 三段的候选集**恰好**是两段的两倍（sigmoid 710/1200 vs 355/600），
   因为传播出的区间宽到 `−τ`、`0`、`+τ` 落在同一批神经元里 —— 多出来的墙在候选集里
   一分钱没赚到。按攻击真正支付的一阶梯度预算算，墙值钱得多（54/600 vs 24/600）。
   **下一个杠杆是候选判据，不是 τ。**
2. **τ 全局固定**，`|z|` 跑到 20 的层上三段会塌成一段。
3. **准入闸门不是机制。** `always` / `state_always` 的隔离证明收益来自能量公式。
   写成「状态新颖性准入提升了 distinct」是把功劳记错了地方。
4. **每个激活只测了一个模型组**（6×100），6×200 和 9×100 没跑过；这个 benchmark
   也没有 ground truth（不是 VNN-COMP category）。
