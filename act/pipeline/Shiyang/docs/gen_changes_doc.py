"""Regenerate the file tables inside CHANGES_VS_MAIN.md from the actual diff.

    python act/pipeline/Shiyang/docs/gen_changes_doc.py [--base origin/main] [--head HEAD]

Every changed path must appear in NOTES below. A file that changed without a note
is an error, not a silent omission -- the point of the document is that a reviewer
can see *why* each file differs from upstream, and a table that quietly drops a
file would defeat it.

Only the `<!-- BEGIN ... -->` blocks are rewritten; the prose is left alone.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DOC = HERE / "CHANGES_VS_MAIN.md"

# (regex on the path, group heading, one-line reason). First match wins, so the
# specific patterns come before the catch-alls.
NOTES: list[tuple[str, str, str]] = [
    # ---- the two behavioural fixes to upstream code --------------------------
    (r"^act/pipeline/verification/torch2act\.py$", "上游行为修正",
     "**真 bug 修复**：flatten 层在 B>1 时 out_vars 带 batch 维而 output_shape 不带，"
     "两边都被区间传播检查，没有任何 batch 能同时满足 —— 于是每个纯 MLP benchmark "
     "在 B>1 时**静默丢失 unstable mask**，退化成「所有神经元都是候选」。"
     "新增 `_per_sample_forward`；B=1 的结果逐字节不变。"),
    (r"^act/front_end/vnnlib_loader/onnx_converter\.py$", "上游行为修正",
     "`convert_onnx_to_pytorch(batch_size=)`：把 ONNX 的**符号 batch 维**钉到调用方真正要用的 "
     "lane 数。注意力图（concat CLS + 按计算出的 shape Reshape）被 onnxsim 按这个值常量折叠，"
     "钉成 1 就只能在 B=1 跑，vit_2023 在 B=4 报 `size of tensor a (401) must match tensor b (5)`。"
     "默认仍是 1，形状不依赖 batch 的图完全不受影响。"),
    (r"^act/front_end/spec_creator_base\.py$", "上游行为修正",
     "配合上一条：校验模型 I/O 形状时把 forward 加宽到 `model_batch_size`，但**报告回来的形状"
     "保持 per-sample** —— 否则 per-instance 的 spec 会因为「只有一行而不是 B 行」被拒。"),
    (r"^act/front_end/vnnlib_loader/create_specs\.py$", "上游行为修正",
     "两个新参数，都默认关：`instance_indices` 让调用方直接点名要 instances.csv 的哪几行，"
     "而不必为了拿到第 199 行先把前 199 行全部 ONNX 转换一遍；`batch_conversion` 把上一条的 "
     "batch 钉法接出来。默认路径与原来完全相同。"),
    (r"^act/front_end/vnnlib_loader/data_model_loader\.py$", "上游行为修正",
     "把 `model_batch_size` 从加载入口透传到转换器，并加了 ONNX 转换缓存。"),

    # ---- the fuzzing engine --------------------------------------------------
    (r"^act/pipeline/fuzzing/actfuzzer\.py$", "fuzzing 引擎（改上游文件）",
     "`FuzzingConfig` 从 16 个字段扩到 60+，覆盖 state 准入、HPGD、调度与能量的全部开关，"
     "**默认值全部等于原版行为**。新增 `_init_state_manager` / `_observe_state` / "
     "`_gce_iteration` / `_propose_hpgd_targets`，以及反例先验和反例模式的 dump。"),
    (r"^act/pipeline/fuzzing/corpus\.py$", "fuzzing 引擎（改上游文件）",
     "`SeedCorpus` 加上诊断「能量垄断」所需的全部仪表：逐实例抽取计数 `draws_by_instance`、"
     "丢弃统计 `drop_stats`、CE 血统/founder 追踪、能量分层，以及 `select()` 的无放回与"
     "逐实例轮询两种取样方式和 CE-parent 替换。"),
    (r"^act/pipeline/fuzzing/mutations\.py$", "fuzzing 引擎（改上游文件）",
     "三个新变异策略 `HPGDMutation` / `HPGDCoverageMutation` / `HPGDPullbackMutation`，"
     "以及批量激活模式提取 `_activation_sign_pattern_batched` —— 后者是让状态码能跑在 "
     "Sigmoid/Tanh 上的前提。"),
    (r"^act/pipeline/fuzzing/coverage\.py$", "fuzzing 引擎（改上游文件）",
     "`has_observations()` 区分「还没建 mask」和「已全覆盖」（原来 `get_uncovered_neurons()` "
     "两种情况返回一样）；`GlobalCov(per_instance=)` 让每个实例有自己的覆盖行，"
     "而不是全 batch 共用一个单调并集 —— 并集下 A 实例点亮一个神经元会把门槛抬给其余 99 个。"),
    (r"^act/pipeline/fuzzing/state_manager\.py$", "fuzzing 引擎（新文件）",
     "`PatternStateManager`：Bloom filter + BK-tree 的全局状态登记表。准入、能量、稀疏调度"
     "都从它读，是 `--admission-mode state` 背后的东西。"),
    (r"^act/pipeline/fuzzing/state_bins\.py$", "fuzzing 引擎（新文件）",
     "状态码的编码：2 段（`sign(z)`，ReLU 划分）或 3 段（`±τ` 两堵墙），"
     "3 段时每个神经元两个 ±1 坐标，所以 BK-tree / Bloom / 指纹打包都不用改。"),
    (r"^act/pipeline/fuzzing/bi_gce_fuzz\.py$", "fuzzing 引擎（新文件）",
     "BI（稀疏性引导的探索）/ GCE（在已知反例附近做锚点拉回）双任务循环，共用一个 "
     "PatternStateManager。"),
    (r"^act/pipeline/fuzzing/bi_threads\.py$", "fuzzing 引擎（新文件）",
     "BI/GCE 的生产者-消费者线程，每个线程一份模型副本（共享模型会竞争）。"),
    (r"^act/pipeline/fuzzing/pattern_search_pgd\.py$", "fuzzing 引擎（新文件）",
     "模式空间两阶段 PGD 的独立 runner。"),
    (r"^act/pipeline/fuzzing/(random_start|auto_attack)_pgd\.py$", "fuzzing 引擎（新文件）",
     "独立攻击 runner：随机重启 PGD / AutoAttack。"),
    (r"^act/pipeline/fuzzing/auto_pgd_batched\.py$", "fuzzing 引擎（新文件）",
     "AutoPGD 的批量实现。"),
    (r"^act/pipeline/fuzzing/test_observe_batch\.py$", "fuzzing 引擎（新文件）",
     "批量状态观测的测试。"),

    # ---- configs -------------------------------------------------------------
    (r"^act/config/pipeline\.yaml$", "配置",
     "上面那些开关的默认值和注释。**默认全关**：`admission_mode: coverage` + "
     "`scheduling_mode: energy` 就是原版 fuzzer。"),
    (r"^act/config/.*\.yaml$", "配置", "对应独立 runner 的配置文件。"),

    # ---- experiment code -----------------------------------------------------
    (r"^act/pipeline/Shiyang/pipeline/paper_cifar100_batch_ani\.py$", "实验代码（纯新增）",
     "**所有 campaign 的唯一 driver**。一个 arm = 一组它的命令行参数。"),
    (r"^act/pipeline/Shiyang/pipeline/ce_diversity\.py$", "实验代码（纯新增）",
     "(D, S) 反例多样性分析。"),
    (r"^act/pipeline/Shiyang/pipeline/.*\.py$", "实验代码（纯新增）",
     "分析/探针脚本，每个的完整说明在它自己的模块 docstring 里。"),
    (r"^act/pipeline/Shiyang/results/.*\.sh$", "实验代码（纯新增）",
     "启动 campaign 的 shell 脚本（`results/` 其余内容不入库）。"),
    (r"^act/pipeline/Shiyang/(README|docs/)", "文档（纯新增）",
     "实验入口、参数说明、实验日志、(D,S) 指标推导。"),
    (r"^ACT_PIPELINE_HELP\.md$", "文档（纯新增）",
     "`python -m act.pipeline --help` 的 Markdown 版。"),

    # ---- benchmark plumbing --------------------------------------------------
    (r"^data/vnnlib/eran_sigmoid_tanh_mlp/", "benchmark 工具（纯新增）",
     "ERAN Sigmoid/Tanh benchmark 的生成脚本与说明（ONNX 和 spec 本身不入库）。"),
    (r"^tools/", "benchmark 工具（纯新增）",
     "VNN-COMP category 下载、VNNLIB 1.0/2.0 互转、给 α,β-CROWN 备料。"),

    # ---- housekeeping --------------------------------------------------------
    (r"^\.gitignore$", "杂项", "忽略结果数据、嵌套的 results 副本、`.idea/`。"),
    (r"^act/pipeline/log/pipeline_tests\.log$", "杂项", "删掉一个被跟踪的测试日志。"),
    (r"^act/pipeline/__main__\.py$", "杂项", "docstring 里少了一个空行，无行为变化。"),
]

STATUS = {"A": "新增", "M": "修改", "D": "删除", "R": "重命名"}


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          text=True, encoding="utf-8", check=True).stdout


def note_for(path: str) -> tuple[str, str]:
    for pat, group, note in NOTES:
        if re.search(pat, path):
            return group, note
    sys.exit(f"{path} changed but has no entry in NOTES -- add one")


def collect(base: str, head: str):
    status = {}
    for line in git("diff", "--name-status", base, head).splitlines():
        parts = line.split("\t")
        status[parts[-1]] = parts[0][0]
    rows = []
    for line in git("diff", "--numstat", base, head).splitlines():
        add, rem, path = line.split("\t")
        group, note = note_for(path)
        rows.append(dict(path=path, add=int(add) if add != "-" else 0,
                         rem=int(rem) if rem != "-" else 0,
                         status=STATUS.get(status.get(path, "M"), "修改"),
                         group=group, note=note))
    return rows


def table(rows) -> str:
    order, seen = [], set()
    for _, g, _ in NOTES:
        if g not in seen:
            seen.add(g)
            order.append(g)
    out = []
    for g in order:
        sub = [r for r in rows if r["group"] == g]
        if not sub:
            continue
        a = sum(r["add"] for r in sub)
        d = sum(r["rem"] for r in sub)
        out.append(f"\n### {g}　　<sub>{len(sub)} 个文件 · +{a:,} / −{d:,}</sub>\n")
        out.append("| 文件 | | 行数 | 为什么 |")
        out.append("|---|---|---|---|")
        for r in sorted(sub, key=lambda r: -r["add"]):
            out.append(f"| `{r['path']}` | {r['status']} | +{r['add']:,} / −{r['rem']:,} | {r['note']} |")
    return "\n".join(out) + "\n"


def summary(rows, base: str, head: str) -> str:
    n = len(git("log", "--oneline", f"{base}..{head}").splitlines())
    mb = git("rev-parse", "--short", git("merge-base", base, head).strip()).strip()
    a = sum(r["add"] for r in rows)
    d = sum(r["rem"] for r in rows)
    new = sum(1 for r in rows if r["status"] == "新增")
    mod = sum(1 for r in rows if r["status"] == "修改")
    dele = sum(1 for r in rows if r["status"] == "删除")
    return (
        f"| | |\n|---|---|\n"
        f"| 对比基准 | `{base}` at `{mb}` |\n"
        f"| 相差 commit | {n} |\n"
        f"| 改动文件 | {len(rows)}（新增 {new} · 修改 {mod} · 删除 {dele}） |\n"
        f"| 行数 | +{a:,} / −{d:,} |\n"
        f"| 改动的**上游**文件 | {mod}，其中在默认参数下就生效的只有 1 个（`torch2act.py`，见下） |\n")


def commits(base: str, head: str) -> str:
    lines = git("log", "--reverse", "--format=%h|%ad|%s", "--date=short",
                f"{base}..{head}").splitlines()
    out = ["| commit | 日期 | 标题 |", "|---|---|---|"]
    for l in lines:
        h, d, s = l.split("|", 2)
        out.append(f"| `{h}` | {d} | {s.replace('|', chr(92) + '|')} |")
    return "\n".join(out) + "\n"


def splice(doc: str, name: str, body: str) -> str:
    a, b = f"<!-- BEGIN {name} -->", f"<!-- END {name} -->"
    i, j = doc.index(a) + len(a), doc.index(b)
    return doc[:i] + "\n" + body + doc[j:]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="origin/main")
    p.add_argument("--head", default="HEAD")
    a = p.parse_args()
    rows = collect(a.base, a.head)
    doc = DOC.read_text(encoding="utf-8")
    doc = splice(doc, "SUMMARY", summary(rows, a.base, a.head))
    doc = splice(doc, "TABLE", table(rows))
    doc = splice(doc, "COMMITS", commits(a.base, a.head))
    DOC.write_text(doc, encoding="utf-8")
    print(f"{DOC.name}: {len(rows)} changed files against {a.base}")


if __name__ == "__main__":
    main()
