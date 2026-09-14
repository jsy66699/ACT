"""Regenerate the flag tables inside PARAMETERS.md from the driver's own argparse.

    python act/pipeline/Shiyang/docs/gen_parameters_doc.py

Only the two `<!-- BEGIN ... -->` / `<!-- END ... -->` blocks are rewritten; every
line of prose around them is left alone. The point is that defaults and choices in
the document can never drift from the code -- if a flag is added to
paper_cifar100_batch_ani.py and not to GROUPS below, this script fails loudly
rather than silently omitting it.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DRIVER = HERE.parent / "pipeline" / "paper_cifar100_batch_ani.py"
FUZZER = HERE.parents[2] / "pipeline" / "fuzzing" / "actfuzzer.py"
DOC = HERE / "PARAMETERS.md"

# Which section each flag belongs to, in the order the sections appear.
# Adding a flag to the driver without adding it here is an error, on purpose.
GROUPS: list[tuple[str, str, list[str]]] = [
    ("workload", "选哪些实例、跑多久", [
        "--category", "--max-instances", "--instance-indices",
        "--timeout", "--max-iterations", "--repeat",
        "--device", "--dtype", "--batch-conversion",
    ]),
    ("portfolio", "变异策略组合（谁来产生下一个样本）", [
        "--perturb-scale", "--pgd-only",
        "--hpgd-weight", "--hpgd-cov-weight", "--hpgd-absorb-non-pgd",
    ]),
    ("hpgd", "HPGD 的瞄准方式", [
        "--hpgd-target-mode", "--hpgd-flip-frac", "--hpgd-schedule",
        "--hpgd-expand-frac", "--hpgd-expand-patience", "--hpgd-cov-nearest-margin",
    ]),
    ("state", "状态码：一个神经元如何变成一个坐标", [
        "--admission-mode", "--state-bins", "--state-bin-tau",
        "--state-dims", "--state-dim-select",
        "--unstable-mask", "--unstable-mask-source", "--unstable-mask-grad-threshold",
    ]),
    ("scheduling", "语料库调度：下一个种子从哪来", [
        "--scheduling-mode", "--ce-energy", "--energy-tiers",
        "--ce-parent-replacement", "--ce-parent-energy-threshold",
        "--select-per-instance", "--select-without-replacement",
        "--coverage-per-instance",
    ]),
    ("io", "输出与探针", [
        "--output", "--no-save", "--report-interval", "--verbose",
        "--dump-founder-clusters", "--dump-ce-prior", "--dump-ce-patterns",
        "--ce-prior",
    ]),
]


def parse_driver(src: str) -> dict[str, dict]:
    out = {}
    for m in re.finditer(r'add_argument\(\s*"(--[a-z0-9-]+)"', src):
        i = src.index("(", m.start())
        depth = 0
        for j in range(i, len(src)):
            if src[j] == "(":
                depth += 1
            elif src[j] == ")":
                depth -= 1
                if depth == 0:
                    break
        call = src[i + 1:j]
        h = re.search(r'help=\s*\(?((?:"[^"]*"\s*)+)\)?', call, re.S)
        help_txt = " ".join(re.findall(r'"([^"]*)"', h.group(1))) if h else ""
        help_txt = re.sub(r"\s+", " ", help_txt).replace("%%", "%").strip()
        d = re.search(r"default=([^,\)]+)", call)
        ch = re.search(r"choices=\[([^\]]*)\]", call)
        act = re.search(r'action="(store_true|store_false)"', call)
        default = (d.group(1).strip() if d else ("False" if act and act.group(1) == "store_true" else ""))
        out[m.group(1)] = dict(
            default=default.strip('"'),
            choices=[c.strip().strip('"') for c in ch.group(1).split(",")] if ch else [],
            help=help_txt,
            flag_only=bool(act),
        )
    return out


def first_sentences(text: str, n: int = 2) -> str:
    if not text:
        return "—"
    parts = re.split(r"(?<=[.!?]) +", text)
    s = " ".join(parts[:n]).strip()
    return s.replace("|", r"\|")


def cli_table(args: dict[str, dict]) -> str:
    seen = set()
    lines = []
    for key, title, flags in GROUPS:
        lines.append(f"\n### {title}\n")
        lines.append("| 参数 | 默认 | 取值 | 作用 |")
        lines.append("|---|---|---|---|")
        for f in flags:
            if f not in args:
                sys.exit(f"{f} is in GROUPS but not in the driver's argparse")
            seen.add(f)
            a = args[f]
            default = "（开关）" if a["flag_only"] else (f"`{a['default']}`" if a["default"] else "—")
            choices = ", ".join(f"`{c}`" for c in a["choices"]) if a["choices"] else "—"
            lines.append(f"| `{f}` | {default} | {choices} | {first_sentences(a['help'])} |")
    missing = sorted(set(args) - seen)
    if missing:
        sys.exit("flags present in the driver but missing from GROUPS: " + ", ".join(missing))
    return "\n".join(lines) + "\n"


def config_only_table(fuzzer_src: str, args: dict[str, dict]) -> str:
    """FuzzingConfig fields with no CLI flag -- editable only in code."""
    block = fuzzer_src[fuzzer_src.index("class FuzzingConfig"):]
    block = block[:block.index("\nclass ", 10)]
    cli_names = {f.lstrip("-").replace("-", "_") for f in args}
    # a few flags whose config field is spelled differently
    cli_names |= {"ce_energy_bonus", "select_with_replacement", "unstable_mask_scope",
                  "save_counterexamples", "output_dir", "timeout_seconds"}
    rows = []
    for m in re.finditer(r"^    ([a-z_][a-z_0-9]*): ([^=\n]+?)(?: = (.+))?$", block, re.M):
        name, typ, default = m.group(1), m.group(2).strip(), (m.group(3) or "").strip()
        if name in cli_names or not default:
            continue
        rows.append(f"| `{name}` | `{default}` | `{typ}` |")
    if not rows:
        return "（当前没有仅代码可改的字段。）\n"
    return ("| 字段 | 默认 | 类型 |\n|---|---|---|\n" + "\n".join(rows) + "\n")


def splice(doc: str, name: str, body: str) -> str:
    a, b = f"<!-- BEGIN {name} -->", f"<!-- END {name} -->"
    i, j = doc.index(a) + len(a), doc.index(b)
    return doc[:i] + "\n" + body + doc[j:]


def main() -> None:
    args = parse_driver(DRIVER.read_text(encoding="utf-8"))
    doc = DOC.read_text(encoding="utf-8")
    doc = splice(doc, "CLI", cli_table(args))
    doc = splice(doc, "CONFIG_ONLY", config_only_table(FUZZER.read_text(encoding="utf-8"), args))
    DOC.write_text(doc, encoding="utf-8")
    print(f"{DOC.relative_to(HERE.parents[3])}: {len(args)} flags in {len(GROUPS)} sections")


if __name__ == "__main__":
    main()
