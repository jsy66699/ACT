"""Convert a VNNLIB 2.0 file (`(vnnlib-version <2.0>)` header, bracket vars
`X[i,j]`/`Y[k]`) to flat VNNLIB 1.0 (`declare-const X_n Real`, `X_n`/`Y_m`)
-- the format the officially-released alpha-beta-CROWN verifier's own
read_vnnlib.py parser expects (confirmed: it hard-fails on our 2.0-format
downloads with "AssertionError: failed parsing line: (vnnlib-version <2.0>)").

Reuses ACT's own already-correct 2.0 bracket-var -> flat-name ravel logic
(act.front_end.vnnlib_loader.vnnlib_parser's private helpers) instead of
reimplementing index math, so the conversion matches exactly what ACT's own
front end already treats as the ground-truth flat-index mapping.

Usage:
    python tools/convert_vnnlib_2_to_1.py <input.vnnlib> <output.vnnlib>
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from act.front_end.vnnlib_loader.vnnlib_parser import (  # noqa: E402
    _extract_vnnlib_2_decl, _numel, _rewrite_vnnlib_2_bracket_vars,
)

_HEADER_RE = re.compile(r"\(\s*vnnlib-version\b.*?\)\s*", re.DOTALL)
_NETWORK_BLOCK_RE = re.compile(r"\(\s*declare-network\b.*?\n\)\s*", re.DOTALL)


def convert(content: str) -> str:
    input_name, _in_dtype, input_shape = _extract_vnnlib_2_decl(content, "input")
    output_name, _out_dtype, output_shape = _extract_vnnlib_2_decl(content, "output")
    num_inputs = _numel(input_shape)
    num_outputs = _numel(output_shape)

    rewritten = _rewrite_vnnlib_2_bracket_vars(content, input_name, input_shape, output_name, output_shape)
    rewritten = _HEADER_RE.sub("", rewritten, count=1)
    rewritten = _NETWORK_BLOCK_RE.sub("", rewritten, count=1)

    decls = [f"(declare-const X_{i} Real)" for i in range(num_inputs)]
    decls += [f"(declare-const Y_{j} Real)" for j in range(num_outputs)]
    return "\n".join(decls) + "\n\n" + rewritten.strip() + "\n"


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(1)
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    content = src.read_text(encoding="utf-8")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(convert(content), encoding="utf-8")
    print(f"Converted {src} -> {dst}")


if __name__ == "__main__":
    main()
