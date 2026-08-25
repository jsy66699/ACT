"""Correctness of the fingerprint fast path against a brute-force reference.

Two separate claims, tested separately so the semantic change is isolated from
the implementation change:

  1. observe_batch(per_instance=False) must reproduce the ORIGINAL admission
     decisions exactly -- "reject iff this exact pattern was already admitted,
     scanning lanes in order so an earlier lane in the same batch counts".
  2. observe_batch(per_instance=True) must apply that same rule PER INSTANCE.

The reference is deliberately brute force (compare against every stored
pattern, no tree, no fingerprint) so it cannot share a bug with either.
"""
import torch
from act.pipeline.fuzzing.state_manager import PatternStateManager

torch.manual_seed(1234)


def reference(batches, instances, per_instance):
    """Ground truth: sequential exact-match against everything admitted so far."""
    seen = []  # list of (instance_or_None, pattern)
    out = []
    for patterns, inst in zip(batches, instances):
        adm = []
        for b in range(patterns.shape[0]):
            p = patterns[b]
            key_inst = int(inst[b]) if per_instance else None
            hit = any(ki == key_inst and bool((sp == p).all()) for ki, sp in seen)
            if hit:
                adm.append(False)
            else:
                adm.append(True)
                seen.append((key_inst, p.clone()))
        out.append(adm)
    return out


def run_impl(batches, instances, per_instance, C):
    m = PatternStateManager(unstable_mask=None, total_neurons=C, device=torch.device("cpu"))
    out = []
    for patterns, inst in zip(batches, instances):
        B = patterns.shape[0]
        adm = m.observe_batch(
            seed_tensors=torch.zeros(B, 3),
            patterns_full=patterns,
            labels=torch.zeros(B, dtype=torch.long),
            original_tensors=torch.zeros(B, 3),
            original_indices=inst,
            is_ce_mask=torch.zeros(B, dtype=torch.bool),
            per_instance=per_instance,
        )
        out.append(adm.tolist())
    return out, m


def make_case(C, B, n_batches, n_distinct, n_instances):
    """Batches drawn from a small pool of patterns, so duplicates are frequent
    both within a batch and across batches -- the cases that separate the
    sequential rule from a naive batched one."""
    pool = torch.randint(0, 2, (n_distinct, C)) * 2 - 1
    batches, instances = [], []
    for _ in range(n_batches):
        idx = torch.randint(0, n_distinct, (B,))
        batches.append(pool[idx].clone())
        instances.append(torch.randint(0, n_instances, (B,), dtype=torch.long))
    return batches, instances


fail = 0
for C, B, nb, nd, ni in [(16, 12, 6, 5, 3), (64, 40, 5, 12, 4),
                         (128, 64, 5, 20, 8), (33, 30, 6, 6, 2), (7, 20, 4, 3, 5)]:
    batches, instances = make_case(C, B, nb, nd, ni)
    for per_inst in (False, True):
        got, mgr = run_impl(batches, instances, per_inst, C)
        want = reference(batches, instances, per_inst)
        tag = "per_instance" if per_inst else "global      "
        if got == want:
            n_adm = sum(sum(x) for x in got)
            print(f"  C={C:4d} B={B:3d} {tag}  OK   (admitted {n_adm}/{B*nb})")
        else:
            fail += 1
            for i, (g, w) in enumerate(zip(got, want)):
                if g != w:
                    print(f"  C={C} B={B} {tag}  MISMATCH batch {i}")
                    print(f"    got ={g}\n    want={w}")
                    break

# The two modes must actually differ, otherwise test 1 passing is meaningless.
batches, instances = make_case(64, 50, 6, 8, 5)
g_glob, m_glob = run_impl(batches, instances, False, 64)
g_inst, m_inst = run_impl(batches, instances, True, 64)
a_glob = sum(sum(x) for x in g_glob)
a_inst = sum(sum(x) for x in g_inst)
print(f"\n  global admits {a_glob}, per-instance admits {a_inst} "
      f"(per-instance must be >= global): {'OK' if a_inst >= a_glob else 'FAIL'}")
if a_inst == a_glob:
    print("  WARNING: identical -- the case did not exercise cross-instance collisions")
    fail += 1

# Registry bookkeeping must match the admitted count.
ok_reg = len(m_inst._registry) == a_inst and m_inst.tree.size == a_inst
print(f"  registry/size bookkeeping matches admitted count: {'OK' if ok_reg else 'FAIL'}")
fail += 0 if ok_reg else 1

print("\nALL PASS" if fail == 0 else f"\n{fail} FAILURE(S)")
