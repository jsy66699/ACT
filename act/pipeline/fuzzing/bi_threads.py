"""BI producer/consumer threading: an HPGD-only thread that continuously
generates candidates into a bounded queue, and a PGD-only thread that
continuously pulls candidates from that queue and runs the (expensive)
PGD-family attack on them.

Only used when FuzzingConfig.bi_threaded is True (requires bi_attack_strategy
to also be set). Splits BI's normally-synchronous "HPGD then PGD" call
(ACTFuzzer._mutate_hpgd_then_pgd, still used unchanged for the non-threaded
default) into two independently-paced loops: HPGD is what actually discovers
new, diverse states and is cheap (hpgd_num_steps gradient steps); PGD is what
tries to turn a candidate into a genuine violation and is expensive
(bi_attack_pgd_steps steps, e.g. 50). Yoking them 1:1 bottlenecks
candidate-discovery throughput on attack throughput even though the two are
logically independent.

GCE is explicitly out of scope here -- it keeps running in ACTFuzzer.fuzz()'s
main thread either way, sharing the same PatternStateManager/seed_corpus
(now lock-guarded via ACTFuzzer._state_lock / SeedCorpus's own lock so that's
safe with these two threads running concurrently).

Copyright (C) 2025 SVF-tools/ACT
License: AGPLv3+
"""

from __future__ import annotations

import queue
import random
import threading
from typing import TYPE_CHECKING, Tuple

import torch
import torch.nn as nn

from act.pipeline.fuzzing.mutations import _relu_sign_pattern_batched

if TYPE_CHECKING:
    from act.pipeline.fuzzing.actfuzzer import ACTFuzzer
    from act.pipeline.fuzzing.corpus import FuzzingSeed

# Queue item: (seeds, hpgd_out, natural_pattern) -- natural_pattern is the
# ORIGINAL seed's pre-HPGD pattern (needed by checkpoint 2's _observe_state
# diff, same as _fuzz_iteration's non-threaded path uses), not hpgd_out's own.
_QueueItem = Tuple["FuzzingSeed", torch.Tensor, torch.Tensor]

_POLL_TIMEOUT = 0.1  # seconds; matches act.pipeline.fuzzing.trace_storage's poll idiom


def hpgd_producer_loop(
    fuzzer: "ACTFuzzer",
    stop_event: threading.Event,
    candidate_queue: "queue.Queue[_QueueItem]",
    batch_size: int,
    hpgd_model: nn.Module,
) -> None:
    """Draws seeds and runs HPGD (stage 1, including checkpoint 1) in a
    tight loop, feeding candidate_queue. Never blocks on the queue being
    full -- drops the candidate instead (see module docstring / the
    approved plan's rationale: an occasional dropped candidate costs
    nothing, since checkpoint 1 already admitted it into
    state_manager/seed_corpus; only a *second*, PGD-reinforced look is
    lost).

    `hpgd_model` is this thread's OWN model instance (ACTFuzzer._bi_hpgd_model),
    never touched by pgd_consumer_loop or the main thread's GCE loop --
    required because _relu_sign_pattern_batched/HPGDMutation.mutate register
    temporary forward hooks directly on the model instance they're given, and
    two threads sharing one instance would corrupt each other's captured
    pre-activations."""
    from act.pipeline.fuzzing.corpus import FuzzingSeed

    while not stop_event.is_set():
        # Same seed-selection logic as ACTFuzzer._fuzz_iteration's non-threaded
        # path (annealed random-restart vs state_manager.pick_seeds()), keyed
        # off this thread's own iteration counter instead of a shared one.
        restart_prob = fuzzer.config.bi_random_restart_prob * (
            fuzzer.config.bi_random_restart_cooling_rate ** fuzzer.bi_hpgd_iterations
        )
        if fuzzer.config.scheduling_mode == "sparse":
            with fuzzer._state_lock:
                has_tree_state = fuzzer.state_manager is not None and len(fuzzer.state_manager) > 0
                roll_restart = (not has_tree_state) or (random.random() < restart_prob)
                if not roll_restart:
                    payloads = fuzzer.state_manager.pick_seeds(batch_size, use_energy=True)
                    seeds: "FuzzingSeed" = fuzzer.state_manager.seeds_to_batch(payloads)
            if roll_restart:
                # _sample_random_restart_seeds calls seed_corpus.select(),
                # which has its own lock -- deliberately outside
                # _state_lock here so the (slower) random-box sampling
                # doesn't hold the shared lock.
                seeds = fuzzer._sample_random_restart_seeds(batch_size)
        else:
            seeds = fuzzer.seed_corpus.select(batch_size)

        with torch.no_grad():
            natural_pattern = _relu_sign_pattern_batched(hpgd_model, seeds.tensor.to(fuzzer.device))

        hpgd_out, natural_pattern = fuzzer._bi_stage1_hpgd(seeds, natural_pattern, model=hpgd_model)

        try:
            candidate_queue.put_nowait((seeds, hpgd_out, natural_pattern))
        except queue.Full:
            fuzzer.bi_candidates_dropped += 1

        fuzzer.bi_hpgd_iterations += batch_size


def pgd_consumer_loop(
    fuzzer: "ACTFuzzer",
    stop_event: threading.Event,
    candidate_queue: "queue.Queue[_QueueItem]",
    pgd_model: nn.Module,
) -> None:
    """Pulls candidates off candidate_queue and runs the PGD-family attack
    (stage 2) + checkpoint 2 on each. Polls with a short timeout instead of
    blocking indefinitely so it notices stop_event promptly; no attempt to
    drain a queue backlog on shutdown (see module docstring).

    `pgd_model` is this thread's OWN model instance (ACTFuzzer._bi_pgd_model)
    -- see hpgd_producer_loop's docstring for why threads don't share one."""
    while not stop_event.is_set():
        try:
            seeds, hpgd_out, natural_pattern = candidate_queue.get(timeout=_POLL_TIMEOUT)
        except queue.Empty:
            continue

        stage2_out = fuzzer._bi_stage2_pgd(seeds, hpgd_out, model=pgd_model)
        source_label = f"bi_{fuzzer.config.bi_attack_strategy}"
        violation_mask, counterexamples = fuzzer._check_and_admit(
            seeds, stage2_out, natural_pattern, source_label=source_label, model=pgd_model,
        )
        if fuzzer.config.verbose >= 2:
            for ce in counterexamples:
                print(f"🚨 [{source_label}] Counterexample #{len(fuzzer.counterexamples)}: {ce.summary()}")

        fuzzer.bi_pgd_iterations += len(seeds)
