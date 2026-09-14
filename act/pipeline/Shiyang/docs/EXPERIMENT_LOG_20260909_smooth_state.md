# State admission on smooth activations (Sigmoid / Tanh) -- 2026-09-09

Question: does the pattern-novelty admission that ACT uses on ReLU nets do
anything for `distinct` and CE-diversity when the activation is smooth?

## Setup

Benchmark `eran_sigmoid_tanh_mlp` (ERAN MNIST MLPs via GenBaB; specs generated
locally, see its README). Two model groups, one run each = one batch:

* `ffnnSIGMOID__Point_6x100`, eps 0.015, rows 0-98, B=99
* `ffnnTANH__Point_6x100`, eps 0.006, rows 297-393, B=97

Three arms, everything else at published defaults:

| arm | flags |
|---|---|
| `baseline` | `--admission-mode coverage` |
| `statebase` | `--admission-mode state` |
| `statehpgd` | `--admission-mode state --hpgd-weight 0.5` |

60 s per group, CPU, `--no-save`, n=5 for distinct
(`results/_eran_state_ab_{sig,tanh}6x100`), plus one
`--dump-founder-clusters 250` run per arm for S (`results/div_eran_*`).

## distinct (n=5, mean +- pstdev, Welch vs baseline)

| arm | sigmoid 6x100 | tanh 6x100 |
|---|---|---|
| baseline | 1.6 +- 0.8 | 1.4 +- 0.5 |
| statebase | **4.6 +- 1.2**  (t=4.16, p=0.0043) | **2.2 +- 0.4**  (t=2.53, p=0.036) |
| statehpgd | 4.4 +- 0.8  (t=4.95, p=0.0011) | 1.8 +- 0.7  (t=0.89, p=0.40) |

Union over reps: sigmoid baseline {14,37,61} vs both state arms
{14,37,61,64,76,94}; tanh baseline {32,63} vs state {32,37,63}.

Violations move the other way, as on every ReLU benchmark: 58.8k -> 53.5k ->
41.4k on sigmoid. The paper's headline metric and distinct are anti-correlated
here too.

## S -- mean pairwise cosine distance of the CE constraint directions

| arm | sigmoid S(paired) | tanh S(paired) | tanh wins |
|---|---|---|---|
| baseline | 0.000 | 0.032 | 0/2 |
| statebase | 0.006 | 0.046 | 0/2 |
| statehpgd | **0.023** | **0.106** | **2/2** |

Same ordering as the four ReLU benchmarks: statehpgd is widest. Read the
sigmoid column with care -- the paired set is **1 instance** and the baseline
dump holds 2 counterexamples, i.e. a single pair. tanh's 2 paired instances
with 250 CEs per arm is the measurement that carries weight.

## Why it bites here when it does not on ReLU

Admission rate over the whole run (`state_novelty.admitted / observed`):

| net | statebase | statehpgd |
|---|---|---|
| sigmoid 6x100 | 0.66 % | 5.17 % |
| tanh 6x100 | 0.37 % | 1.70 % |

On ReLU the same filter admits 73-100 % (see the state-novelty-dimensionality
note): the unstable subspace is so high-dimensional that nothing is ever
revisited and novelty is constantly true. Smooth activations fail the opposite
way. Interval propagation calls nearly every deep neuron unstable -- the box
straddles the inflection point -- but only a few percent of those signs can
actually be moved from inside the box, so the restricted pattern is almost
constant and nearly everything reads as already-seen.

`state_novelty.admitted/observed` is NOT a rejection rate, and an earlier
version of this log wrongly read it as one. Counterexamples enter the corpus
whether or not their pattern is novel; that counter only reports how many
samples were flagged novel. On the counter both arms share, `corpus_drops`,
state rejects slightly LESS than the baseline:

| net | arm | offered | added | reject |
|---|---|---|---|---|
| sigmoid | baseline | 602,613 | 293,866 | 51.2 % |
| | statebase | 536,382 | 270,148 | 49.6 % |
| | statehpgd | 435,996 | 225,380 | 48.3 % |
| tanh | baseline | 559,690 | 303,989 | 45.7 % |
| | statebase | 518,950 | 290,672 | 44.0 % |
| | statehpgd | 419,525 | 279,558 | 33.4 % |

Split by whether the sample was a counterexample, the real difference appears.
The coverage gate admits counterexamples and **nothing else** -- `added` equals
`violations` to within the rep-boundary bookkeeping (+-161). State admits the
same counterexamples plus a thin stream of novel non-CE seeds: 2,708 of 268,942
non-CE offers on sigmoid statebase, 18,442 of 229,058 on statehpgd (99.0 % and
91.9 % non-CE rejection, against the baseline's 100 %).

So the gain cannot come from filtering harder. Two candidates remained --
(a) the novel non-CE seeds, (b) `state` also replacing coverage's ENERGY
formula with the density-aware one -- and the `always` / `state_always` arms
separate them by forcing the gate open under each energy formula (n=5 each):

| net | arm | gate / energy | distinct | non-CE admitted | top-instance draws |
|---|---|---|---|---|---|
| sigmoid | baseline | cov / cov | 1.6 +- 0.8 | 0.0 % | 92.6 % |
| | `always` | OPEN / cov | 2.4 +- 0.8 (p=0.20) | 99.7 % | 94.3 % |
| | statebase | state / state | 4.6 +- 1.2 (p=0.0043) | 1.0 % | 81.6 % |
| | `state_always` | OPEN / state | 4.8 +- 1.0 (p=0.0011) | 99.7 % | 78.7 % |
| tanh | baseline | cov / cov | 1.4 +- 0.5 | 0.0 % | 95.0 % |
| | `always` | OPEN / cov | 1.0 +- 0.0 (p=0.18) | 99.9 % | 99.3 % |
| | statebase | state / state | 2.2 +- 0.4 (p=0.036) | 0.6 % | 81.1 % |
| | `state_always` | OPEN / state | 2.0 +- 0.6 (p=0.17) | 99.7 % | 98.2 % |

**(a) is refuted and the admission gate is not the mechanism.** Opening the
gate under coverage's energy moves nothing (sigmoid p=0.20; tanh goes DOWN).
Under state's energy the gate can be removed entirely without cost --
statebase 4.6 vs state_always 4.8 are indistinguishable, both p ~ 0.001-0.004
against the baseline. The top-instance draw share tracks the energy formula
and not the gate: 92.6/94.3 % under coverage energy, 81.6/78.7 % under state
energy.

So on smooth activations the whole effect belongs to the density-aware ENERGY
formula that `--admission-mode state` swaps in as a side effect. It must not be
reported as "state admission improves distinct on Sigmoid/Tanh" -- that credits
the wrong component, and it is the same conclusion the ReLU work reached: sign-
pattern novelty has no useful correlation with cracking an instance; scheduling
does. (tanh's `state_always` falling back to p=0.17 with a 98.2 % draw share
says the gate contributes something there, but n=5 and the effect size do not
support a separate claim.)

## What the coverage gate actually does, and when it dies

`coverage_strategy="GlobalCov"` (pipeline.yaml, and hard-set at
paper_cifar100_batch_ani.py:327). AFL-style monotone union: a sample is
interesting iff it fires a neuron never fired before, and admission is
`interesting_mask = violation_mask | cov_interesting` (actfuzzer.py:1214). A
monotone union can only saturate. Worse, `coverage_per_instance` defaults to
False, so the union is over all 99 lanes at once -- GlobalCov's own comment
notes that one instance firing a neuron raises the bar for the other 98 for the
rest of the run.

Measured with `--report-interval 20` (`results/_eran_cov_curve/`):

| net | first report | GlobalCov there | final GlobalCov | never-activated | non-CE admissions, whole run |
|---|---|---|---|---|---|
| sigmoid 6x100 | iter 99 (1 batch, ~0.05 s) | **98.03 %** | 98.03 % | 12 | 8, all in the first batch |
| tanh 6x100 | iter 97 (1 batch) | **99.34 %** | 99.34 % | 4 | **0, ever** |

Coverage is already at its final value at the end of the FIRST batch and never
rises again over 122,265 / 103,984 iterations of 60 s. Seeds stay at 107 (99
initial + 8) and 97 (initial only) until the first counterexample arrives
(TTFV 0.449 s / 0.160 s). For 99.9 % of the run the baseline's gate is a
constant False and the corpus is fed by counterexamples alone -- which is why
its draws collapse onto the one or two already-broken lanes.
Whatever the split, the honest reading stays: HPGD on top adds nothing
(sigmoid) or costs (tanh) for distinct while still buying the widest S, exactly
the split the ReLU campaigns show; and nothing here shows the sign pattern
carries meaning on a Sigmoid.

## Caveats

* One model group per activation. 6x200 and 9x100 are untested.
* No ground truth. This is not a VNN-COMP category and GenBaB's Table 2 reports
  only verified counts (sigmoid 6x100: 71/100, tanh 6x100: 65/100), so
  SAT + timeout <= 29 and <= 35 respectively -- an upper bound, no per-instance
  labels. The distinct ceiling is unknown until an abcrown pass is run.

---

# Three-bin state on smooth activations -- 2026-09-10

Question: split the state at `z = -tau, +tau` instead of at `z = 0`, and can
HPGD still move the code?

Implementation: `act/pipeline/fuzzing/state_bins.py`. Three segments = two
walls, encoded as TWO +-1 coordinates per neuron (`sign(z+tau)`, `sign(z-tau)`)
so the BK-tree, Bloom filter, fingerprint packing and marginals are untouched;
the two coordinates of a neuron are adjacent, so one bin of movement is Hamming
distance 1. HPGD's hinge becomes `target * (z - tau)` -- the same projection at
a different wall. Targets are projected onto the reachable code set before use
(`(-1,+1)` asks for `z < -tau` and `z > +tau` at once). `--state-bins {2,3}`,
`--state-bin-tau`, default 2 = the old `sign(z)` exactly (verified: the mask
reports 355/600 on sigmoid under both the old and new code).

**Three segments has no wall at 0.** It is a different partition, not a
refinement. On these nets that costs 2 of 600 neurons.

## Does HPGD still transfer the code? Yes, and it is the only strategy that does

Share of samples whose code was novel, by strategy, summed over 5 reps:

| net | strategy | 2-bin | 3-bin |
|---|---|---|---|
| sigmoid | **hpgd** | **11.3 %** | **29.9 %** |
| | pgd | 0.9 % | 2.7 % |
| | random | 2.3 % | 7.8 % |
| tanh | **hpgd** | **5.5 %** | **15.2 %** |
| | pgd | 0.5 % | 1.0 % |
| | random | 1.0 % | 2.6 % |

HPGD produces a novel code 12x (sigmoid) / 11x (tanh) more often than plain
PGD, and the three-bin split roughly triples its rate. Counterexamples that are
also novel states -- the ones worth having -- go 3,508 -> 7,586 (sigmoid) and
2,353 -> 4,719 (tanh). So the hinge projection drives smooth neurons across a
+-tau wall as readily as it drives ReLU neurons across zero.

## distinct (n=5, mean +- pstdev, Welch within each pair)

| net | arm | 2-bin | 3-bin | p |
|---|---|---|---|---|
| sigmoid | statebase | 3.6 +- 0.5 | **4.8 +- 0.7** | **0.032** |
| | statehpgd | 4.4 +- 1.2 | 4.4 +- 1.0 | 1.00 |
| tanh | statebase | 1.6 +- 0.5 | 2.2 +- 0.4 | 0.096 |
| | statehpgd | 1.6 +- 0.5 | **2.6 +- 0.5** | **0.020** |

Three of four pairs rise, two significantly, none falls. The top instance's
draw share falls with it (sigmoid statebase 93.1 -> 89.9 %, tanh statehpgd
96.2 -> 92.2 %), which is the same scheduling channel the admission-vs-energy
isolation identified -- a finer code makes the density-aware energy formula
discriminate where the two-bin code had collapsed.

Note the one flat cell: on sigmoid, statehpgd is already at 4.4 with the
two-bin code and three bins does not add. Only statebase gains there. The two
arms converge to the same 4.4-4.8 either way, which reads as a ceiling of this
budget rather than as HPGD failing.

## S: not measurable on this run

Paired instances: **1** on sigmoid (and `statebase_b3`'s founder dump holds 5
counterexamples, i.e. 10 pairs), **0** on tanh. The four arms cannot be ranked
on S here. Fixing it needs more budget or more reps, not more analysis -- the
same collapse the portfolio three-arm runs hit on mnist/cifar.

## The limitation that matters most

Under IBP the three-bin candidate set is **exactly** twice the two-bin one
(sigmoid 710/1200 vs 355/600), because the propagated intervals are wide enough
that all of -tau, 0, +tau fall inside the same neurons' bounds. So the extra
walls buy nothing *in the candidate set*; the gains above come from the finer
code alone (admission, energy coordinates, hinge target).

The reachability measured with the box's first-order gradient budget -- what
the attack actually pays -- says the walls are worth much more than IBP can
see: sigmoid 24/600 neurons can reach the sign wall against 54/600 for +-tau,
tanh 20/600 against 40/600. **The next lever is therefore the candidate
criterion, not tau**: replacing IBP with the gradient budget would let the
candidate set express the 2x difference the state code already exploits.
