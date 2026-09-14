"""
Batched seed corpus for GPU-accelerated fuzzing.

FuzzingSeed — every field carries a leading batch dim B (no scalar path).
SeedCorpus  — parallel-tensor pool.
  Selection: energy-weighted sampling with replacement → FuzzingSeed(B).
  Insertion: boolean-mask filtered add() with per-spec-row byte-hash dedup.
  Storage:   N parallel 1-D/N-D tensors grown via torch.cat on insert.
  Retirement: rows are never deleted -- an _alive mask hides them from
              select()/__iter__/__len__ (see add()'s ce_mask).

Copyright (C) 2025 SVF-tools/ACT
License: AGPLv3+
"""

from __future__ import annotations
from typing import Optional, Union
import numpy as np
import torch

from act.front_end.spec_creator_base import LabeledInputTensor
from act.util.device_manager import get_default_device


class FuzzingSeed:
    """
    Batched fuzzing seed. All fields are tensors with leading batch dim B.
    A single FuzzingSeed instance represents B seeds. 
    
    Attributes:
        tensor:          [B, ...] Input tensors (current, may be mutated)
        original_tensor: [B, ...] Original clean inputs (NEVER mutated)
        original_index:  [B] int64 — synthesized spec-row index
        label:           [B] int64 — ground truth label (-1 = no label)
        energy:          [B] float — seed energy (higher = more interesting)
        depth:           [B] int64 — how many mutations from original seed
        id:              [B] int64 — unique seed identifiers (auto-generated)
        parent_id:       [B] int64 — parent seed IDs (-1 = no parent)
        select_count:    [B] int64 — how many times this seed has been selected
    """
    
    _id_counter: int = 0
    
    @classmethod
    def _next_ids(cls, n: int) -> torch.Tensor:
        """Generate n unique sequential int64 IDs."""
        start = cls._id_counter
        cls._id_counter += n
        return torch.arange(start, start + n, dtype=torch.long)
    
    def __init__(
        self,
        tensor: torch.Tensor,
        original_tensor: Optional[torch.Tensor] = None,
        original_index: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
        energy: Optional[torch.Tensor] = None,
        depth: Optional[torch.Tensor] = None,
        id: Optional[torch.Tensor] = None,
        parent_id: Optional[torch.Tensor] = None,
        select_count: Optional[torch.Tensor] = None,
    ):
        B = tensor.shape[0]
        self.tensor = tensor
        self.original_tensor = original_tensor if original_tensor is not None else tensor.clone()
        self.original_index = original_index if original_index is not None else torch.zeros(B, dtype=torch.long)
        self.label = label if label is not None else torch.full((B,), -1, dtype=torch.long)
        self.energy = energy if energy is not None else torch.ones(B)
        self.depth = depth if depth is not None else torch.zeros(B, dtype=torch.long)
        self.id = id if id is not None else FuzzingSeed._next_ids(B)
        self.parent_id = parent_id if parent_id is not None else torch.full((B,), -1, dtype=torch.long)
        self.select_count = select_count if select_count is not None else torch.zeros(B, dtype=torch.long)
    
    def __len__(self) -> int:
        """Return batch size B."""
        return self.tensor.shape[0]
    
    def __getitem__(self, idx: Union[int, slice, torch.Tensor]) -> 'FuzzingSeed':
        """Slice the batch. Returns FuzzingSeed with sub-batch."""
        if isinstance(idx, int):
            idx = slice(idx, idx + 1)
        return FuzzingSeed(
            tensor=self.tensor[idx],
            original_tensor=self.original_tensor[idx],
            original_index=self.original_index[idx],
            label=self.label[idx],
            energy=self.energy[idx],
            depth=self.depth[idx],
            id=self.id[idx],
            parent_id=self.parent_id[idx],
            select_count=self.select_count[idx],
        )


class SeedCorpus:
    """
    AFL-style seed corpus with energy-based scheduling.
    
    Features:
    - Energy-based seed selection (higher energy = more likely to select)
    - Random selection as fallback
    - Automatic seed deduplication by spec row and tensor hash
    - All storage on device_manager's device (GPU when available)
    
    Example:
        >>> corpus = SeedCorpus(initial_seeds, strategy="energy")
        >>> batch = corpus.select(32)       # FuzzingSeed with B=32
        >>> corpus.add(child, mask)   # Add interesting children
    """
    
    def __init__(self, 
                 initial_seeds: list[LabeledInputTensor],
                 strategy: str = "energy"):
        """
        Initialize seed corpus from LabeledInputTensor list.
        
        Converts list of LabeledInputTensor into parallel tensor storage.
        
        Args:
            initial_seeds: Initial seeds from spec creators
            strategy: Selection strategy ("energy" or "random")
        """
        self.strategy = strategy
        self.seen_hashes: set[tuple[int, int]] = set()
        self._device = get_default_device()
        
        # Build parallel lists from initial seeds, then stack to tensors
        tensors = []
        indices = []
        labels = []

        for i, labeled_tensor in enumerate(initial_seeds):
            t = labeled_tensor.tensor.to(self._device)
            label_val = int(labeled_tensor.label.item()) if isinstance(labeled_tensor.label, torch.Tensor) else labeled_tensor.label
            
            seed_hash = (i, self._hash_tensor(t))
            self.seen_hashes.add(seed_hash)
            
            tensors.append(t)
            indices.append(i)
            labels.append(label_val if label_val is not None else -1)
        
        N = len(tensors)
        device = self._device
        # Stack into parallel tensors on device manager's device
        self._tensors = torch.cat(tensors, dim=0)                                       # [N, ...]
        self._original_tensors = self._tensors.clone()                                  # [N, ...]
        self._original_indices = torch.tensor(indices, dtype=torch.long, device=device) # [N]
        self._labels = torch.tensor(labels, dtype=torch.long, device=device)            # [N]
        self._energies = torch.ones(N, device=device)                                   # [N]
        self._depths = torch.zeros(N, dtype=torch.long, device=device)                  # [N]
        self._ids = FuzzingSeed._next_ids(N)                                            # [N]
        self._parent_ids = torch.full((N,), -1, dtype=torch.long, device=device)        # [N]
        self._select_counts = torch.zeros(N, dtype=torch.long, device=device)           # [N]
        # Retirement bookkeeping (CE-parent replacement, see add()). Rows are
        # masked out rather than spliced away: rebuilding nine parallel tensors
        # on every removal would cost more than the pool it is pruning.
        self._alive = torch.ones(N, dtype=torch.bool, device=device)                    # [N]
        # Which rows are counterexamples. Recorded rather than inferred from
        # energy: energies are additive (`admitted*10 + violation*ce_energy`),
        # so with a ce_energy below the admitted bonus the CE tier sits BELOW
        # ordinary novel seeds and no single threshold can name it. The initial
        # seeds are the unmutated spec rows, none of which violates.
        self._is_ce = torch.zeros(N, dtype=torch.bool, device=device)                   # [N]
        # A parent is named by its seed id; ids are global and never reused,
        # so a dict is all that is needed to get from a child's parent_id to
        # the parent's row.
        self._id_to_row: dict[int, int] = {
            int(seed_id): row for row, seed_id in enumerate(self._ids.tolist())
        }
        self._retired = 0
        # Why a child did not become a row. The reported rejection rate
        # `1 - (rows - B)/iterations` sums three unrelated filters, and the
        # third exists only on cerepl arms and scales with counterexample
        # yield -- measured, coverage admission alone kept 53,841 rows where
        # the same admission plus replacement kept 21,386. Counting them
        # apart is what makes the number comparable across arms. Pure
        # bookkeeping: no decision and no RNG draw reads these.
        #   offered    children handed to add() (= mutated samples)
        #   gate       mask False: neither a violation nor judged interesting
        #   dedup      this exact input tensor is already a row for this lane
        #   ce_sibling CE child refused because a sibling already took the
        #              parent's slot this batch (cerepl only)
        #   added      rows actually appended
        self._drops = {"offered": 0, "gate": 0, "dedup": 0,
                       "ce_sibling": 0, "added": 0}
        # How many select() draws each INSTANCE's lineage has received. The
        # per-row `_select_counts` cannot answer this: a child inherits its
        # parent's count at add() time, so summing rows double-counts. This
        # accumulates the drawn rows' original_index directly, which is the
        # quantity the whole scheduling argument is about -- an instance whose
        # lineage is never drawn can never be broken, however long the run.
        self._draws_by_instance = torch.zeros(
            int(self._original_indices.max().item()) + 1 if N else 0,
            dtype=torch.long, device=device)

    def draws_by_instance(self) -> list:
        """Draws per instance, indexed by original_index. Retired lineages keep
        the draws they already received, which is correct: the question is how
        the budget WAS spent, not how it would be spent now."""
        return self._draws_by_instance.tolist()

    @property
    def drop_stats(self) -> dict:
        """Counts of why children did not become rows; see `_drops`."""
        return dict(self._drops)

    def _hash_tensor(self, tensor: torch.Tensor) -> int:
        """Compute hash of tensor for deduplication."""
        return hash(tensor.flatten().cpu().numpy().tobytes())
    
    def select(self, n: int, replace: bool = True,
               per_instance: bool = False) -> FuzzingSeed:
        """
        Select n seeds based on strategy. Returns a FuzzingSeed batch of size n.

        Sampling is with replacement — high-energy seeds may appear multiple times,
        which is intentional for exploitation.

        `replace=False` draws n DISTINCT rows instead, still energy-weighted
        (Efraimidis-Spirakis via Gumbel top-k, which is equivalent to numpy's
        weighted choice without replacement but linear rather than quadratic in
        the pool). It changes only who occupies the B lanes, never the pool.

        Why it is worth a flag. With replacement, a batch of B draws over a pool
        whose energy is concentrated on a few counterexample seeds puts those
        seeds on most lanes: measured on tinyimagenet at B=200, ~12 CE seeds
        held 80% of the mass and took ~161 of the 200 lanes, while the 200
        per-instance initial seeds together held 0.01% of the mass and were
        drawn about 3 times per 18,000-sample run. Even at iteration 1, when
        every energy is still 1.0, drawing B from B with replacement reaches
        only B(1-1/e) = 63% of the instances -- 37% are skipped before any
        energy imbalance exists. Without replacement that first batch is a
        permutation: every instance gets exactly one lane.

        Falls back to replacement when the pool is smaller than n, which cannot
        be satisfied otherwise (CE-parent replacement can prune the live pool
        below B).

        `per_instance=True` is the instance-level version of the same idea, and
        the one that actually fixes starvation: round-robin over instances,
        energy-weighted only within each. `replace=False` was not enough,
        because an instance owns thousands of rows and distinct rows can all
        belong to it -- measured on mnist it left the allocation untouched
        (Gini 0.77 -> 0.76, Simpson effective instances 4.7 -> 4.7). Here every
        instance contributes its best row before any contributes a second.

        Args:
            n: Number of seeds to select (required).
            replace: draw with replacement (default, the shipped behaviour).
            per_instance: round-robin over instances (overrides `replace`).

        Returns:
            FuzzingSeed batch with B=n.
        """
        corpus_size = self._tensors.shape[0]
        if corpus_size == 0:
            raise ValueError("Corpus is empty!")

        # Nothing retired => every row is live and this is the original draw,
        # RNG call for RNG call (np.random.choice consumes the same stream for
        # an int population as for the equivalent index array).
        if self._retired == 0:
            live_rows = None
            pool = corpus_size
        else:
            live_rows = self._alive.nonzero(as_tuple=True)[0].cpu().numpy()
            pool = int(live_rows.size)
            if pool == 0:
                raise ValueError("Corpus has no live seeds -- all were retired!")

        # A pool smaller than the batch cannot supply n distinct rows.
        without = (not replace) and pool >= n
        population = pool if live_rows is None else live_rows

        if per_instance and self.strategy == "energy":
            # Round-robin over INSTANCES, energy-weighted only WITHIN each one.
            # `replace=False` was not enough for fairness: it draws distinct
            # ROWS, and one instance owns thousands of them, so it still takes
            # many lanes (measured on mnist: Gini 0.77 -> 0.76, unchanged).
            # Here every instance contributes its best row before any instance
            # contributes a second, so each gets floor(B/I) or ceil(B/I) lanes.
            # With B == I -- mnist's 30 specs on 30 lanes -- every batch is a
            # permutation of the instances and starvation is impossible by
            # construction.
            energies = self._energies.cpu().numpy()
            if live_rows is not None:
                energies = energies[live_rows]
            inst = self._original_indices.cpu().numpy()
            if live_rows is not None:
                inst = inst[live_rows]
            keys = np.log(np.maximum(energies, 1e-300)) + np.random.gumbel(size=pool)
            # Rank each row within its own instance, best key first.
            order = np.lexsort((-keys, inst))
            sorted_inst = inst[order]
            starts = np.concatenate(
                ([0], np.flatnonzero(sorted_inst[1:] != sorted_inst[:-1]) + 1))
            group_len = np.diff(np.concatenate((starts, [pool])))
            rank = np.empty(pool, dtype=np.int64)
            rank[order] = np.arange(pool) - np.repeat(starts, group_len)
            # Rank ascending first (one per instance, then two, ...), key
            # descending within a rank.
            take = np.lexsort((-keys, rank))[:n]
            idx = torch.from_numpy(take if live_rows is None else live_rows[take]).long()
            idx = idx.to(self._device)
            result = self._gather(idx)
            self._note_draw(idx)
            return result

        if self.strategy == "energy":
            energies = self._energies.cpu().numpy()  # numpy requires CPU
            if live_rows is not None:
                energies = energies[live_rows]
            total = energies.sum()
            if total == 0:
                probs = np.ones(pool) / pool
            else:
                probs = energies / total
            if without:
                # Gumbel top-k == Efraimidis-Spirakis weighted sampling without
                # replacement, and linear in the pool where numpy's
                # choice(replace=False, p=...) is not.
                keys = np.log(np.maximum(probs, 1e-300)) + np.random.gumbel(size=pool)
                take = np.argpartition(-keys, n - 1)[:n]
                indices = take if live_rows is None else live_rows[take]
            else:
                indices = np.random.choice(population, size=n, p=probs, replace=True)
        else:
            indices = np.random.choice(population, size=n, replace=not without)
        
        idx = torch.from_numpy(indices).long().to(self._device)
        result = self._gather(idx)
        self._note_draw(idx)
        return result

    def _gather(self, idx: torch.Tensor) -> FuzzingSeed:
        """Rows `idx` as a FuzzingSeed batch."""
        return FuzzingSeed(
            tensor=self._tensors[idx],
            original_tensor=self._original_tensors[idx],
            original_index=self._original_indices[idx],
            label=self._labels[idx],
            energy=self._energies[idx],
            depth=self._depths[idx],
            id=self._ids[idx],
            parent_id=self._parent_ids[idx],
            select_count=self._select_counts[idx],
        )

    def _note_draw(self, idx: torch.Tensor) -> None:
        """Book-keeping every select() path must do: per-row and per-instance
        draw counts. Kept in one place so a new selection mode cannot silently
        skip it and make the fairness measurements wrong."""
        self._select_counts.scatter_add_(0, idx, torch.ones_like(idx))
        inst = self._original_indices[idx]
        self._draws_by_instance.scatter_add_(0, inst, torch.ones_like(inst))
    
    def add(
        self,
        seeds: FuzzingSeed,
        mask: torch.Tensor,
        ce_mask: Optional[torch.Tensor] = None,
        ce_parent_energy_threshold: float = 100.0,
        replace_ce_parents: Optional[bool] = None,
    ):
        """
        Add interesting seeds from a batch to the corpus.

        Only seeds where mask[i] == True are added, with dedup checking.

        CE-parent replacement (ce_mask, default off): energies here are
        `admitted*10 + violation*100` clamped to 0.1, so the pool holds only
        four distinct values and the counterexample tier outweighs everything
        else by 1000:1. Since nothing ever prunes, one cracked instance's CE
        lineage compounds until it owns nearly all of select()'s probability
        mass (measured: 95%, from 3 instances), and a starved instance's lone
        0.1-energy seed is never drawn again. Passing ce_mask makes a CE child
        whose parent is ALREADY in the CE tier *replace* that parent instead
        of joining it: the tier stops compounding, while seeds below the
        threshold -- new and uncracked instances -- keep their slots untouched.

        Args:
            seeds: FuzzingSeed batch (all children from one iteration)
            mask: BoolTensor[B] indicating which seeds are interesting
            ce_mask: BoolTensor[B] marking which children are counterexamples.
                Recorded on every row it is given for (see `_is_ce`), which is
                what `ce_count`/`ce_mass` report and what the parent test
                below asks. None (default) records nothing and, unless
                `replace_ce_parents` says otherwise, disables replacement,
                leaving the append-only behaviour byte-for-byte as it was.
            replace_ce_parents: whether a CE child retires its CE parent.
                Defaults to "yes if ce_mask was given", the original meaning;
                pass False to record the flags without retiring anything.
            ce_parent_energy_threshold: superseded by the recorded CE flag and
                no longer consulted. Kept so existing callers still work; with
                the shipped ce_energy of 100 the two agree exactly, because
                only a violation could put a row at or above 100.
        """
        # Counted before the early return, or a batch that offered nothing
        # would not show up as gate rejections at all.
        self._drops["offered"] += int(mask.numel())
        self._drops["gate"] += int(mask.numel() - int(mask.sum()))
        if not mask.any():
            return

        # Filter to interesting seeds
        interesting_idx = mask.nonzero(as_tuple=True)[0]

        # One host copy each rather than a per-child device sync; the dedup
        # hash below already pulls every candidate tensor to CPU, so this is
        # not the expensive part of the loop.
        marking = ce_mask is not None
        replacing = marking and (True if replace_ce_parents is None
                                 else bool(replace_ce_parents))
        if marking:
            ce_flags = ce_mask.cpu().tolist()
        if replacing:
            parent_ids = seeds.parent_id.cpu().tolist()
            # Corpus-sized transfers, unlike the two batch-sized ones above --
            # only the retirement path needs them.
            alive_cpu = self._alive.cpu()
            is_ce_cpu = self._is_ce.cpu()

        # Dedup and collect indices to actually add
        keep = []
        retire_rows: list[int] = []
        retired_ids: set[int] = set()
        for i in interesting_idx.tolist():
            original_index = int(seeds.original_index[i].item())
            seed_hash = (original_index, self._hash_tensor(seeds.tensor[i:i+1]))
            if seed_hash in self.seen_hashes:
                self._drops["dedup"] += 1
                continue
            if replacing and ce_flags[i]:
                parent_id = int(parent_ids[i])
                row = self._id_to_row.get(parent_id)
                if (row is not None
                        and bool(alive_cpu[row])
                        and bool(is_ce_cpu[row])):
                    # select() samples WITH REPLACEMENT, so one parent can hold
                    # several lanes and come back with several CE children in
                    # the same batch. The replacement is one-for-one per
                    # parent: the first child takes the parent's slot, the rest
                    # are dropped. Appending them all would retire one row and
                    # add N -- the very growth this exists to stop. Only the
                    # corpus slot is refused; the caller has already recorded
                    # their counterexamples.
                    if parent_id in retired_ids:
                        self._drops["ce_sibling"] += 1
                        continue
                    retired_ids.add(parent_id)
                    retire_rows.append(row)
            self.seen_hashes.add(seed_hash)
            keep.append(i)

        if retire_rows:
            rows = torch.tensor(retire_rows, dtype=torch.long, device=self._device)
            self._alive[rows] = False
            self._retired += len(retire_rows)

        if not keep:
            return

        self._drops["added"] += len(keep)
        idx = torch.tensor(keep, dtype=torch.long, device=self._device)
        base = self._tensors.shape[0]
        self._tensors = torch.cat([self._tensors, seeds.tensor[idx]], dim=0)
        self._original_tensors = torch.cat([self._original_tensors, seeds.original_tensor[idx]], dim=0)
        self._original_indices = torch.cat([self._original_indices, seeds.original_index[idx]])
        self._labels = torch.cat([self._labels, seeds.label[idx]])
        self._energies = torch.cat([self._energies, seeds.energy[idx]])
        self._depths = torch.cat([self._depths, seeds.depth[idx]])
        self._ids = torch.cat([self._ids, seeds.id[idx]])
        self._parent_ids = torch.cat([self._parent_ids, seeds.parent_id[idx]])
        self._select_counts = torch.cat([self._select_counts, seeds.select_count[idx]])
        self._alive = torch.cat([
            self._alive,
            torch.ones(len(keep), dtype=torch.bool, device=self._device),
        ])
        self._is_ce = torch.cat([
            self._is_ce,
            torch.tensor([bool(ce_flags[i]) for i in keep] if marking
                         else [False] * len(keep),
                         dtype=torch.bool, device=self._device),
        ])
        for offset, i in enumerate(keep):
            self._id_to_row[int(seeds.id[i].item())] = base + offset
    
    def __len__(self) -> int:
        """Number of LIVE seeds -- the pool select() actually draws from.
        Equals `rows` until CE-parent replacement retires something."""
        if self._retired == 0:
            return self._tensors.shape[0]
        return int(self._alive.sum().item())

    @property
    def rows(self) -> int:
        """Every row ever added, retired ones included. Use this, not len(),
        for "how many seeds did this run explore" -- retirement must not make
        an arm look like it explored less than it did."""
        return self._tensors.shape[0]

    @property
    def retired(self) -> int:
        """How many rows CE-parent replacement has taken out of circulation."""
        return self._retired

    def ce_count(self) -> int:
        """How many LIVE seeds are counterexamples, by the recorded flag rather
        than by an energy threshold. Equals count_above(100) under the shipped
        ce_energy of 100, and is the only correct answer below it -- with
        ce_energy 1 the CE tier sits at 1, under an admitted seed's 10, so
        every threshold either misses it or sweeps the admitted tier in.
        Zero unless the caller passes add()'s ce_mask."""
        flags = self._is_ce[self._alive] if self._retired else self._is_ce
        return int(flags.sum())

    def ce_mass(self) -> float:
        """Share of select()'s probability mass held by live counterexample
        seeds -- how much of the draw the CE tier owns. See ce_count for why
        this is asked of the flag and not of the energies."""
        alive = self._alive if self._retired else None
        energies = self._energies[alive] if alive is not None else self._energies
        flags = self._is_ce[alive] if alive is not None else self._is_ce
        total = float(energies.sum())
        if total == 0:
            return 0.0
        return float(energies[flags].sum()) / total

    def count_above(self, threshold: float = 100.0) -> int:
        """How many LIVE seeds sit at or above `threshold` energy. The count
        answers a different question from energy_mass_above: mass is the share
        of the DRAW the tier takes, count is the share of the POOL it occupies.
        With CE seeds weighted 100 against an admitted seed's 10, a tier can
        own a fifth of the draw while being a fortieth of the pool -- which is
        what "the corpus is not saturated with counterexamples yet" means."""
        energies = self._energies[self._alive] if self._retired else self._energies
        return int((energies >= threshold).sum())

    def ce_lineage_by_instance(self) -> dict:
        """Per instance, how many INDEPENDENT breakthroughs into the CE tier.

        A founder is a counterexample whose parent is NOT one: the single
        mutation that carried a lineage from an ordinary seed into the CE tier.
        Everything downstream of it is that same discovery being elaborated, so
        founders -- not rows, and not immediate parents -- is the count of
        genuinely separate finds. Counting immediate parents instead inflates
        the number by the chain length, because each generation of one lineage
        contributes a distinct parent id.

        Read over every row ever admitted, alive and retired alike: retirement
        masks rows out of select() but leaves the parallel tensors intact, so
        the run's whole genealogy is still here. A live-only reading would drop
        the rows CE-parent replacement retires and undercount the very arms it
        exists to measure. A retired founder still founded its lineage.

        A CE whose parent id is -1, or whose parent has been dropped from
        `_id_to_row`, counts as its own founder: the chain cannot be walked
        past it, and it is by definition the earliest CE reachable there.

        Returns:
            {instance_index: {"ce_rows", "founders", "per_founder",
            "largest_lineage", "max_depth", "median_depth"}} for instances with
            at least one counterexample. `per_founder` is ce_rows/founders --
            1.0 means every counterexample was its own breakthrough, 100.0 means
            each breakthrough was elaborated into a hundred descendants.
        """
        ce = self._is_ce
        if not bool(ce.any()):
            return {}
        is_ce = ce.tolist()
        parents = self._parent_ids.tolist()
        insts = self._original_indices.tolist()
        depths = self._depths.tolist()

        founder_of: dict[int, int] = {}

        def founder(row: int) -> int:
            """Topmost CE ancestor of `row`, memoised; iterative to stay safe
            on chains that reach depth 20+."""
            chain = []
            cur = row
            while cur not in founder_of:
                prow = self._id_to_row.get(int(parents[cur]))
                if prow is None or not is_ce[prow]:
                    founder_of[cur] = cur
                    break
                chain.append(cur)
                cur = prow
            top = founder_of[cur]
            for node in chain:
                founder_of[node] = top
            return top

        by: dict[int, dict] = {}
        for row in range(len(is_ce)):
            if not is_ce[row]:
                continue
            b = by.setdefault(insts[row], {"ce_rows": 0, "_f": {}, "_d": []})
            b["ce_rows"] += 1
            f = founder(row)
            b["_f"][f] = b["_f"].get(f, 0) + 1
            b["_d"].append(depths[row])

        out = {}
        for i, b in by.items():
            d = sorted(b["_d"])
            n = len(b["_f"])
            out[i] = {"ce_rows": b["ce_rows"], "founders": n,
                      "per_founder": round(b["ce_rows"] / n, 2),
                      "largest_lineage": max(b["_f"].values()),
                      "max_depth": d[-1], "median_depth": d[len(d) // 2]}
        return out

    def ce_founder_rows(self) -> dict:
        """{instance: {founder_row: [rows in that lineage]}} over every CE row.

        The row-level companion to :meth:`ce_lineage_by_instance`, which reports
        only counts. Exposed so a caller can ask what a lineage actually FOUND
        -- pull `_tensors[rows]` and compare the clusters -- rather than just
        how many there were. Same founder rule and same alive-and-retired
        coverage as that method; see it for both.
        """
        ce = self._is_ce
        if not bool(ce.any()):
            return {}
        is_ce = ce.tolist()
        parents = self._parent_ids.tolist()
        insts = self._original_indices.tolist()
        founder_of: dict[int, int] = {}

        def founder(row: int) -> int:
            chain = []
            cur = row
            while cur not in founder_of:
                prow = self._id_to_row.get(int(parents[cur]))
                if prow is None or not is_ce[prow]:
                    founder_of[cur] = cur
                    break
                chain.append(cur)
                cur = prow
            top = founder_of[cur]
            for node in chain:
                founder_of[node] = top
            return top

        out: dict = {}
        for row in range(len(is_ce)):
            if not is_ce[row]:
                continue
            out.setdefault(insts[row], {}).setdefault(founder(row), []).append(row)
        return out

    def row_tensors(self, rows):
        """Input tensors for `rows`, alive or retired. Retirement only masks."""
        import torch as _torch
        return self._tensors[_torch.as_tensor(rows, dtype=_torch.long)]

    def energy_mass_above(self, threshold: float = 100.0) -> float:
        """Share of select()'s probability mass held by live seeds at or above
        `threshold` energy -- i.e. how much of the draw one tier owns. This is
        the number CE-parent replacement exists to move: measured at 0.95 for
        the CE tier on an unpruned 60s run, from three cracked instances.
        Meaningless under strategy="random", which ignores energy."""
        energies = self._energies[self._alive] if self._retired else self._energies
        total = float(energies.sum())
        if total == 0:
            return 0.0
        return float(energies[energies >= threshold].sum()) / total

    def __iter__(self):
        """Iterate over LIVE seeds as single-element FuzzingSeed batches."""
        for i in range(self._tensors.shape[0]):
            if self._retired and not bool(self._alive[i]):
                continue
            yield FuzzingSeed(
                tensor=self._tensors[i:i+1],
                original_tensor=self._original_tensors[i:i+1],
                original_index=self._original_indices[i:i+1],
                label=self._labels[i:i+1],
                energy=self._energies[i:i+1],
                depth=self._depths[i:i+1],
                id=self._ids[i:i+1],
                parent_id=self._parent_ids[i:i+1],
                select_count=self._select_counts[i:i+1],
            )
    
    def get_stats(self) -> dict[str, object]:
        """Get corpus statistics."""
        n = len(self)
        if n == 0:
            return {
                "total_seeds": 0,
                "avg_energy": 0.0,
                "max_depth": 0,
            }
        
        live = self._alive if self._retired else None
        energies = self._energies[live] if live is not None else self._energies
        depths = self._depths[live] if live is not None else self._depths
        return {
            "total_seeds": n,
            "rows": self.rows,
            "retired_seeds": self._retired,
            "avg_energy": float(energies.mean()),
            "max_energy": float(energies.max()),
            "max_depth": int(depths.max().item()),
            "strategy": self.strategy,
        }
