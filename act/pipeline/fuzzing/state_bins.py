"""How a pre-activation vector is turned into the discrete state code.

The ReLU state is `sign(z)`: one bit per neuron, split at the kink.  Ported
verbatim to Sigmoid/Tanh that bit splits at the inflection point instead, and
measured on the ERAN nets it is nearly constant inside a verification box --
pre-activations sit at |z| ~ 3.5-22 while the only boundary is at 0, so almost
no neuron can reach it and the "state" barely varies.

A three-way split at `z = -tau, +tau` gives those neurons a boundary they can
actually reach.  Measured on ERAN 6x100 (a wall counts when it lies within the
box's first-order budget for moving that pre-activation): sigmoid 24/600
neurons under the sign partition against 54/600 under this one, tanh 20/600
against 40/600 -- 2.0-2.3x more reachable walls.

Note that three segments means exactly TWO walls, so the sign wall at 0 is
GONE: this is a different partition, not a refinement of the sign one. On these
nets that costs almost nothing (of the 24 sign-reachable neurons on sigmoid,
22 are also +-tau-reachable), but it is not free in general.

## Why two BITS per neuron and not one trit

Everything downstream of the code -- the BK-tree's Hamming radius, the
fingerprint packing, the Bloom filter, the per-neuron marginal occupancy that
steers target proposal -- is written against a +-1 vector.  A three-valued
digit would need all of it rewritten, and a Hamming distance over trits is not
the same metric.  So a three-way split is expressed as TWO binary coordinates
of the same +-1 alphabet:

    b_lo = sign(z + tau)      b_hi = sign(z - tau)

      z < -tau        (-1, -1)     low / saturated negative
    -tau < z < tau    (+1, -1)     middle / responsive band
      tau < z         (+1, +1)     high / saturated positive
                      (-1, +1)     unreachable: z < -tau and z > tau

The two coordinates of a neuron are kept ADJACENT (2j, 2j+1) so that moving one
bin is Hamming distance 1 and crossing the whole range is 2 -- the metric then
reads as "how many bin boundaries separate these states", which is what the
BK-tree's radius is supposed to mean.

`bins=2` reproduces the old behaviour exactly: M == N, every threshold 0, and
`code()` is `sign(z)`.  It is the default everywhere, so nothing changes for a
run that does not ask for the split.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class StateBinning:
    """The map from pre-activations [B, N] to a state code [B, M] in {-1,+1}.

    thresholds  [M] float: coordinate m tests `z[neuron_of[m]] - thresholds[m]`
    neuron_of   [M] long : which pre-activation each coordinate reads
    """

    bins: int
    tau: float
    num_neurons: int
    thresholds: torch.Tensor
    neuron_of: torch.Tensor

    @property
    def num_coords(self) -> int:
        return int(self.thresholds.numel())

    @staticmethod
    def build(num_neurons: int, bins: int = 2, tau: float = 1.0,
              device: Optional[torch.device] = None,
              dtype: torch.dtype = torch.float32) -> "StateBinning":
        n = int(num_neurons)
        if bins == 2:
            return StateBinning(
                bins=2, tau=0.0, num_neurons=n,
                thresholds=torch.zeros(n, device=device, dtype=dtype),
                neuron_of=torch.arange(n, device=device),
            )
        if bins != 3:
            raise ValueError(f"state bins must be 2 or 3, got {bins}")
        t = abs(float(tau))
        thr = torch.empty(2 * n, device=device, dtype=dtype)
        thr[0::2] = -t   # b_lo tests z > -tau
        thr[1::2] = t    # b_hi tests z > +tau
        return StateBinning(
            bins=3, tau=t, num_neurons=n, thresholds=thr,
            neuron_of=torch.arange(n, device=device).repeat_interleave(2),
        )

    def to(self, device) -> "StateBinning":
        if self.thresholds.device == torch.device(device):
            return self
        return StateBinning(self.bins, self.tau, self.num_neurons,
                            self.thresholds.to(device), self.neuron_of.to(device))

    # ------------------------------------------------------------------
    def code(self, z: torch.Tensor) -> torch.Tensor:
        """[B, N] pre-activations -> [B, M] code in {-1, +1}."""
        s = self.shifted(z)
        return torch.where(s > 0, torch.ones_like(s), -torch.ones_like(s))

    def shifted(self, z: torch.Tensor) -> torch.Tensor:
        """[B, M] `z[neuron_of] - thresholds`, keeping any autograd graph.

        This is what the hinge loss must be written against: a coordinate is
        satisfied when `target * shifted` clears the margin, exactly as the
        two-bin case tests `target * z`.
        """
        if self.bins == 2:
            return z
        return z.repeat_interleave(2, dim=-1) - self._walls(z.shape[-1], z)

    def expand_bounds(self, lb: torch.Tensor, ub: torch.Tensor) -> torch.Tensor:
        """Which coordinates the box straddles: `lb < threshold < ub`.

        The two-bin case reduces to the familiar `(lb < 0) & (ub > 0)`.
        """
        if self.bins == 2:
            return (lb < 0) & (ub > 0)
        thr = self._walls(lb.shape[-1], lb)
        return ((lb.repeat_interleave(2, dim=-1) < thr)
                & (ub.repeat_interleave(2, dim=-1) > thr))

    # ------------------------------------------------------------------
    def repair(self, code: torch.Tensor) -> torch.Tensor:
        """Force a code onto the reachable set.

        Only (-1, +1) is unreachable -- it asks for z < -tau and z > +tau at
        once.  Independent per-coordinate flips produce it regularly, and
        aiming at an infeasible target is already known to be what makes the
        projection land nowhere, so targets are projected back before use.
        The repair sends it to (+1, +1) by restoring b_lo. Either flip could
        have produced the bad pair, so this is a convention, not a recovery of
        the original intent; what matters is that the aimed-at code is one the
        network can actually be in.
        """
        if self.bins == 2:
            return code
        out = code.clone()
        lo, hi = out[..., 0::2], out[..., 1::2]
        bad = (lo < 0) & (hi > 0)
        lo[bad] = 1.0
        return out

    # ------------------------------------------------------------------
    # Naming a destination SEGMENT, rather than flipping a coordinate.
    #
    # Independent per-coordinate flips cannot express this partition properly.
    # Enumerated, of the six single-coordinate flips two are wrong: from the
    # high segment, flipping b_lo names the unreachable pair and `repair` sends
    # it straight back, so the flip is a silent no-op; and from the low
    # segment, flipping b_hi is repaired into the HIGH segment, a two-segment
    # jump dressed up as a Hamming-1 move. Worse, "high -> low" cannot be asked
    # for at all that way -- it needs both coordinates to move together.
    #
    # So a caller that means "put neuron j in segment s" must say exactly that.
    LOW, MID, HIGH = 0, 1, 2

    def bin_of(self, code: torch.Tensor) -> torch.Tensor:
        """[..., M] code -> [..., N] segment index in {0, 1, 2}."""
        if self.bins == 2:
            return (code > 0).long()
        return (code[..., 0::2] > 0).long() + (code[..., 1::2] > 0).long()

    def write_bin(self, code: torch.Tensor, rows: torch.Tensor,
                  neurons: torch.Tensor, dest: torch.Tensor) -> torch.Tensor:
        """Set `code[rows, neurons]` to segment `dest`, in place.

        Writes BOTH coordinates of the neuron, so the result is always a
        reachable code and the requested move is the one that happens --
        including high -> low, which no single coordinate flip can name.
        """
        if self.bins == 2:
            code[rows, neurons] = torch.where(
                dest > 0, torch.ones_like(dest, dtype=code.dtype),
                -torch.ones_like(dest, dtype=code.dtype))
            return code
        lo = torch.where(dest > self.LOW, 1.0, -1.0).to(code.dtype)
        hi = torch.where(dest >= self.HIGH, 1.0, -1.0).to(code.dtype)
        code[rows, 2 * neurons] = lo
        code[rows, 2 * neurons + 1] = hi
        return code

    def _walls(self, num_neurons: int, like: torch.Tensor) -> torch.Tensor:
        """[2 * num_neurons] wall positions, tiled (-tau, +tau) per neuron.

        Built for whatever WIDTH it is handed rather than read off the stored
        `thresholds`, because bound propagation expands one LAYER at a time and
        only concatenates afterwards. Interleaving is per neuron, so expanding
        each layer and concatenating gives the same coordinate order as
        expanding the concatenated vector -- but only if the expansion does not
        assume the full network's width.
        """
        w = torch.tensor([-self.tau, self.tau], device=like.device, dtype=like.dtype)
        return w.repeat(int(num_neurons))

    def neuron_index_of_coord(self, coords: torch.Tensor) -> torch.Tensor:
        return self.neuron_of.to(coords.device).index_select(0, coords)
