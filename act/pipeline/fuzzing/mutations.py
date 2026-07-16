"""
Mutation strategies for ACTFuzzer.

Implements gradient-guided, activation-guided, boundary, and random mutations.
All mutations automatically respect InputSpec constraints via projection.

Gradient-guided now accommodates two mutated input generation methods: FGSM (Fast Gradient Sign Method) and PGD (Projected Gradient Descent).
    1) FGSM: single-step gradient-based perturbation.
    2) PGD: iterative gradient-based perturbation.

## Batch Tensor-Based Mutation

All mutation strategies operate on batched inputs [B, C, H, W] for GPU parallelism.
The MutationEngine selects a single strategy per batch and applies it to all seeds
simultaneously, enabling efficient gradient computation (FGSM/PGD) across the batch.
The batch size is aligned with model synthesis (N VNNLib instances), so InputSpec bounds
are already [N, ...] and match the batch dimension directly. After mutation, projection
ensures each sample respects its InputSpec bounds (BOX per-sample bounds via original_index,
or LINF_BALL eps-ball around each seed's original_tensor).

## Adaptive Perturbation Sizing

NOTE: We use "perturb_size" (not "epsilon") to avoid confusion with InputSpec.eps (L∞ radius).
- InputSpec.eps: Defines constraint boundaries (e.g., center ± eps for LINF_BALL)
- Mutation perturb_size: Controls mutation perturbation magnitude (exploration granularity)

This module supports adaptive perturbation sizing that scales with InputSpec bounds to ensure
consistent exploration across different problem scales.

### What is perturb_scale?

`perturb_scale` is the **fraction of the feasible range** that each mutation perturbation covers.

**Interpretation Formula:**
    steps_to_traverse = 1 / perturb_scale

**Calculation:**
    range / perturb_size = range / (range * perturb_scale) = 1 / perturb_scale

**Examples:**
    - perturb_scale=0.1  → Each perturbation covers 10% of range → Takes ~10 steps to traverse from lb to ub
    - perturb_scale=0.2  → Each perturbation covers 20% of range → Takes ~5 steps to traverse from lb to ub
    - perturb_scale=0.05 → Each perturbation covers 5% of range  → Takes ~20 steps to traverse from lb to ub

### Perturbation Modes

1. **adaptive_scalar** (default):
   - Computes single perturb_size from mean range: perturb_size = mean(ub - lb) * perturb_scale
   - Best for: Uniform ranges (e.g., VNNLib BOX constraints with consistent bounds)
   - Example: VNNLib with lb=0.0, ub=1.0 → range=1.0, perturb_size=0.1 (10 steps)

2. **adaptive_perdim** (advanced):
   - Computes per-dimension perturb_size tensor: perturb_size[i] = (ub[i] - lb[i]) * perturb_scale
   - Best for: Non-uniform ranges (e.g., different features with vastly different scales)
   - Example: lb=[0, -100], ub=[1, 100] → perturb_size=[0.1, 20.0] (10 steps per dimension)

3. **fixed** (conventional):
   - Computes perturb_size from mean range: perturb_size = mean(ub - lb)
   - Note: Uses full feasible range as perturbation size 

### Configuration

Set in `act/config/pipeline.yaml`:
```yaml
perturb_mode: "adaptive_scalar"  # Options: "adaptive_scalar", "adaptive_perdim", "fixed"
perturb_scale: 0.1               # Fraction of range per step (default: 0.1 = 10 steps)
```

Copyright (C) 2025 SVF-tools/ACT
License: AGPLv3+
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from act.front_end.specs import InputSpec, InKind
from act.util.device_manager import get_default_device

if TYPE_CHECKING:
    from act.pipeline.fuzzing.corpus import FuzzingSeed


class MutationStrategy(ABC):
    """Base class for mutation strategies."""
    
    @abstractmethod
    def mutate(self, 
               input_tensor: torch.Tensor,
               model: nn.Module,
               activations: Optional[Dict[str, torch.Tensor]] = None,
               label: Optional[torch.Tensor] = None
              ) -> torch.Tensor:
        """
        Apply mutation to input tensor.
        
        Args:
            input_tensor: Seed input
            model: Model for gradient computation
            activations: Activations from previous inference (optional)
            label: Ground truth label for targeted attacks (optional)
        
        Returns:
            Mutated input tensor
        """
        pass


class FGSMMutation(MutationStrategy):
    """
    FGSM-style gradient-guided mutation (single-step).

    Computes gradients to maximize output variance, then applies a single-step
    sign-gradient perturbation.
    """

    def __init__(self, perturb_size: Union[float, torch.Tensor] = 8/255):
        """
        Initialize FGSM mutation.

        Args:
            perturb_size: Mutation perturbation magnitude (scalar or per-dimension tensor)
        """
        self.perturb_size = perturb_size

    def mutate(self, input_tensor, model, activations=None, label=None):
        """Apply FGSM gradient-based perturbation (single-step).
        
        Args:
            input_tensor: Seed input tensor
            model: Model for gradient computation
            activations: Activations from previous inference (unused)
            label: Ground truth label (unused by FGSM, kept for interface consistency)
        """
        # Enable gradients
        x = input_tensor.clone().detach().requires_grad_(True)

        # Forward pass
        output = model(x)

        # Extract output tensor if dict (from VerifiableModel)
        if isinstance(output, dict):
            output = output['output']

        # Compute loss: maximize output variance (unsupervised, label-free)
        loss = output.var()

        # Get gradient w.r.t. input only (avoid accumulating grads on model params)
        grad = torch.autograd.grad(loss, x, retain_graph=False, create_graph=False)[0].detach()

        # FGSM: sign of gradient
        perturb_size = self.perturb_size.to(input_tensor.device) if isinstance(self.perturb_size, torch.Tensor) else self.perturb_size
        perturbation = perturb_size * torch.sign(grad)

        # Apply perturbation
        return input_tensor + perturbation


class PGDMutation(MutationStrategy):
    """
    PGD-style gradient-guided mutation (iterative).

    Implementation follows the notebook approach:
    - Define a feasible box around x0: [x0 - perturb_size, x0 + perturb_size]
    - Optional random start within the feasible box
    - Iterative sign-gradient ascent with projection back to the feasible box

    Loss Function:
    - If label is provided: Cross-entropy loss (adversarial attack, more effective for counterexamples)
    - If label is None: Output variance (unsupervised exploration)

    Note: Global InputSpec constraints are enforced by MutationEngine projection after mutation.
    """

    def __init__(
        self,
        perturb_size: Union[float, torch.Tensor] = 8/255,
        num_steps: int = 50,
        step_size: Optional[float] = None,
        random_start: bool = True,
    ):
        """
        Initialize PGD mutation.

        Args:
            perturb_size: L_infinity radius of local feasible box around the seed (scalar or per-dimension tensor)
            num_steps: Number of PGD iterations
            step_size: Per-iteration step size (if None, computed from feasible box range / steps as in notebook)
            random_start: Whether to start uniformly within the feasible box (recommended)
        """
        self.perturb_size = perturb_size
        self.num_steps = int(num_steps)
        self.step_size = step_size
        self.random_start = random_start

    def mutate(self, input_tensor, model, activations=None, label=None):
        """Apply PGD mutation.
        
        Args:
            input_tensor: Seed input tensor [B, C, H, W] or [1, C, H, W]
            model: Model for gradient computation
            activations: Activations from previous inference (unused by PGD)
            label: Tensor[B] int64, -1 = no label. Uses cross-entropy loss when
                   any label >= 0, otherwise maximizes output variance.
        
        Returns:
            Adversarially perturbed input tensor [B, C, H, W]
        """
        x0 = input_tensor.detach()
        B = x0.shape[0]

        perturb_size = self.perturb_size.to(input_tensor.device) if isinstance(self.perturb_size, torch.Tensor) else self.perturb_size
        x_low = x0 - perturb_size
        x_high = x0 + perturb_size

        # Default step size: spread movement across the available range (notebook heuristic)
        if self.step_size is None:
            # (x_high - x_low) == 2*perturb_size; take max range element as scalar step size
            step_size = float((x_high - x_low).abs().max().item()) / max(self.num_steps, 1)
            step_size = max(step_size, 1e-6)
        else:
            step_size = float(self.step_size)

        # Random start inside feasible box
        if self.random_start:
            x_adv = x_low + torch.rand_like(x0) * (x_high - x_low)
        else:
            x_adv = x0.clone()

        # Ensure start in-bounds
        x_adv = torch.max(torch.min(x_adv, x_high), x_low).detach()

        for _ in range(self.num_steps):
            x_adv.requires_grad_(True)

            # Forward pass
            output = model(x_adv)

            # Extract output tensor if dict (from VerifiableModel)
            if isinstance(output, dict):
                output = output['output']

            # Loss selection based on label availability
            # label is a Tensor[B] int64, -1 = no label
            label_tensor = label if isinstance(label, torch.Tensor) else None
            has_labels = label_tensor is not None and bool((label_tensor >= 0).any().item())
            if has_labels and label_tensor is not None:
                # CW-style margin loss: maximize (max_{j != target} z_j - z_target).
                # More directed than cross-entropy for TOP1/MARGIN robustness - its
                # gradient does not vanish once the target probability is small, so
                # it keeps climbing toward narrow, small-margin violations that an
                # unbounded-CE ascent stalls before.
                assert output.dim() >= 2, (
                    f"Model output should have batch dimension, got shape {output.shape}. "
                    f"Ensure model outputs include batch dimension."
                )
                target = label_tensor.clamp(min=0).to(output.device).view(-1, 1)
                tgt_logit = output.gather(1, target).squeeze(1)
                other = output.scatter(1, target, float("-inf"))
                loss = (other.max(dim=1).values - tgt_logit).sum()
            else:
                # If no label is provided, maximize output variance
                loss = output.var()

            grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0].detach()

            # Gradient ascent on loss
            x_adv = (x_adv + step_size * torch.sign(grad)).detach()

            # Project back to feasible box
            x_adv = torch.max(torch.min(x_adv, x_high), x_low).detach()

        return x_adv.detach()


def _relu_preactivations_batched(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Concatenated ReLU pre-activation values [B, N] across every ReLU
    layer in module-registration order. Keeps the autograd graph (used for
    the hinge-loss gradient in HPGDMutation)."""
    preacts: List[torch.Tensor] = []
    handles = []

    def hook(_module, inputs):
        if inputs and torch.is_tensor(inputs[0]):
            preacts.append(inputs[0])

    for module in model.modules():
        if isinstance(module, nn.ReLU):
            handles.append(module.register_forward_pre_hook(hook))
    try:
        model(x)
    finally:
        for handle in handles:
            handle.remove()
    if not preacts:
        raise RuntimeError("Model has no ReLU layers reachable from a forward pass on this input.")
    return torch.cat([z.flatten(start_dim=1) for z in preacts], dim=1)


def _relu_sign_pattern_batched(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Concrete ReLU pre-activation sign pattern (+-1) [B, N], no grad."""
    with torch.no_grad():
        z = _relu_preactivations_batched(model, x)
    return torch.where(z > 0, torch.ones_like(z), -torch.ones_like(z))


class HPGDMutation(MutationStrategy):
    """
    HPGD (hinge-loss pattern PGD): moves the seed toward a randomly flipped
    ReLU sign-pattern target via hinge-loss sign-gradient descent, instead
    of an attack loss. See act.pipeline.fuzzing.pattern_search_pgd for the
    standalone two-phase exploration this is adapted from -- this is the
    "phase 1 projection" step made into a reusable MutationStrategy.

    Each call flips `flip_count` neurons of the seed's OWN current sign
    pattern (computed fresh from `input_tensor`, per sample) to build a
    per-sample target pattern, then runs `num_steps` of margin-loss
    ("hinge loss") sign-gradient descent toward it, clamped every step to a
    local box of radius `perturb_size` around the seed (mirroring
    PGDMutation's own box). This is the "randomly generate the target
    state" first-pass version; MutationEngine's post-mutate InputSpec
    projection still applies afterward like every other strategy.

    Flip-candidate selection:
      - `candidate_indices` (optional): restrict which neurons are eligible
        to flip (e.g. only the instance's unstable ReLUs, from
        act.pipeline.fuzzing.state_manager.compute_unstable_mask). None =
        every neuron is a candidate. Sampling K candidates out of a
        restricted set (rather than out of every neuron in the network) is
        also what keeps this tractable on CNN-sized networks, where most
        neurons are typically stable and irrelevant to flip anyway.
      - `flip_weights` (optional, settable per call like `perturb_size`):
        per-candidate sampling weight, shape [C] (shared across the batch)
        or [B, C] (per sample -- e.g. PatternStateManager.local_bias(),
        biasing toward neurons that recently proved flippable from that
        specific seed). None = uniform random choice among candidates.

    After mutate() returns, `last_natural_pattern`/`last_achieved_pattern`
    hold the full (unrestricted) pre- and post-mutation ReLU sign patterns,
    for the caller to do its own novelty/diversity bookkeeping and
    local-bias updates via PatternStateManager -- HPGD itself no longer
    keeps its own dedup set; that used to be a plain hash set here, but is
    now PatternStateManager's job (Bloom filter + BK-tree, fixed memory
    instead of growing with every distinct pattern seen).
    """

    def __init__(
        self,
        perturb_size: Union[float, torch.Tensor] = 8 / 255,
        flip_count: int = 10,
        num_steps: int = 10,
        step_size: Optional[float] = None,
        margin: float = 0.01,
        candidate_indices: Optional[torch.Tensor] = None,
    ):
        """
        Initialize HPGD mutation.

        Args:
            perturb_size: L_infinity radius of the local feasible box around the seed (scalar or per-dimension tensor)
            flip_count: Number of ReLU neurons to flip away from the seed's own natural pattern, per sample
            num_steps: Number of hinge-loss sign-gradient steps
            step_size: Per-step size (if None, computed from the feasible box range / steps)
            margin: Hinge-loss margin -- a neuron counts as "matching" the target once its
                signed pre-activation clears this margin
            candidate_indices: Optional 1-D LongTensor restricting which neurons may be
                flipped (e.g. unstable-only). None = every neuron is a candidate.
        """
        self.perturb_size = perturb_size
        self.flip_count = int(flip_count)
        self.num_steps = int(num_steps)
        self.step_size = step_size
        self.margin = float(margin)
        self.candidate_indices = candidate_indices
        self.flip_weights: Optional[torch.Tensor] = None
        self.last_natural_pattern: Optional[torch.Tensor] = None
        self.last_achieved_pattern: Optional[torch.Tensor] = None

    def mutate(self, input_tensor, model, activations=None, label=None):
        """Apply HPGD mutation.

        Args:
            input_tensor: Seed input tensor [B, ...]
            model: Model for gradient computation
            activations: Activations from previous inference (unused by HPGD)
            label: Ground truth label (unused by HPGD -- pattern-driven, not label-driven)

        Returns:
            Mutated input tensor [B, ...], moved toward each sample's randomly-flipped
            target ReLU sign pattern.
        """
        x0 = input_tensor.detach()
        B = x0.shape[0]
        device = x0.device

        natural_pattern = _relu_sign_pattern_batched(model, x0)
        num_neurons = natural_pattern.shape[1]
        candidates = (
            self.candidate_indices.to(device) if self.candidate_indices is not None
            else torch.arange(num_neurons, device=device)
        )
        C = candidates.numel()

        target_pattern = natural_pattern.clone()
        k = min(self.flip_count, C)
        if k > 0:
            if self.flip_weights is not None:
                weights = self.flip_weights.to(device)
                if weights.dim() == 1:
                    weights = weights.unsqueeze(0).expand(B, -1)
                for b in range(B):
                    local_idx = torch.multinomial(weights[b].clamp(min=1e-8), k, replacement=False)
                    target_pattern[b, candidates[local_idx]] *= -1
            else:
                for b in range(B):
                    local_idx = torch.randperm(C, device=device)[:k]
                    target_pattern[b, candidates[local_idx]] *= -1

        perturb_size = self.perturb_size.to(device) if isinstance(self.perturb_size, torch.Tensor) else self.perturb_size
        x_low = x0 - perturb_size
        x_high = x0 + perturb_size

        if self.step_size is None:
            step_size = float((x_high - x_low).abs().max().item()) / max(self.num_steps, 1)
            step_size = max(step_size, 1e-6)
        else:
            step_size = float(self.step_size)

        x = x0.clone()
        for _ in range(self.num_steps):
            x_req = x.detach().clone().requires_grad_(True)
            z_all = _relu_preactivations_batched(model, x_req)
            # Hinge loss: penalize any neuron whose signed pre-activation hasn't
            # cleared `margin` in the target's direction yet.
            violation = (self.margin - target_pattern * z_all).clamp(min=0)
            loss = violation.sum()
            grad = torch.autograd.grad(loss, x_req, retain_graph=False, create_graph=False)[0].detach()
            x = (x_req.detach() - step_size * torch.sign(grad)).detach()
            x = torch.max(torch.min(x, x_high), x_low).detach()

        achieved_pattern = _relu_sign_pattern_batched(model, x)
        self.last_natural_pattern = natural_pattern
        self.last_achieved_pattern = achieved_pattern

        return x.detach()


class HPGDPullbackMutation(MutationStrategy):
    """
    HPGD pull-back: the GCE (generate-counterexample) counterpart to
    HPGDMutation, adapted from PatternSearchPGD's round 3. Given a batch of
    ANCHOR patterns (typically confirmed counterexamples' own achieved
    patterns, from PatternStateManager.pick_ce_seeds), nudges each seed with
    a small random perturbation and then hinge-loss gradient-walks it back
    toward its OWN anchor pattern -- densely resampling the neighborhood of
    a known-violating region instead of exploring blindly. Stops each
    sample independently once its Hamming distance to its own anchor is
    <= `hamming_radius`.

    Unlike every other strategy, this one needs external per-sample state
    (the anchors) that MutationEngine's uniform interface has no slot for,
    so it is not registered in MutationEngine.strategies / driven by
    weighted strategy selection -- the GCE task in ACTFuzzer calls
    `set_anchors()` then `mutate()` on it directly.
    """

    def __init__(
        self,
        num_steps: int = 10,
        step_size: Optional[float] = None,
        margin: float = 0.01,
        hamming_radius: int = 3,
        noise_scale: float = 0.05,
    ):
        """
        Args:
            num_steps: Number of hinge-loss sign-gradient steps
            step_size: Per-step size (if None, derived from noise_scale / num_steps)
            margin: Hinge-loss margin, same meaning as HPGDMutation's
            hamming_radius: Stop a sample's gradient walk once its achieved pattern is
                within this many neurons of its own anchor pattern
            noise_scale: Initial random perturbation magnitude applied to each anchor
                input before gradient-walking back -- starting exactly at the anchor
                gives zero margin-loss gradient (nothing to fix)
        """
        self.num_steps = int(num_steps)
        self.step_size = step_size
        self.margin = float(margin)
        self.hamming_radius = int(hamming_radius)
        self.noise_scale = float(noise_scale)
        self._anchor_patterns: Optional[torch.Tensor] = None
        self.last_final_hamming: Optional[torch.Tensor] = None

    def set_anchors(self, anchor_patterns: torch.Tensor) -> None:
        """anchor_patterns: [B, N] full (unrestricted) ReLU sign patterns,
        one per sample in the next mutate() call's batch."""
        self._anchor_patterns = anchor_patterns

    def mutate(self, input_tensor, model, activations=None, label=None):
        """Apply HPGD pull-back mutation. `set_anchors()` must be called first.

        Args:
            input_tensor: Anchor seed input tensor [B, ...] (the CE inputs themselves)
            model: Model for gradient computation
            activations: Unused
            label: Unused -- pattern-driven, not label-driven

        Returns:
            Mutated input tensor [B, ...], nudged off each anchor and gradient-walked
            back toward it.
        """
        if self._anchor_patterns is None:
            raise RuntimeError("HPGDPullbackMutation.set_anchors() must be called before mutate().")
        x0 = input_tensor.detach()
        B = x0.shape[0]
        device = x0.device
        anchor = self._anchor_patterns.to(device)

        noise = (torch.rand_like(x0) * 2 - 1) * self.noise_scale
        x = (x0 + noise).detach()

        step_size = self.step_size
        if step_size is None:
            step_size = max(self.noise_scale / max(self.num_steps, 1), 1e-6)
        else:
            step_size = float(step_size)

        final_hamming = torch.zeros(B, dtype=torch.long, device=device)
        broadcast_shape = (B,) + (1,) * (x0.dim() - 1)
        for _ in range(self.num_steps):
            x_req = x.detach().clone().requires_grad_(True)
            z_all = _relu_preactivations_batched(model, x_req)
            achieved_sign = torch.where(z_all.detach() > 0, torch.ones_like(z_all), -torch.ones_like(z_all))
            hamming_now = (achieved_sign != anchor).sum(dim=1)
            final_hamming = hamming_now
            done = hamming_now <= self.hamming_radius
            if bool(done.all().item()):
                break
            violation = (self.margin - anchor * z_all).clamp(min=0)
            loss = violation.sum()
            grad = torch.autograd.grad(loss, x_req, retain_graph=False, create_graph=False)[0].detach()
            stepped = (x_req.detach() - step_size * torch.sign(grad)).detach()
            # Only samples that haven't converged yet keep moving, so ones
            # already close to their own anchor don't drift back away from it.
            x = torch.where(done.view(*broadcast_shape), x, stepped)

        self.last_final_hamming = final_hamming
        return x.detach()


class ActivationMutation(MutationStrategy):
    """
    Mutation to maximize neuron activation changes.
    
    Uses random direction weighted by recent activation patterns.
    """
    
    def __init__(self, perturb_size: Union[float, torch.Tensor] = 0.01):
        """
        Initialize activation mutation.
        
        Args:
            perturb_size: Mutation perturbation magnitude (scalar or per-dimension tensor)
        """
        self.perturb_size = perturb_size
    
    def mutate(self, input_tensor, model, activations=None, label=None):
        """Apply activation-guided perturbation.
        
        Args:
            input_tensor: Seed input tensor
            model: Model (unused)
            activations: Activations from previous inference (unused currently)
            label: Ground truth label (unused, kept for interface consistency)
        """
        # Random direction (future: weight by inactive neurons)
        direction = torch.randn_like(input_tensor)
        
        # Normalize and scale
        direction = direction / (direction.norm() + 1e-8)
        # Handle both scalar and tensor perturb_size
        perturb_size = self.perturb_size.to(input_tensor.device) if isinstance(self.perturb_size, torch.Tensor) else self.perturb_size
        perturbation = perturb_size * direction
        
        return input_tensor + perturbation


class BoundaryMutation(MutationStrategy):
    """
    Mutation toward InputSpec boundaries.
    
    Explores edge cases where properties are more likely to fail.
    """
    
    def __init__(self, perturb_size: Union[float, torch.Tensor] = 0.005):
        """
        Initialize boundary mutation.
        
        Args:
            perturb_size: Mutation perturbation magnitude toward boundary (scalar or per-dimension tensor)
        """
        self.perturb_size = perturb_size
    
    def mutate(self, input_tensor, model, activations=None, label=None):
        """Push toward boundaries (will be projected by engine).
        
        Args:
            input_tensor: Seed input tensor
            model: Model (unused)
            activations: Activations (unused)
            label: Ground truth label (unused, kept for interface consistency)
        """
        # Random direction
        direction = torch.sign(torch.randn_like(input_tensor))
        
        # Scale
        # Handle both scalar and tensor perturb_size
        perturb_size = self.perturb_size.to(input_tensor.device) if isinstance(self.perturb_size, torch.Tensor) else self.perturb_size
        perturbation = perturb_size * direction
        
        return input_tensor + perturbation


class RandomMutation(MutationStrategy):
    """Random Gaussian perturbation (baseline)."""
    
    def __init__(self, perturb_size: Union[float, torch.Tensor] = 0.005):
        """
        Initialize random mutation.
        
        Args:
            perturb_size: Standard deviation of Gaussian noise (scalar or per-dimension tensor)
        """
        self.perturb_size = perturb_size
    
    def mutate(self, input_tensor, model, activations=None, label=None):
        """Apply random Gaussian noise.
        
        Args:
            input_tensor: Seed input tensor
            model: Model (unused)
            activations: Activations (unused)
            label: Ground truth label (unused, kept for interface consistency)
        """
        # Handle both scalar and tensor perturb_size
        perturb_size = self.perturb_size.to(input_tensor.device) if isinstance(self.perturb_size, torch.Tensor) else self.perturb_size
        noise = torch.randn_like(input_tensor) * perturb_size
        return input_tensor + noise


class MutationEngine:
    """
    Mutation engine with strategy selection and constraint projection.
    
    Features:
    - Weighted random strategy selection
    - Automatic InputSpec projection
    - Activation capture via forward hooks
    - Single strategy per mutation call for GPU parallelism
    
    Example:
        >>> engine = MutationEngine(model, input_spec, weights)
        >>> mutated = engine.mutate([seed1, seed2])
        >>> activations = engine.get_activation_map()
    """
    
    def __init__(self,
                 model: nn.Module,
                 input_spec: Optional[InputSpec],
                 weights: Dict[str, float],
                 perturb_mode: str = "fixed",
                 perturb_scale: float = 0.1):
        """
        Initialize mutation engine.
        
        Args:
            model: VerifiableModel (or core model) for gradient computation
            input_spec: InputSpec for constraint projection (batched bounds from model synthesis)
            weights: Strategy weights (e.g., {"gradient": 0.4, "random": 0.1})
            perturb_mode: Perturbation size computation mode ("adaptive_scalar", "adaptive_perdim", "fixed")
            perturb_scale: Fraction of range per mutation perturbation (e.g., 0.1 = 10% = ~10 steps to traverse)
        """
        self.model = model
        self.input_spec = input_spec
        self.device = get_default_device()
        self.perturb_mode = perturb_mode
        self.perturb_scale = perturb_scale
        
        # Compute perturb_size based on mode
        perturb_size = self._compute_adaptive_perturb_size()
        
        # Initialize strategies with computed perturb_size
        self.strategies = {
            "gradient": FGSMMutation(perturb_size=perturb_size),
            "pgd": PGDMutation(perturb_size=perturb_size),
            "hpgd": HPGDMutation(perturb_size=perturb_size),
            "activation": ActivationMutation(perturb_size=perturb_size),
            "boundary": BoundaryMutation(perturb_size=perturb_size),
            "random": RandomMutation(perturb_size=perturb_size),
        }

        # Validate and normalize weights
        unknown = set(weights.keys()) - set(self.strategies.keys())
        if unknown:
            raise ValueError(
                f"Unknown mutation strategy keys in weights: {sorted(unknown)}. "
                f"Valid options: {sorted(self.strategies.keys())}"
            )
        total = sum(float(v) for v in weights.values())
        if total <= 0.0:
            raise ValueError(f"Mutation weights must sum to > 0. Got total={total}.")
        self.weights = {k: float(v) / total for k, v in weights.items()}
        
        # Statistics
        self.total_mutations = 0
        self.activation_map: Dict[str, torch.Tensor] = {}
        self.last_strategy: Optional[str] = None  # Track last mutation strategy for tracing
        self.last_gradients: Optional[Dict[str, torch.Tensor]] = None  # For Level 3 tracing: gradient capture
        self.last_loss: Optional[float] = None  # For Level 3 tracing: loss value capture
        
        # Setup hooks for activation capture
        self._setup_hooks()
    
    def _compute_adaptive_perturb_size(self) -> Union[float, torch.Tensor]:
        """
        Compute perturb_size based on InputSpec bounds and perturb_mode.
        
        Note: We use "perturb_size" to avoid confusion with InputSpec.eps (L∞ radius constraint).
        
        Returns:
            - float: Scalar perturb_size (for "adaptive_scalar" or "fixed" modes)
            - torch.Tensor: Per-dimension perturb_size (for "adaptive_perdim" mode)
        
        Algorithm:
            1. adaptive_scalar: perturb_size = mean(ub - lb) * perturb_scale
               - Single perturb_size value computed from mean range
               - Best for uniform ranges (e.g., VNNLib BOX constraints)
            
            2. adaptive_perdim: perturb_size = (ub - lb) * perturb_scale
               - Tensor of perturb_size values, one per dimension
               - Best for non-uniform ranges (different feature scales)
            
            3. fixed: Computes perturb_size from mean range (range-aware)
               - perturb_size = mean(ub - lb)
               - Uses the average feasible range as perturbation size
        
        Interpretation:
            perturb_scale represents the fraction of range each perturbation covers.
            steps_to_traverse = 1 / perturb_scale
            
            Examples:
                - perturb_scale=0.1  → 10% per perturbation → ~10 steps to traverse
                - perturb_scale=0.2  → 20% per perturbation → ~5 steps to traverse
                - perturb_scale=0.05 → 5% per perturbation  → ~20 steps to traverse
        """
        
        if self.input_spec is None:
            print(f"[MutationEngine] No InputSpec provided, falling back to fixed perturb_size=0.01")
            return 0.01
        
        # Extract bounds based on InputSpec kind
        # BOX: explicit lb/ub bounds define the feasible region directly
        # LINF_BALL: feasible region is [center - eps, center + eps] (L∞ ball around center)
        if self.input_spec.kind == InKind.BOX:
            # BOX constraints: lb and ub are directly specified
            assert self.input_spec.lb is not None and self.input_spec.ub is not None
            lb = self.input_spec.lb
            ub = self.input_spec.ub
        elif self.input_spec.kind == InKind.LINF_BALL:
            # L∞ ball constraints: range is 2*eps around center point
            # The feasible region is all points x such that ||x - center||_∞ <= eps
            assert self.input_spec.center is not None and self.input_spec.eps is not None
            center = self.input_spec.center
            eps = torch.as_tensor(self.input_spec.eps, device=center.device, dtype=center.dtype)
            lb = center - eps
            ub = center + eps
        elif self.input_spec.kind == InKind.LP_EMBEDDING:
            lb, ub = self.input_spec.materialize_box_seed()
        else:
            print(f"[MutationEngine] Unsupported InputSpec kind '{self.input_spec.kind}', falling back to fixed perturb_size=0.01")
            return 0.01
        
        # Compute range for perturbation scaling
        range_tensor = ub - lb
        
        # Range-aware fixed mode: use mean range as perturb_size
        if self.perturb_mode == "fixed":
            return (ub - lb).mean().item()
        
        # Compute single perturb_size from mean range
        if self.perturb_mode == "adaptive_scalar":
            mean_range = range_tensor.mean().item()
            perturb_size = mean_range * self.perturb_scale
            
            print(f"[MutationEngine] Adaptive Scalar Perturbation Size:")
            print(f"  - perturb_scale: {self.perturb_scale} (fraction of range per perturbation)")
            print(f"  - mean_range: {mean_range:.6f}")
            print(f"  - computed perturb_size: {perturb_size:.6f}")
            print(f"  - steps_to_traverse: ~{1/self.perturb_scale:.1f} steps")
            print(f"  - interpretation: Each mutation perturbation covers {self.perturb_scale*100:.1f}% of the range")
            
            return perturb_size
        
        # Compute per-dimension perturb_size tensor
        elif self.perturb_mode == "adaptive_perdim":
            perturb_size_tensor = range_tensor * self.perturb_scale
            
            print(f"[MutationEngine] Adaptive Per-Dimension Perturbation Size:")
            print(f"  - perturb_scale: {self.perturb_scale} (fraction of range per perturbation)")
            print(f"  - range shape: {range_tensor.shape}")
            print(f"  - perturb_size shape: {perturb_size_tensor.shape}")
            print(f"  - perturb_size range: [{perturb_size_tensor.min().item():.6f}, {perturb_size_tensor.max().item():.6f}]")
            print(f"  - perturb_size mean: {perturb_size_tensor.mean().item():.6f}")
            print(f"  - steps_to_traverse: ~{1/self.perturb_scale:.1f} steps per dimension")
            print(f"  - interpretation: Each mutation perturbation covers {self.perturb_scale*100:.1f}% of each dimension's range")
            
            return perturb_size_tensor
        
        else:
            raise ValueError(f"Unknown perturb_mode: {self.perturb_mode}. "
                           f"Valid options: 'adaptive_scalar', 'adaptive_perdim', 'fixed'")
    
    def _setup_hooks(self):
        """Setup forward hooks to capture activations."""
        def make_hook(name):
            def hook(module, input, output):
                # Store activation (handle both tensor and dict outputs)
                if isinstance(output, torch.Tensor):
                    self.activation_map[name] = output.detach()
                elif isinstance(output, dict) and 'output' in output:
                    self.activation_map[name] = output['output'].detach()
            
            return hook
        
        # Register hooks on computational layers (ReLU, Linear, Conv2d)
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.ReLU, nn.Linear, nn.Conv2d)):
                module.register_forward_hook(make_hook(name))
                
    def mutate(self, seeds: 'FuzzingSeed') -> torch.Tensor:
        """
        Apply mutation to seeds.
        
        A single strategy is selected for all seeds, enabling GPU parallelism
        for gradient-based strategies (FGSM/PGD).
        
        Args:
            seeds: FuzzingSeed batch with .tensor [B,C,H,W] and .label [B] int64 (-1=no label)
        
        Returns:
            Mutated tensor [B, C, H, W] satisfying InputSpec constraints
        """
        if not seeds:
            raise ValueError("Empty seed batch")
        
        B = len(seeds)
        
        # Use batch tensor directly: [B, C, H, W]
        batch_input = seeds.tensor.to(self.device)
        labels = seeds.label.to(self.device)
        
        # Select strategy (same for all samples)
        strategy_names = list(self.weights.keys())
        strategy_probs = list(self.weights.values())
        strategy_name = np.random.choice(strategy_names, p=strategy_probs)
        strategy = self.strategies[strategy_name]
        
        # Store strategy for tracing
        self.last_strategy = strategy_name
        
        # Dynamic per-seed scale: s_b = 1 - (1 - s₀)^{n_b+1}
        if self.perturb_mode != "fixed" and self.input_spec is not None:
            if self.input_spec.kind == InKind.LINF_BALL:
                assert self.input_spec.eps is not None
                _orig = seeds.original_tensor.to(self.device)
                _eps = torch.as_tensor(self.input_spec.eps, device=_orig.device, dtype=_orig.dtype)
                _lb = _orig - _eps
                _ub = _orig + _eps
            else:
                if self.input_spec.kind == InKind.LP_EMBEDDING:
                    _lb, _ub = self.input_spec.materialize_box_seed()
                    _lb = _lb.to(self.device)
                    _ub = _ub.to(self.device)
                else:
                    assert self.input_spec.lb is not None and self.input_spec.ub is not None
                    _lb = self.input_spec.lb.to(self.device)
                    _ub = self.input_spec.ub.to(self.device)
                if _lb.shape[0] > 1:
                    _ix = seeds.original_index.clamp(max=_lb.shape[0] - 1).to(self.device)
                    _lb, _ub = _lb[_ix], _ub[_ix]
                elif _lb.shape[0] == 1 and B > 1:
                    _lb = _lb.expand(B, *_lb.shape[1:])
                    _ub = _ub.expand(B, *_ub.shape[1:])
                else:
                    _lb, _ub = _lb[:B], _ub[:B]
            n = seeds.select_count.float().to(self.device)
            s_b = 1.0 - (1.0 - self.perturb_scale) ** (n + 1.0)
            reach = torch.max(batch_input - _lb, _ub - batch_input)
            shape = [B] + [1] * (reach.dim() - 1)
            perturb_size = s_b.view(*shape) * reach
            if self.perturb_mode == "adaptive_scalar":
                perturb_size = perturb_size.mean(dim=list(range(1, perturb_size.dim())), keepdim=True)
            strategy.perturb_size = perturb_size
        
        # Apply mutation
        mutated = strategy.mutate(
            batch_input,
            self.model,
            self.activation_map,
            label=labels
        )
        
        # Project to InputSpec constraints
        mutated = self._project(mutated, seeds)
        
        self.total_mutations += B
        return mutated
    
    def _project(self, tensor: torch.Tensor, seeds: 'Optional[FuzzingSeed]' = None) -> torch.Tensor:
        """
        Project mutated tensor back into the feasible region defined by InputSpec constraints.
        
        Since fuzzing batch size is aligned with model synthesis (N VNNLib instances),
        the InputSpec bounds (lb/ub) are already [N, ...] and match the batch dimension directly.
        
        **BOX (InKind.BOX)**:
            Clamps to per-sample lb/ub bounds from InputSpec (already batch-aligned).
            Uses seeds.original_index tensor to gather correct per-sample bounds.
        
        **LINF_BALL (InKind.LINF_BALL)**:
            Clamps perturbation delta to [-eps, +eps] around each seed's ORIGINAL input,
            preserving the L∞ distance invariant across mutation chains.
        
        **LIN_POLY (InKind.LIN_POLY)**:
            Not yet implemented — returns tensor unchanged.
        
        Args:
            tensor: Mutated input tensor [B, ...] to project
            seeds: FuzzingSeed batch with original_index [B] and original_tensor [B, ...]
        
        Returns:
            Projected tensor [B, ...] satisfying InputSpec constraints
        """
        if self.input_spec is None:
            return tensor
        
        B = tensor.shape[0]
        
        if self.input_spec.kind in (InKind.BOX, InKind.LP_EMBEDDING):
            if self.input_spec.kind == InKind.LP_EMBEDDING:
                lb, ub = self.input_spec.materialize_box_seed()
                lb = lb.to(tensor.device)
                ub = ub.to(tensor.device)
            else:
                assert self.input_spec.lb is not None and self.input_spec.ub is not None
                lb = self.input_spec.lb.to(tensor.device)
                ub = self.input_spec.ub.to(tensor.device)
            
            # bounds: use seeds.original_index to gather correct bounds
            if lb.shape[0] > 1 and seeds is not None:
                indices = seeds.original_index.clamp(max=lb.shape[0] - 1).to(lb.device)
                lb = lb[indices]  # (B, ...) vectorized gather
                ub = ub[indices]
            elif lb.shape[0] == 1 and B > 1:
                lb = lb.expand(B, *lb.shape[1:])
                ub = ub.expand(B, *ub.shape[1:])
            else:
                lb = lb[:B]
                ub = ub[:B]
            
            return torch.clamp(tensor, lb, ub)
        
        elif self.input_spec.kind == InKind.LINF_BALL:
            eps = self.input_spec.eps
            assert eps is not None
            
            assert seeds is not None and len(seeds) == B, \
                f"LINF_BALL projection requires seeds (got {len(seeds) if seeds else 0}, expected {B})"
            
            # Use original_tensor as center to maintain L∞ distance from original
            center = seeds.original_tensor.to(tensor.device)
            eps = torch.as_tensor(eps, device=tensor.device, dtype=tensor.dtype)
            
            delta = tensor - center
            delta = torch.clamp(delta, -eps, eps)
            return center + delta
        
        elif self.input_spec.kind == InKind.LIN_POLY:
            # TODO: Implement quadratic programming projection
            return tensor
        
        return tensor
    
    def get_activation_map(self) -> Dict[str, torch.Tensor]:
        """Get activations from last inference."""
        return self.activation_map
    
    
    def get_last_gradients(self) -> Optional[Dict[str, torch.Tensor]]:
        """Get gradients from last mutation (Level 3 tracing only)."""
        return self.last_gradients
    
    def get_last_loss(self) -> Optional[float]:
        """Get loss value from last mutation (Level 3 tracing only)."""
        return self.last_loss
    
    def get_stats(self) -> Dict[str, Any]:
        """Get mutation statistics."""
        perturb_size_info = {}
        for strategy_name, strategy in self.strategies.items():
            perturb_size = strategy.perturb_size
            if isinstance(perturb_size, torch.Tensor):
                perturb_size_info[strategy_name] = {
                    "type": "tensor",
                    "shape": list(perturb_size.shape),
                    "min": perturb_size.min().item(),
                    "max": perturb_size.max().item(),
                    "mean": perturb_size.mean().item()
                }
            else:
                perturb_size_info[strategy_name] = {
                    "type": "scalar",
                    "value": perturb_size
                }
        
        return {
            "total_mutations": self.total_mutations,
            "strategy_weights": self.weights,
            "perturb_mode": self.perturb_mode,
            "perturb_scale": self.perturb_scale,
            "perturb_size_values": perturb_size_info,
        }
