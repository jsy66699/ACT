
from __future__ import annotations
from typing import Optional, TYPE_CHECKING
import numpy as np
import os
from act.back_end.solver.solver_base import (
    Solver,
    SolverCaps,
    SolveStatus,
)
from act.config.config import GurobiConfig
from act.util.path_config import get_project_root

if TYPE_CHECKING:
    from act.back_end.core import Bounds
    from act.back_end.solver.solver_base import BatchLPProblem, BatchLPSolution

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    print("Warning: Gurobi not available. Some operations will use alternative solvers.")
    GUROBI_AVAILABLE = False


def is_gurobi_available() -> bool:
    return bool(GUROBI_AVAILABLE)

def setup_gurobi_license():
    """Setup Gurobi license path based on current folder layout."""
    if 'GRB_LICENSE_FILE' not in os.environ:
        if 'ACTHOME' in os.environ:
            license_path = os.path.join(os.environ['ACTHOME'], 'modules', 'gurobi', 'gurobi.lic')
            print(f"[ACT] Using ACTHOME environment variable: {os.path.relpath(os.environ['ACTHOME'])}")
        else:
            project_root = get_project_root()
            license_path = os.path.join(project_root, 'modules', 'gurobi', 'gurobi.lic')
            print(f"[ACT] Auto-detecting project root: {os.path.relpath(project_root)}")
        
        license_path = os.path.abspath(license_path)
        
        if os.path.exists(license_path):
            os.environ['GRB_LICENSE_FILE'] = license_path
            print(f"[ACT] Gurobi license found: {os.path.relpath(license_path)}")
        else:
            print(f"[WARN] Gurobi license not found: {os.path.relpath(license_path)}")
            print(f"[INFO] Please place gurobi.lic in: {os.path.relpath(os.path.dirname(license_path))}")
    else:
        print(f"[ACT] Using existing Gurobi license: {os.path.relpath(os.environ['GRB_LICENSE_FILE'])}")

setup_gurobi_license()


class GurobiSolver(Solver):
    """Gurobi backend for exact LP/MILP solving (CPU-only)."""

    def capabilities(self) -> SolverCaps:
        return SolverCaps(False)

    def __init__(self, config: Optional[GurobiConfig] = None):
        self._cfg: GurobiConfig = config or GurobiConfig()
        if not GUROBI_AVAILABLE:
            raise RuntimeError("gurobipy is not available in this environment.")

    def compute_bounds(self, domain_obj) -> 'Bounds':
        from act.back_end.core import Bounds
        import torch
        hz = domain_obj
        n = int(hz.c.shape[0])
        p = int(hz.Gc.shape[1])
        q = int(hz.Gb.shape[1])
        c_np = hz.c.detach().cpu().numpy().astype("float64").reshape(-1)
        Gc_np = hz.Gc.detach().cpu().numpy().astype("float64")
        Gb_np = hz.Gb.detach().cpu().numpy().astype("float64")
        Ac_np = hz.Ac.detach().cpu().numpy().astype("float64")
        Ab_np = hz.Ab.detach().cpu().numpy().astype("float64")
        b_np = hz.b.detach().cpu().numpy().astype("float64").reshape(-1)
        nc = Ac_np.shape[0]
        LB = np.empty((n,), dtype=np.float64)
        UB = np.empty((n,), dtype=np.float64)
        for i in range(n):
            m = gp.Model(f"hz_dim_{i}")
            m.Params.OutputFlag = self._cfg.output_flag
            m.Params.MIPGap = self._cfg.mip_gap
            m.Params.Threads = self._cfg.threads
            if self._cfg.time_limit is not None:
                m.Params.TimeLimit = float(self._cfg.time_limit)
            xi_c = m.addMVar(p, lb=-1.0, ub=1.0, name="xi_c")
            xi_b = m.addMVar(q, vtype=GRB.BINARY, name="xi_b") if q > 0 else None
            if nc > 0:
                if xi_b is not None:
                    for r in range(nc):
                        m.addConstr(Ac_np[r] @ xi_c + Ab_np[r] @ xi_b == b_np[r])
                else:
                    for r in range(nc):
                        m.addConstr(Ac_np[r] @ xi_c == b_np[r])
            obj_c = Gc_np[i]
            obj_b = Gb_np[i] if q > 0 else np.zeros(0)
            if xi_b is not None:
                m.setObjective(obj_c @ xi_c + obj_b @ xi_b, GRB.MINIMIZE)
            else:
                m.setObjective(obj_c @ xi_c, GRB.MINIMIZE)
            m.optimize()
            LB[i] = c_np[i] + (m.ObjVal if m.Status == GRB.OPTIMAL else 0.0)
            if xi_b is not None:
                m.setObjective(obj_c @ xi_c + obj_b @ xi_b, GRB.MAXIMIZE)
            else:
                m.setObjective(obj_c @ xi_c, GRB.MAXIMIZE)
            m.optimize()
            UB[i] = c_np[i] + (m.ObjVal if m.Status == GRB.OPTIMAL else 0.0)
        dtype, device = hz.c.dtype, hz.c.device
        return Bounds(lb=torch.from_numpy(LB).to(device=device, dtype=dtype),
                      ub=torch.from_numpy(UB).to(device=device, dtype=dtype))

    def solve_batch(
        self,
        problem: "BatchLPProblem",  # noqa: F821
        timelimit: Optional[float] = None,
    ) -> "BatchLPSolution":  # noqa: F821
        """Solve a batch of N independent LPs.

        N=1: builds a single gp.Model, solves, returns BatchLPSolution[N=1].
        N>1: raises NotImplementedError — Gurobi's multi-scenario API does not
             expose truly parallel solving for varying constraint matrices.
        """
        from act.back_end.solver.solver_base import BatchLPSolution
        import torch

        if problem.N != 1:
            raise NotImplementedError(
                f"GurobiSolver.solve_batch: N={problem.N} not supported. "
                f"Gurobi does not expose a truly parallel multi-LP API for "
                f"varying constraint matrices. Use TorchLPSolver for N>1, "
                f"or constrain BaB to bab_max_batch_size=1 and "
                f"verify_lp_batched is skipped (set lp_enabled=False)."
            )

        nvars = problem.nvars
        lb = problem.lb[0].cpu().numpy().astype(np.float64)
        ub = problem.ub[0].cpu().numpy().astype(np.float64)

        env = gp.Env(empty=True)
        env.setParam("OutputFlag", self._cfg.output_flag)
        env.start()
        m = gp.Model("verify_batch_n1", env=env)
        m.Params.MIPGap = self._cfg.mip_gap
        m.Params.Threads = self._cfg.threads
        x = m.addMVar(nvars, lb=lb, ub=ub, name="x")

        # Decompose block-diagonal sparse: for N=1 the block is the full matrix.
        if problem.m_eq > 0:
            A_eq = (
                problem.A_eq_blockdiag.to_dense()[: problem.m_eq, :nvars]
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            b_eq = problem.b_eq[0].cpu().numpy().astype(np.float64)
            m.addConstr(A_eq @ x == b_eq)

        if problem.m_le > 0:
            A_le = (
                problem.A_le_blockdiag.to_dense()[: problem.m_le, :nvars]
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            b_le = problem.b_le[0].cpu().numpy().astype(np.float64)
            m.addConstr(A_le @ x <= b_le)

        obj_c = problem.obj_c[0].cpu().numpy().astype(np.float64)
        obj_const = float(problem.obj_const[0].item())
        sense = GRB.MINIMIZE if problem.sense == "min" else GRB.MAXIMIZE
        m.setObjective(obj_c @ x + obj_const, sense)

        active_timelimit = self._cfg.time_limit if self._cfg.time_limit is not None else timelimit
        if active_timelimit is not None:
            m.Params.TimeLimit = float(active_timelimit)
        m.optimize()

        dtype = problem.lb.dtype
        device = problem.lb.device

        if m.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            status = SolveStatus.SAT
            x_val = torch.as_tensor(x.X, dtype=dtype, device=device).unsqueeze(0)
            max_viol = torch.zeros(1, dtype=dtype, device=device)
        elif m.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
            status = SolveStatus.UNSAT
            x_val = torch.zeros_like(problem.lb)
            max_viol = torch.full((1,), float("inf"), dtype=dtype, device=device)
        else:
            status = SolveStatus.UNKNOWN
            x_val = torch.zeros_like(problem.lb)
            max_viol = torch.full((1,), float("nan"), dtype=dtype, device=device)

        return BatchLPSolution(
            statuses=(status,),
            x=x_val,
            max_viol=max_viol,
        )
