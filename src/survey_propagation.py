"""
Survey Propagation for H-(dis)assortative balanced K-partitions on random
d-regular graphs."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch


# ----------------------------------------------------------------------------
# Result container
# ----------------------------------------------------------------------------


@dataclass
class SPResult:
    """Lightweight container for the output of `KStateSurveyPropagation.solve`."""

    eta: torch.Tensor
    history: list[dict]
    converged: bool
    n_iter: int


# ----------------------------------------------------------------------------
# Survey propagation
# ----------------------------------------------------------------------------


class KStateSurveyPropagation:
    """
    Exact deterministic survey propagation for K-state assortative /
    disassortative partitions on homogeneous d-regular graphs."""

    # Practical cap (above this we can't materialise the eta vector).
    K_HARD_MAX = 4

    def __init__(
        self,
        K: int,
        d: int,
        H: int,
        mode: str = "assortative",
        reject_contradiction: bool = True,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
        num_threads: Optional[int] = None,
    ) -> None:
        """
        Parameters
        """
        if mode not in {"assortative", "disassortative"}:
            raise ValueError("mode must be 'assortative' or 'disassortative'.")
        if K < 2:
            raise ValueError("K must be at least 2.")
        if d < 2:
            raise ValueError("d must be at least 2.")
        if not (0 <= H <= d):
            raise ValueError(f"H must satisfy 0 <= H <= d (got H={H}, d={d}).")
        if K > self.K_HARD_MAX:
            raise ValueError(
                f"K={K} exceeds the practical cap K<={self.K_HARD_MAX}. "
                f"The bit-mask SP implementation stores a vector of size "
                f"M=2^(K^2)={1 << (K * K)} which is not feasible for K>4."
            )

        if num_threads is not None:
            torch.set_num_threads(num_threads)

        self.K = int(K)
        self.d = int(d)
        self.H = int(H)
        self.mode = mode
        self.reject_contradiction = bool(reject_contradiction)
        self.n_in = self.d - 1

        self.device = torch.device(device)
        self.dtype = dtype

        self.K2 = self.K * self.K
        self.M = 1 << self.K2
        self.dont_care_id = self.M - 1  # all-ones K x K matrix
        self.contradiction_id = 0        # all-zeros K x K matrix

        # bit p = a * K + b   indexes entry w[a, b]
        self.bit_weights = 2 ** torch.arange(
            self.K2, dtype=torch.long, device=self.device
        )

        # warning_bits[w, a, b] : binary matrix for warning id w
        ids = torch.arange(self.M, dtype=torch.long, device=self.device)
        bits = ids[:, None].bitwise_and(self.bit_weights[None, :]) != 0
        self.warning_bits = bits.reshape(self.M, self.K, self.K)

        self.eye = torch.eye(self.K, dtype=torch.bool, device=self.device)
        self.not_eye = ~self.eye
        self.eye_long = self.eye.to(torch.long)

        # Symmetry: color-permutation group S_K, used to project onto the
        # balanced sector m = (1/K, ..., 1/K).
        self.perm_ids = self._build_perm_ids()

        # Precomputed effect of a single incoming warning, indexed by
        # (warning_id, central_color).
        (
            self.warn_invalid,
            self.warn_min_inc,
            self.warn_max_inc,
        ) = self._precompute_warning_effects()

        # DP state encoding parameters.
        # Counts can reach d (full neighborhood, in fixed_point_observables);
        # the cavity update only uses d-1, but we size for d to be safe.
        self.count_base = self.d + 1               # values for min/max counts
        self.invalid_code = self.count_base ** 2   # sentinel for invalid color
        self.color_state_base = self.invalid_code + 1

        # Guard against int64 overflow in the state encoding.
        # state_id < color_state_base^K must fit in int64.
        max_state_id = self.color_state_base ** self.K
        if max_state_id >= 2 ** 62:
            raise ValueError(
                f"State-id encoding would overflow int64 for "
                f"(K={K}, d={d}). Reduce d or K."
            )

        self.base_powers = self.color_state_base ** torch.arange(
            self.K, dtype=torch.long, device=self.device
        )

    # ------------------------------------------------------------------
    # Warning <-> integer id
    # ------------------------------------------------------------------

    def _matrix_to_ids(self, mats: torch.Tensor) -> torch.Tensor:
        """Convert binary K x K matrices of shape [..., K, K] to int ids."""
        flat = mats.reshape(-1, self.K2).to(torch.long)
        return (flat * self.bit_weights[None, :]).sum(dim=-1)

    def warning_matrix(self, warning_id: int) -> torch.Tensor:
        """Return the K x K matrix associated with `warning_id` (on CPU)."""
        return self.warning_bits[warning_id].detach().cpu().to(torch.int64)

    # ------------------------------------------------------------------
    # Color-permutation symmetry
    # ------------------------------------------------------------------

    def _build_perm_ids(self) -> torch.Tensor:
        """
        Build a [K!, M] table such that perm_ids[p, w] = id(pi . w), where
        (pi . w)[a, b] = w[pi^{-1}(a), pi^{-1}(b)]. This is a simultaneous
        relabeling of both color indices, which is the right symmetry for
        the balanced case m = (1/K, ..., 1/K).
        """
        all_ids = []
        for p in itertools.permutations(range(self.K)):
            inv = [0] * self.K
            for old, new in enumerate(p):
                inv[new] = old
            inv = torch.tensor(inv, dtype=torch.long, device=self.device)
            mats = self.warning_bits[:, inv, :][:, :, inv]
            all_ids.append(self._matrix_to_ids(mats))
        return torch.stack(all_ids, dim=0)

    @torch.no_grad()
    def symmetrize(self, eta: torch.Tensor) -> torch.Tensor:
        """Average eta over the S_K color-relabeling action and normalize."""
        eta = eta.to(device=self.device, dtype=self.dtype)
        out = torch.zeros_like(eta)
        contribution = eta / self.perm_ids.shape[0]
        for pids in self.perm_ids:
            out.index_add_(0, pids, contribution)
        out = torch.clamp(out, min=0.0)
        total = out.sum()
        if total <= 0 or not torch.isfinite(total):
            raise RuntimeError("Invalid survey mass after symmetrization.")
        return out / total

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    @torch.no_grad()
    def init_small_noise(self, eps: float = 1e-3) -> torch.Tensor:
        """
        Small-noise reconstruction initialization. The boundary is mostly
        made of *node-conclusive row warnings* (which fix x_i = a and leave
        x_j free), with a probability eps of don't-care warnings mixed in.

        - With probability  eps:   the don't-care warning (all K^2 ones).
        - With probability  1-eps: a row warning w[c, b] = 1[c = a],
                                   uniform over a in {1, ..., K}.

        eps small => strong reconstruction signal (test for frozen 1RSB).
        eps = 1   => trivial don't-care initialization (will trivially be
                     a fixed point if it is one).
        """
        if not (0.0 <= eps <= 1.0):
            raise ValueError("eps must satisfy 0 <= eps <= 1.")

        eta = torch.zeros(self.M, dtype=self.dtype, device=self.device)

        # don't-care
        eta[self.dont_care_id] = float(eps)

        # node-conclusive row warnings
        row_ids = []
        for a in range(self.K):
            mat = torch.zeros((self.K, self.K), dtype=torch.bool, device=self.device)
            mat[a, :] = True
            wid = self._matrix_to_ids(mat[None, :, :])[0]
            row_ids.append(wid)
        row_ids = torch.stack(row_ids)
        eta[row_ids] = (1.0 - float(eps)) / self.K

        return self.symmetrize(eta)

    @torch.no_grad()
    def init_dont_care(self) -> torch.Tensor:
        """Trivial all-don't-care initialization (the RS fixed point)."""
        eta = torch.zeros(self.M, dtype=self.dtype, device=self.device)
        eta[self.dont_care_id] = 1.0
        return eta

    # ------------------------------------------------------------------
    # Per-warning sufficient statistics
    # ------------------------------------------------------------------

    def _precompute_warning_effects(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        For every incoming warning w and every central color a, compute:

            invalid[w, a]:  True iff no incoming neighbor color c is allowed
                            when the central node has x_i = a.

            min_inc[w, a]:  contribution to the minimum possible number of
                            same-color incoming neighbors (1 if same color is
                            forced, 0 otherwise).

            max_inc[w, a]:  contribution to the maximum possible number of
                            same-color incoming neighbors (1 if same color is
                            allowed, 0 otherwise).

        Convention: in w[c, a], c = color of incoming neighbor k,
                                 a = color of central node i.
        """
        W = self.warning_bits  # [M, K, K]

        # same_allowed[w, a] = w[a, a]
        same_allowed = torch.diagonal(W, dim1=-2, dim2=-1)  # [M, K]

        # other_allowed[w, a] : exists c != a with w[c, a] = 1
        other_allowed = (W & self.not_eye[None, :, :]).any(dim=-2)  # [M, K]

        invalid = ~(same_allowed | other_allowed)
        min_inc = (same_allowed & ~other_allowed).to(torch.long)
        max_inc = same_allowed.to(torch.long)

        return invalid, min_inc, max_inc

    # ------------------------------------------------------------------
    # DP state encoding
    # ------------------------------------------------------------------

    def _encode_states(
        self,
        min_same: torch.Tensor,
        max_same: torch.Tensor,
        invalid: torch.Tensor,
    ) -> torch.Tensor:
        """Encode arrays of shape [S, K] into integer state ids."""
        color_codes = min_same * self.count_base + max_same
        color_codes = torch.where(
            invalid,
            torch.full_like(color_codes, self.invalid_code),
            color_codes,
        )
        return (color_codes * self.base_powers[None, :]).sum(dim=1)

    def _decode_states(
        self,
        state_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode state ids back into (min_same, max_same, invalid) arrays."""
        color_codes = (
            state_ids[:, None] // self.base_powers[None, :]
        ) % self.color_state_base
        invalid = color_codes == self.invalid_code
        valid_codes = torch.where(
            invalid, torch.zeros_like(color_codes), color_codes
        )
        min_same = valid_codes // self.count_base
        max_same = valid_codes % self.count_base
        return min_same, max_same, invalid

    # ------------------------------------------------------------------
    # Convert final DP state to outgoing warning (cavity / non-cavity)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _state_to_warning_ids(self, state_ids: torch.Tensor) -> torch.Tensor:
        """
        Map a DP state (over d-1 incoming neighbors) to the *outgoing*
        warning that this center would send to its (d-th) neighbor j.
        """
        min_same, max_same, invalid = self._decode_states(state_ids)
        eq_ab = self.eye_long[None, :, :]  # [1, K, K]

        if self.mode == "assortative":
            allowed = (~invalid[:, :, None]) & (
                max_same[:, :, None] + eq_ab >= self.H
            )
        else:  # disassortative
            allowed = (~invalid[:, :, None]) & (
                min_same[:, :, None] + eq_ab < self.H
            )
        return self._matrix_to_ids(allowed)

    # ------------------------------------------------------------------
    # Exact SP update T : eta -> T[eta]
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _run_dp(
        self,
        eta: torch.Tensor,
        n_steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Iterate the sufficient-statistic DP for `n_steps` incoming neighbors
        drawn iid from eta. Returns (state_ids, state_probs).
        """
        eta = eta / eta.sum()
        support = torch.nonzero(eta > 0, as_tuple=False).flatten()
        support_probs = eta[support]

        # Initial DP state (no incoming neighbor yet): zeros, valid.
        state_ids = torch.zeros(1, dtype=torch.long, device=self.device)
        state_probs = torch.ones(1, dtype=self.dtype, device=self.device)

        for _ in range(n_steps):
            min_same, max_same, invalid = self._decode_states(state_ids)
            S = state_ids.numel()
            L = support.numel()

            w_invalid = self.warn_invalid[support]  # [L, K]
            w_min_inc = self.warn_min_inc[support]  # [L, K]
            w_max_inc = self.warn_max_inc[support]  # [L, K]

            new_invalid = invalid[:, None, :] | w_invalid[None, :, :]
            new_min = min_same[:, None, :] + w_min_inc[None, :, :]
            new_max = max_same[:, None, :] + w_max_inc[None, :, :]

            flat_state_ids = self._encode_states(
                new_min.reshape(S * L, self.K),
                new_max.reshape(S * L, self.K),
                new_invalid.reshape(S * L, self.K),
            )
            flat_probs = (
                state_probs[:, None] * support_probs[None, :]
            ).reshape(S * L)

            unique_ids, inverse = torch.unique(
                flat_state_ids, sorted=False, return_inverse=True
            )
            new_probs = torch.zeros(
                unique_ids.numel(), dtype=self.dtype, device=self.device
            )
            new_probs.index_add_(0, inverse, flat_probs)

            state_ids = unique_ids
            state_probs = new_probs

        return state_ids, state_probs

    @torch.no_grad()
    def sp_update_exact(self, eta: torch.Tensor) -> torch.Tensor:
        """Exact deterministic SP update T[eta] on a d-regular graph."""
        eta = eta.to(device=self.device, dtype=self.dtype)
        state_ids, state_probs = self._run_dp(eta, n_steps=self.n_in)
        out_ids = self._state_to_warning_ids(state_ids)

        eta_new = torch.zeros(self.M, dtype=self.dtype, device=self.device)
        eta_new.index_add_(0, out_ids, state_probs)

        if self.reject_contradiction:
            eta_new[self.contradiction_id] = 0.0

        total = eta_new.sum()
        if total <= 0 or not torch.isfinite(total):
            # All mass on the contradiction warning => problem is locally
            # unsatisfiable under this initialization. Return a sentinel
            # delta on the contradiction so the solver can flag UNSAT.
            eta_new = torch.zeros_like(eta_new)
            eta_new[self.contradiction_id] = 1.0
            return eta_new
        eta_new = eta_new / total

        # Enforce the balanced color-symmetric sector.
        return self.symmetrize(eta_new)

    # ------------------------------------------------------------------
    # Fixed-point solver
    # ------------------------------------------------------------------

    @torch.no_grad()
    def solve(
        self,
        eps: float = 1e-3,
        gamma: float = 0.15,
        max_iter: int = 300,
        tol: float = 2e-8,
        verbose: bool = True,
        log_every: int = 10,
        eta_init: Optional[torch.Tensor] = None,
        on_iter: Optional[callable] = None,
    ) -> SPResult:
        """
        Damped exact fixed-point iteration

            eta_{t+1} = (1 - gamma) * eta_t + gamma * T(eta_t)

        gamma in (0, 1].  gamma -> 0 is heavy damping; gamma = 1 is no damping.
        """
        if not (0.0 < gamma <= 1.0):
            raise ValueError("gamma must satisfy 0 < gamma <= 1.")

        eta = (
            eta_init.to(device=self.device, dtype=self.dtype)
            if eta_init is not None
            else self.init_small_noise(eps=eps)
        )

        history: list[dict] = []
        converged = False
        n_iter = 0

        for it in range(max_iter):
            Teta = self.sp_update_exact(eta)
            eta_next = (1.0 - gamma) * eta + gamma * Teta
            eta_next = self.symmetrize(eta_next)

            residual = 0.5 * torch.abs(eta_next - eta).sum().item()
            eta_dc = eta_next[self.dont_care_id].item()
            support = int((eta_next > 1e-14).sum().item())

            info = {
                "iter": it,
                "residual_l1_half": residual,
                "eta_dont_care": eta_dc,
                "support": support,
            }
            history.append(info)
            if on_iter is not None:
                on_iter(it, info)

            if verbose and (it % log_every == 0 or residual < tol):
                print(
                    f"it={it:4d}  residual={residual:.4e}  "
                    f"eta_dc={eta_dc:.12g}  support={support}",
                    flush=True,
                )

            eta = eta_next
            n_iter = it + 1

            if residual < tol:
                converged = True
                break

        return SPResult(
            eta=eta.detach().cpu(),
            history=history,
            converged=converged,
            n_iter=n_iter,
        )

    # ------------------------------------------------------------------
    # Observables on the fixed point
    # ------------------------------------------------------------------

    @torch.no_grad()
    def dont_care_probability(self, eta: torch.Tensor) -> float:
        """eta restricted to the don't-care warning."""
        return float(eta[self.dont_care_id])

    @torch.no_grad()
    def fixed_point_observables(self, eta: torch.Tensor) -> dict:
        """
        Compute diagnostic observables on the SP fixed point.
        """
        eta = eta.to(device=self.device, dtype=self.dtype)
        state_ids, state_probs = self._run_dp(eta, n_steps=self.d)
        min_same, max_same, invalid = self._decode_states(state_ids)

        if self.mode == "assortative":
            allowed = (~invalid) & (max_same >= self.H)
        else:
            allowed = (~invalid) & (min_same < self.H)

        n_allowed = allowed.sum(dim=1).to(self.dtype)  # [S]

        prob_unsat = float((state_probs * (n_allowed == 0).to(self.dtype)).sum())
        prob_frozen = float((state_probs * (n_allowed == 1).to(self.dtype)).sum())
        mean_allowed = float((state_probs * n_allowed).sum())

        return {
            "prob_unsat": prob_unsat,
            "frozen_fraction": prob_frozen,
            "mean_allowed_colors": mean_allowed,
        }

    @torch.no_grad()
    def frozen_fraction(self, eta: torch.Tensor) -> float:
        """Convenience wrapper around `fixed_point_observables`."""
        return self.fixed_point_observables(eta)["frozen_fraction"]

    @torch.no_grad()
    def nonzero_warnings(
        self, eta: torch.Tensor, threshold: float = 1e-12
    ) -> list[dict]:
        """
        Return the nonzero entries of eta as a list of dicts:
            [{"id": int, "prob": float, "matrix": [[0/1, ...], ...]}, ...]
        """
        eta_cpu = eta.detach().cpu()
        nz = torch.nonzero(eta_cpu > threshold, as_tuple=False).flatten().tolist()
        out = []
        for wid in nz:
            mat = self.warning_matrix(wid).numpy().astype(int).tolist()
            out.append({"id": int(wid), "prob": float(eta_cpu[wid]), "matrix": mat})
        out.sort(key=lambda r: -r["prob"])
        return out


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def make_output_path(out_dir: str, K: int, d: int, H: int, mode: str, seed: int) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out / f"sp_K{K}_d{d}_H{H}_{mode}_seed{seed}.json"


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Survey propagation for H-(dis)assortative balanced K-partitions "
            "on random d-regular graphs."
        )
    )
    # problem
    p.add_argument("--K", type=int, required=True, help="Number of groups (2..4).")
    p.add_argument("--d", type=int, required=True, help="Graph regularity (>=2).")
    p.add_argument("--H", type=int, required=True, help="Assortativity threshold.")
    p.add_argument(
        "--mode",
        choices=("assortative", "disassortative"),
        required=True,
    )
    # solver
    p.add_argument("--eps", type=float, default=1e-3,
                   help="Init don't-care mass for small-noise reconstruction.")
    p.add_argument("--gamma", type=float, default=0.15,
                   help="Step size in [0,1] (1 = no damping).")
    p.add_argument("--max-iter", type=int, default=500)
    p.add_argument("--tol", type=float, default=2e-8)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0,
                   help="Recorded in metadata. Deterministic init is used.")
    p.add_argument(
        "--keep-contradiction", action="store_true",
        help="Reproduce the literal eq. (135) iteration that lets mass leak "
             "onto the all-zero (contradiction) warning. Default is to "
             "reject contradictions and renormalise on satisfiable warnings, "
             "which is the standard SP convention needed for frozen-1RSB "
             "analysis.",
    )
    # runtime
    p.add_argument("--device", default="cpu")
    p.add_argument("--num-threads", type=int, default=None)
    # output
    p.add_argument("--out-dir", default="results/wp")
    p.add_argument("--save-full-eta", action="store_true",
                   help="Save the full eta vector (otherwise only nonzero entries).")
    p.add_argument("--quiet", action="store_true")
    # wandb
    p.add_argument("--use-wandb", action="store_true")
    p.add_argument("--wandb-project", default="assortative-partitions-sp")
    p.add_argument("--wandb-run-name", default=None)
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    verbose = not args.quiet

    config = dict(
        K=args.K, d=args.d, H=args.H, mode=args.mode,
        eps=args.eps, gamma=args.gamma, max_iter=args.max_iter, tol=args.tol,
        reject_contradiction=not args.keep_contradiction,
        seed=args.seed, device=args.device,
    )

    # ----- wandb (optional) -----
    wandb_run = None
    if args.use_wandb:
        try:
            import wandb
            run_name = args.wandb_run_name or (
                f"sp_K{args.K}_d{args.d}_H{args.H}_{args.mode}_seed{args.seed}"
            )
            wandb_run = wandb.init(
                project=args.wandb_project, name=run_name, config=config
            )
        except ImportError:
            print("[warn] --use-wandb requested but wandb not installed; skipping.",
                  file=sys.stderr)

    def on_iter(it: int, info: dict) -> None:
        if wandb_run is not None:
            wandb_run.log(info, step=it)

    # ----- solve -----
    out_path = make_output_path(
        args.out_dir, args.K, args.d, args.H, args.mode, args.seed
    )

    record = {
        "config": config,
        "output_path": str(out_path),
        "git_commit": _git_commit_or_none(),
        "started_at": time.time(),
    }

    t0 = time.time()
    try:
        sp = KStateSurveyPropagation(
            K=args.K, d=args.d, H=args.H, mode=args.mode,
            reject_contradiction=not args.keep_contradiction,
            device=args.device, num_threads=args.num_threads,
        )
        result = sp.solve(
            eps=args.eps,
            gamma=args.gamma,
            max_iter=args.max_iter,
            tol=args.tol,
            verbose=verbose,
            log_every=args.log_every,
            on_iter=on_iter,
        )
        eta = result.eta

        record.update({
            "converged": result.converged,
            "n_iter": result.n_iter,
            "residual_final": result.history[-1]["residual_l1_half"]
                if result.history else None,
            "eta_dont_care": sp.dont_care_probability(eta),
            "eta_contradiction": float(eta[sp.contradiction_id]),
            "support_size": int((eta > 1e-14).sum().item()),
            **sp.fixed_point_observables(eta),
            "history": result.history,
            "eta_nonzero": sp.nonzero_warnings(eta),
        })
        if args.save_full_eta:
            record["eta_full"] = eta.tolist()

        if verbose:
            print(
                f"\n[done] converged={result.converged} "
                f"n_iter={result.n_iter} "
                f"eta_dc={record['eta_dont_care']:.6g} "
                f"frozen={record['frozen_fraction']:.6f} "
                f"unsat={record['prob_unsat']:.6f}",
                flush=True,
            )
    except Exception as exc:  # noqa: BLE001 -- top-level reporting
        record["error"] = repr(exc)
        record["error_type"] = type(exc).__name__
        if verbose:
            print(f"[error] {exc!r}", file=sys.stderr, flush=True)
    finally:
        record["wall_time_s"] = time.time() - t0
        record["finished_at"] = time.time()
        with open(out_path, "w") as f:
            json.dump(record, f, indent=2)
        if verbose:
            print(f"[saved] {out_path}", flush=True)
        if wandb_run is not None:
            wandb_run.summary.update({
                k: record[k] for k in (
                    "converged", "n_iter", "eta_dont_care",
                    "eta_contradiction", "frozen_fraction", "prob_unsat",
                    "mean_allowed_colors", "support_size", "wall_time_s",
                ) if k in record
            })
            wandb_run.finish()

    return 0 if "error" not in record else 1


def _git_commit_or_none() -> Optional[str]:
    try:
        import subprocess
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:  # noqa: BLE001
        return None


if __name__ == "__main__":
    sys.exit(main())