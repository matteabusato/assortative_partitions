from __future__ import annotations

import itertools
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import torch


@dataclass
class SPResult:
    eta: torch.Tensor
    history: list[dict]


class KStateSurveyPropagation:
    """
    Exact deterministic Survey Propagation for K-state assortative/disassortative
    threshold partitions on homogeneous d-regular graphs.

    Warning convention:
        w[a,b] = 1 iff the oriented edge assignment (x_i=a, x_j=b)
        is allowed by node i in the cavity graph i -> j.

    Important:
        The "don't care" warning is not always the all-one matrix.

        It is the model-dependent neutral warning produced when the other
        d-1 incoming messages impose no additional restriction.

        For disassortative H=1, i.e. coloring:

            w0[a,b] = 1[a != b].

        For the binary paper case, this reproduces the usual all-one
        don't-care warning when appropriate.

    SP complexity:
        Phi_SP = log Z_node - d/2 log Z_edge

    Damping convention:
        eta_next = damping * eta_old + (1 - damping) * T(eta_old)
    """

    def __init__(
        self,
        K: int,
        d: int,
        H: int,
        mode: str = "disassortative",
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
        num_threads: Optional[int] = None,
        remove_contradictions: bool = True,
    ):
        if mode not in {"assortative", "disassortative"}:
            raise ValueError("mode must be 'assortative' or 'disassortative'.")

        if K < 2:
            raise ValueError("K must be at least 2.")
        if d < 1:
            raise ValueError("d must be at least 1.")
        if H < 0:
            raise ValueError("H must be non-negative.")

        if num_threads is not None:
            torch.set_num_threads(num_threads)

        self.K = int(K)
        self.d = int(d)
        self.H = int(H)
        self.mode = mode
        self.n_in = self.d - 1
        self.remove_contradictions = bool(remove_contradictions)

        self.device = torch.device(device)
        self.dtype = dtype

        self.K2 = self.K * self.K
        if self.K2 >= 62:
            raise ValueError("This raw bit-mask implementation requires K^2 < 62.")

        self.M = 1 << self.K2

        # Raw special ids.
        self.zero_id = 0
        self.all_one_id = self.M - 1

        # Bit index p = a*K + b.
        self.bit_weights = 2 ** torch.arange(
            self.K2, dtype=torch.long, device=self.device
        )

        ids = torch.arange(self.M, dtype=torch.long, device=self.device)
        bits = (ids[:, None].bitwise_and(self.bit_weights[None, :]) != 0)
        self.warning_bits = bits.reshape(self.M, self.K, self.K)

        self.eye = torch.eye(self.K, dtype=torch.bool, device=self.device)
        self.not_eye = ~self.eye
        self.eye_long = self.eye.to(torch.long)

        # Model-dependent neutral / don't-care warning.
        self.dont_care_id = self._build_neutral_warning_id()

        # Color-permutation symmetry.
        self.perm_ids = self._build_perm_ids()

        # Precomputed warning effects.
        self.warn_invalid, self.warn_min_inc, self.warn_max_inc = (
            self._precompute_warning_effects()
        )

        # DP state encoding.
        self.count_base = self.n_in + 1
        self.invalid_code = self.count_base * self.count_base
        self.color_state_base = self.invalid_code + 1

        self.base_powers = self.color_state_base ** torch.arange(
            self.K, dtype=torch.long, device=self.device
        )

    # ------------------------------------------------------------------
    # Warning representation
    # ------------------------------------------------------------------

    def _matrix_to_ids(self, mats: torch.Tensor) -> torch.Tensor:
        """
        Convert binary K x K matrices to integer warning ids.

        mats shape:
            [..., K, K]
        """
        flat = mats.reshape(-1, self.K2).to(torch.long)
        return (flat * self.bit_weights[None, :]).sum(dim=-1)

    def warning_matrix(self, warning_id: int) -> torch.Tensor:
        """
        Return the K x K binary matrix associated with warning_id.
        """
        return self.warning_bits[warning_id].detach().cpu().to(torch.int64)

    @torch.no_grad()
    def _canonicalize_mats(self, mats: torch.Tensor) -> torch.Tensor:
        """
        Collapse raw K x K relations to the canonical warning with the same
        effect on the threshold dynamics.

        For fixed receiver/central color b, only two pieces of information matter:

            same_allowed[b]  = whether sender color a=b is possible
            other_allowed[b] = whether at least one sender color a!=b is possible

        The identity of the other color does not matter for the threshold rule.
        """
        mats = mats.to(dtype=torch.bool, device=self.device)

        same_allowed = torch.diagonal(mats, dim1=-2, dim2=-1)  # [..., K]
        other_allowed = (mats & self.not_eye).any(dim=-2)      # [..., K]

        canon = (
            (same_allowed[..., None, :] & self.eye)
            | (other_allowed[..., None, :] & self.not_eye)
        )

        return canon

    @torch.no_grad()
    def _build_neutral_warning_id(self) -> int:
        """
        Build the model-dependent neutral / don't-care warning.

        Disassortative:
            Need #same < H.
            With free other neighbors, the minimum contribution from them is 0.
            Therefore pair (a,b) is allowed iff 1[a=b] < H.

            For H=1:
                allowed iff a != b,
            which is exactly the coloring no-warning relation.

        Assortative:
            Need #same >= H.
            With free other neighbors, the maximum contribution from them is d-1.
            Therefore pair (a,b) is allowed iff d-1 + 1[a=b] >= H.
        """
        eq = self.eye_long  # [K,K], axes [a,b]

        if self.mode == "disassortative":
            allowed = eq < self.H
        else:
            allowed = (self.d - 1 + eq) >= self.H

        allowed = allowed.to(torch.bool)
        allowed = self._canonicalize_mats(allowed[None, :, :])[0]

        return int(self._matrix_to_ids(allowed[None, :, :])[0].item())

    def dont_care_matrix(self) -> torch.Tensor:
        return self.warning_matrix(self.dont_care_id)

    # ------------------------------------------------------------------
    # Color-permutation symmetry
    # ------------------------------------------------------------------

    def _build_perm_ids(self) -> torch.Tensor:
        """
        perm_ids[p,w] = id(pi.w), where

            (pi.w)[a,b] = w[pi^{-1}(a), pi^{-1}(b)].

        Then canonicalize again.
        """
        all_ids = []

        for p in itertools.permutations(range(self.K)):
            inv = [0] * self.K
            for old, new in enumerate(p):
                inv[new] = old

            inv = torch.tensor(inv, dtype=torch.long, device=self.device)

            mats = self.warning_bits[:, inv, :][:, :, inv]
            mats = self._canonicalize_mats(mats)

            all_ids.append(self._matrix_to_ids(mats))

        return torch.stack(all_ids, dim=0)

    @torch.no_grad()
    def normalize_valid_survey(self, eta: torch.Tensor) -> torch.Tensor:
        """
        Remove contradictory all-zero warnings and normalize.

        The all-zero warning means no pair (x_i,x_j) is compatible. It is
        a local contradiction, not a frozen warning. For the physical SP branch
        over valid WP fixed points, we condition it away.
        """
        eta = eta.to(device=self.device, dtype=self.dtype).clone()

        if self.remove_contradictions:
            eta[self.zero_id] = 0.0

        total = eta.sum()

        if total <= 0 or not torch.isfinite(total):
            raise RuntimeError(
                "No valid survey mass remains after removing contradictory warnings."
            )

        return eta / total

    @torch.no_grad()
    def symmetrize(self, eta: torch.Tensor) -> torch.Tensor:
        """
        Enforce eta(w) = eta(pi.w) for all color permutations pi.
        """
        eta = eta.to(device=self.device, dtype=self.dtype)
        out = torch.zeros_like(eta)

        contribution = eta / self.perm_ids.shape[0]

        for pids in self.perm_ids:
            out.index_add_(0, pids, contribution)

        out = torch.clamp(out, min=0.0)

        return self.normalize_valid_survey(out)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    @torch.no_grad()
    def init_small_noise(self, eps: float = 1e-3) -> torch.Tensor:
        """
        Small-noise reconstruction initialization.

        eps:
            mass on the model-dependent neutral / don't-care warning.

        1-eps:
            uniformly over color-conclusive warnings.

        For H=1 disassortative coloring:
            neutral warning is off-diagonal,
            conclusive warning for color a forbids receiver color a.
        """
        if not (0.0 <= eps <= 1.0):
            raise ValueError("eps must satisfy 0 <= eps <= 1.")

        eta = torch.zeros(self.M, dtype=self.dtype, device=self.device)

        # Correct model-dependent don't-care.
        eta[self.dont_care_id] = float(eps)

        neutral_mat = self.warning_bits[self.dont_care_id]

        row_ids = []
        for a in range(self.K):
            mat = torch.zeros((self.K, self.K), dtype=torch.bool, device=self.device)

            # Fix sender color x_i=a while respecting neutral compatibility.
            mat[a, :] = neutral_mat[a, :]

            # Convert to canonical warning.
            mat = self._canonicalize_mats(mat[None, :, :])[0]

            wid = self._matrix_to_ids(mat[None, :, :])[0]
            row_ids.append(wid)

        row_ids = torch.stack(row_ids)

        row_mass = torch.full(
            (self.K,),
            (1.0 - float(eps)) / self.K,
            dtype=self.dtype,
            device=self.device,
        )

        eta.index_add_(0, row_ids, row_mass)

        eta = self.normalize_valid_survey(eta)

        return self.symmetrize(eta)

    # ------------------------------------------------------------------
    # WP sufficient statistics
    # ------------------------------------------------------------------

    def _precompute_warning_effects(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        For each incoming warning w and each central color a, compute:

        invalid[w,a]:
            True if no incoming neighbor color c is allowed when x_i=a.

        min_inc[w,a]:
            contribution to the minimum possible number of same-color incoming
            neighbors.

        max_inc[w,a]:
            contribution to the maximum possible number of same-color incoming
            neighbors.

        Incoming warning w is queried as w[c,a], where:
            c = color of incoming neighbor k,
            a = color of central node i.

        For fixed a:
            same_allowed  = w[a,a]
            other_allowed = exists c != a such that w[c,a] = 1

        Then:
            invalid      if not same_allowed and not other_allowed
            forced same  if same_allowed and not other_allowed
            optional     if same_allowed and other_allowed
            forced other if not same_allowed and other_allowed
        """
        W = self.warning_bits  # [M,K,K]

        same_allowed = torch.diagonal(W, dim1=-2, dim2=-1)  # [M,K]
        other_allowed = (W & self.not_eye[None, :, :]).any(dim=-2)  # [M,K]

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
        """
        Encode arrays of shape [S,K] into integer state ids.
        """
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
        """
        Decode integer state ids into:
            min_same, max_same, invalid
        each of shape [S,K].
        """
        color_codes = (
            state_ids[:, None] // self.base_powers[None, :]
        ) % self.color_state_base

        invalid = color_codes == self.invalid_code

        valid_codes = torch.where(
            invalid,
            torch.zeros_like(color_codes),
            color_codes,
        )

        min_same = valid_codes // self.count_base
        max_same = valid_codes % self.count_base

        return min_same, max_same, invalid

    # ------------------------------------------------------------------
    # DP over incoming warnings
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _dp_states_after_n_inputs(
        self,
        eta: torch.Tensor,
        n_inputs: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Distribution over sufficient-statistic states after n_inputs iid
        incoming warnings drawn from eta.

        Chunks the support dimension to bound peak memory: the broadcast
        intermediates [S, L, K] and flat arrays of size S*L are otherwise
        the dominant cost at K=4, where the support can reach M = 2^16.
        Per chunk we dedupe states locally, then merge into a running
        (ids, probs) pair via a second dedupe to keep state counts bounded.
        """
        eta = self.normalize_valid_survey(eta)

        support = torch.nonzero(eta > 0, as_tuple=False).flatten()
        support_probs = eta[support]
        L = support.numel()

        w_invalid = self.warn_invalid[support]   # [L,K]
        w_min_inc = self.warn_min_inc[support]   # [L,K]
        w_max_inc = self.warn_max_inc[support]   # [L,K]

        state_ids = torch.zeros(1, dtype=torch.long, device=self.device)
        state_probs = torch.ones(1, dtype=self.dtype, device=self.device)

        # Rough byte cost per (state, warning) pair in the inner intermediates.
        # Conservative estimate covering the [S,c,K] bool + 3 longs and the
        # flat [S*c] id + prob arrays. ~256 MB budget per chunk.
        bytes_per_pair = 32 * self.K + 32
        target_bytes = 256 * 1024 * 1024

        for _ in range(n_inputs):
            min_same, max_same, invalid = self._decode_states(state_ids)
            S = state_ids.numel()

            chunk = max(1, target_bytes // max(1, S * bytes_per_pair))
            chunk = min(chunk, L)

            running_ids: Optional[torch.Tensor] = None
            running_probs: Optional[torch.Tensor] = None

            for ls in range(0, L, chunk):
                le = min(ls + chunk, L)
                c = le - ls

                new_invalid = invalid[:, None, :] | w_invalid[ls:le][None, :, :]
                new_min = min_same[:, None, :] + w_min_inc[ls:le][None, :, :]
                new_max = max_same[:, None, :] + w_max_inc[ls:le][None, :, :]

                chunk_flat_ids = self._encode_states(
                    new_min.reshape(S * c, self.K),
                    new_max.reshape(S * c, self.K),
                    new_invalid.reshape(S * c, self.K),
                )
                chunk_flat_probs = (
                    state_probs[:, None] * support_probs[ls:le][None, :]
                ).reshape(S * c)

                chunk_ids, inverse = torch.unique(
                    chunk_flat_ids,
                    sorted=False,
                    return_inverse=True,
                )
                chunk_probs = torch.zeros(
                    chunk_ids.numel(),
                    dtype=self.dtype,
                    device=self.device,
                )
                chunk_probs.index_add_(0, inverse, chunk_flat_probs)

                if running_ids is None:
                    running_ids = chunk_ids
                    running_probs = chunk_probs
                else:
                    merged_ids = torch.cat([running_ids, chunk_ids])
                    merged_probs = torch.cat([running_probs, chunk_probs])
                    running_ids, inv = torch.unique(
                        merged_ids,
                        sorted=False,
                        return_inverse=True,
                    )
                    running_probs = torch.zeros(
                        running_ids.numel(),
                        dtype=self.dtype,
                        device=self.device,
                    )
                    running_probs.index_add_(0, inv, merged_probs)

            state_ids = running_ids
            state_probs = running_probs

        return state_ids, state_probs

    # ------------------------------------------------------------------
    # Convert final DP state to outgoing warning
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _state_to_warning_ids(self, state_ids: torch.Tensor) -> torch.Tensor:
        """
        Convert final DP states into outgoing warning ids.

        For outgoing pair (a,b):
            a = x_i,
            b = x_j.

        Neighbor j contributes one same-color neighbor iff b=a.

        Assortative:
            exists compatible assignment iff
                max_same[a] + 1[b=a] >= H.

        Disassortative:
            exists compatible assignment iff
                min_same[a] + 1[b=a] < H.
        """
        min_same, max_same, invalid = self._decode_states(state_ids)

        eq_ab = self.eye_long[None, :, :]  # [1,K,K], axes [a,b]

        if self.mode == "assortative":
            allowed = (~invalid[:, :, None]) & (
                max_same[:, :, None] + eq_ab >= self.H
            )
        else:
            allowed = (~invalid[:, :, None]) & (
                min_same[:, :, None] + eq_ab < self.H
            )

        allowed = self._canonicalize_mats(allowed)

        return self._matrix_to_ids(allowed)

    # ------------------------------------------------------------------
    # Exact SP update
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sp_update_exact(self, eta: torch.Tensor) -> torch.Tensor:
        """
        Exact deterministic homogeneous SP update.

        Computes:

            eta_new(w) proportional to
                P[F(w_1,...,w_{d-1}) = w and F != contradiction].

        The contradiction is the all-zero warning and is conditioned away.
        """
        state_ids, state_probs = self._dp_states_after_n_inputs(eta, self.n_in)

        out_ids = self._state_to_warning_ids(state_ids)

        eta_new = torch.zeros(self.M, dtype=self.dtype, device=self.device)

        if self.remove_contradictions:
            valid = out_ids != self.zero_id
            eta_new.index_add_(0, out_ids[valid], state_probs[valid])
        else:
            eta_new.index_add_(0, out_ids, state_probs)

        eta_new = self.normalize_valid_survey(eta_new)

        return self.symmetrize(eta_new)

    # ------------------------------------------------------------------
    # Fixed-point solver
    # ------------------------------------------------------------------

    @torch.no_grad()
    def solve(
        self,
        eps: float = 1e-3,
        damping: float = 0.85,
        max_iter: int = 500,
        tol: float = 2e-8,
        verbose: bool = True,
    ) -> SPResult:
        """
        Damped exact fixed-point iteration.

        Use eps=1e-3 for the small-noise reconstruction test.
        Use eps close to 1 only if you want to test stability of the trivial
        no-warning fixed point.
        """
        if not (0.0 <= damping < 1.0):
            raise ValueError("damping must satisfy 0 <= damping < 1.")

        eta = self.init_small_noise(eps=eps)
        history: list[dict] = []

        for it in range(max_iter):
            Teta = self.sp_update_exact(eta)

            eta_next = damping * eta + (1.0 - damping) * Teta

            eta_next = self.symmetrize(eta_next)
            eta_next = self.normalize_valid_survey(eta_next)

            residual = 0.5 * torch.abs(eta_next - eta).sum().item()
            eta_dc = eta_next[self.dont_care_id].item()
            eta_all_one = eta_next[self.all_one_id].item()
            eta_zero = eta_next[self.zero_id].item()
            support = int((eta_next > 1e-14).sum().item())

            history.append(
                {
                    "iter": it,
                    "residual_l1_half": residual,
                    "eta_dont_care": eta_dc,
                    "eta_all_one": eta_all_one,
                    "eta_zero": eta_zero,
                    "support": support,
                }
            )

            if verbose and (it % 10 == 0 or residual < tol):
                print(
                    f"it={it:4d}  "
                    f"residual={residual:.4e}  "
                    f"eta_dc={eta_dc:.12g}  "
                    f"eta_all_one={eta_all_one:.12g}  "
                    f"eta_zero={eta_zero:.12g}  "
                    f"support={support}"
                )

            eta = eta_next

            if residual < tol:
                break

        return SPResult(eta=eta.detach().cpu(), history=history)

    # ------------------------------------------------------------------
    # SP complexity: node and edge terms
    # ------------------------------------------------------------------

    @torch.no_grad()
    def node_compatibility_probability(self, eta: torch.Tensor) -> float:
        """
        Compute Z_node.

        Draw d incoming warnings independently from eta. Z_node is the
        probability that there exists at least one central color a and allowed
        neighbor colors satisfying the local threshold constraint.
        """
        eta = self.normalize_valid_survey(eta)

        state_ids, state_probs = self._dp_states_after_n_inputs(eta, self.d)

        min_same, max_same, invalid = self._decode_states(state_ids)

        if self.mode == "assortative":
            compatible = ((~invalid) & (max_same >= self.H)).any(dim=1)
        else:
            compatible = ((~invalid) & (min_same < self.H)).any(dim=1)

        Z_node = state_probs[compatible].sum()

        return float(Z_node.item())

    @torch.no_grad()
    def edge_compatibility_probability(self, eta: torch.Tensor) -> float:
        """
        Compute Z_edge.

        For two opposite warnings w^{i->j} and w^{j->i}, the edge is compatible
        iff there exists a pair (a,b) such that

            w^{i->j}[a,b] = 1
            w^{j->i}[b,a] = 1.

        Equivalently, sum_{a,b} W[i,a,b] * W[j,b,a] > 0, which is the (i,j)
        entry of W_flat @ W_T_flat.T after flattening the (a,b) index. We
        chunk over rows of L to avoid materializing the full [L,L,K,K]
        broadcast intermediate, which does not fit in memory at K=4 with
        a large support.
        """
        eta = self.normalize_valid_survey(eta)

        support = torch.nonzero(eta > 0, as_tuple=False).flatten()
        probs = eta[support]
        L = support.numel()

        W = self.warning_bits[support]                              # [L,K,K]
        W_flat = W.reshape(L, self.K2).to(self.dtype)
        W_T_flat = W.transpose(-1, -2).reshape(L, self.K2).to(self.dtype)

        # Each chunk produces a [c, L] dtype matrix from the matmul. Cap peak
        # at ~256 MB.
        target_bytes = 256 * 1024 * 1024
        chunk = max(1, target_bytes // max(1, L * 8))
        chunk = min(chunk, L)

        Z_edge = torch.zeros((), dtype=self.dtype, device=self.device)

        for ls in range(0, L, chunk):
            le = min(ls + chunk, L)
            counts = W_flat[ls:le] @ W_T_flat.T          # [c, L]
            compat = (counts > 0).to(self.dtype)
            Z_edge = Z_edge + (
                probs[ls:le, None] * probs[None, :] * compat
            ).sum()

        return float(Z_edge.item())

    @torch.no_grad()
    def sp_complexity(self, eta: torch.Tensor) -> tuple[float, float, float]:
        """
        Return:

            Phi_SP, Z_node, Z_edge

        with

            Phi_SP = log Z_node - d/2 log Z_edge.
        """
        eta = self.normalize_valid_survey(eta)

        Z_node = self.node_compatibility_probability(eta)
        Z_edge = self.edge_compatibility_probability(eta)

        if Z_node <= 0.0 or Z_edge <= 0.0:
            return float("-inf"), Z_node, Z_edge

        phi_sp = math.log(Z_node) - 0.5 * self.d * math.log(Z_edge)

        return phi_sp, Z_node, Z_edge

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def dont_care_probability(self, eta: torch.Tensor) -> float:
        return float(eta[self.dont_care_id])

    def all_one_probability(self, eta: torch.Tensor) -> float:
        return float(eta[self.all_one_id])

    def zero_probability(self, eta: torch.Tensor) -> float:
        return float(eta[self.zero_id])

    def print_dont_care_warning(self) -> None:
        print("model dont_care_id:", self.dont_care_id)
        print(self.dont_care_matrix().numpy())

        print("raw all_one_id:", self.all_one_id)
        print(self.warning_matrix(self.all_one_id).numpy())

        print("zero_id:", self.zero_id)
        print(self.warning_matrix(self.zero_id).numpy())

    def print_nonzero_warnings(
        self,
        eta: torch.Tensor,
        threshold: float = 1e-10,
    ) -> None:
        eta_cpu = eta.detach().cpu()
        nz = torch.nonzero(eta_cpu > threshold, as_tuple=False).flatten().tolist()

        print(f"\nWarnings with probability > {threshold}:")
        for wid in nz:
            mat = self.warning_matrix(wid)
            label = ""
            if wid == self.dont_care_id:
                label += "  <-- model don't-care"
            if wid == self.all_one_id:
                label += "  <-- raw all-one"
            if wid == self.zero_id:
                label += "  <-- contradiction/zero"

            print(f"id={wid:6d}  p={eta_cpu[wid].item():.12g}{label}")
            print(mat.numpy())


# ----------------------------------------------------------------------
# Convenience functions
# ----------------------------------------------------------------------

def _mode_abbrev(mode: str) -> str:
    return "ass" if mode == "assortative" else "dis"


def _save_run_json(
    save_dir: str,
    K: int,
    d: int,
    H: int,
    mode: str,
    eps: float,
    config: dict,
    result: dict,
    history: list[dict],
    eta_sparse: dict[int, float],
) -> str:
    """
    Save a single run to:

        {save_dir}/{tag}/{tag}_{eps}_{YYYYMMDD_HHMMSS}.json

    where tag = "{ass|dis}_K{K}_d{d}_H{H}".
    """
    tag = f"{_mode_abbrev(mode)}_K{K}_d{d}_H{H}"
    folder = os.path.join(save_dir, tag)
    os.makedirs(folder, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{tag}_{eps:g}_{timestamp}.json"
    path = os.path.join(folder, fname)

    payload = {
        "saved_at": timestamp,
        "config": config,
        "result": result,
        "history": history,
        "eta_sparse": {str(k): v for k, v in eta_sparse.items()},
    }

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    return path


def run_single(
    K: int,
    d: int,
    H: int,
    mode: str = "disassortative",
    eps: float = 1e-3,
    damping: float = 0.85,
    max_iter: int = 500,
    tol: float = 2e-8,
    verbose: bool = False,
    save_dir: Optional[str] = None,
    eta_sparse_threshold: float = 1e-12,
    log_every: int = 1,
) -> dict:
    sp = KStateSurveyPropagation(
        K=K,
        d=d,
        H=H,
        mode=mode,
        device="cpu",
        dtype=torch.float64,
    )

    result = sp.solve(
        eps=eps,
        damping=damping,
        max_iter=max_iter,
        tol=tol,
        verbose=verbose,
    )

    eta = result.eta.to(sp.device, dtype=sp.dtype)
    phi_sp, Z_node, Z_edge = sp.sp_complexity(eta)
    last = result.history[-1]

    row = {
        "K": K,
        "d": d,
        "H": H,
        "mode": mode,
        "eta_dc": sp.dont_care_probability(eta),
        "eta_all_one": sp.all_one_probability(eta),
        "eta_zero": sp.zero_probability(eta),
        "Phi_SP": phi_sp,
        "Z_node": Z_node,
        "Z_edge": Z_edge,
        "residual": last["residual_l1_half"],
        "iterations": last["iter"] + 1,
        "support": last["support"],
        "dont_care_id": sp.dont_care_id,
        "all_one_id": sp.all_one_id,
        "zero_id": sp.zero_id,
    }

    if save_dir is not None:
        config = {
            "K": K,
            "d": d,
            "H": H,
            "mode": mode,
            "eps": eps,
            "damping": damping,
            "max_iter": max_iter,
            "tol": tol,
        }

        eta_cpu = eta.detach().cpu()
        nz = torch.nonzero(
            eta_cpu > eta_sparse_threshold, as_tuple=False
        ).flatten().tolist()
        eta_sparse = {int(wid): float(eta_cpu[wid].item()) for wid in nz}

        if log_every < 1:
            raise ValueError("log_every must be >= 1.")

        history_to_save = [
            h for h in result.history if h["iter"] % log_every == 0
        ]
        if result.history and history_to_save[-1] is not result.history[-1]:
            history_to_save.append(result.history[-1])

        path = _save_run_json(
            save_dir=save_dir,
            K=K, d=d, H=H, mode=mode, eps=eps,
            config=config,
            result=row,
            history=history_to_save,
            eta_sparse=eta_sparse,
        )
        row["saved_to"] = path
        print(f"Saved run to {path}")

    return row


def scan_K_H(
    Ks: tuple[int, ...] = (2, 3),
    d_max: int = 14,
    mode: str = "disassortative",
    eps: float = 1e-3,
    damping: float = 0.85,
    max_iter: int = 500,
    tol: float = 2e-8,
    save_dir: Optional[str] = None,
    log_every: int = 1,
) -> list[dict]:
    rows: list[dict] = []

    for K in Ks:
        for H in range(1, d_max + 1):
            for d in range(max(1, H), d_max + 1):
                row = run_single(
                    K=K,
                    d=d,
                    H=H,
                    mode=mode,
                    eps=eps,
                    damping=damping,
                    max_iter=max_iter,
                    tol=tol,
                    verbose=False,
                    save_dir=save_dir,
                    log_every=log_every,
                )
                rows.append(row)

                print(
                    f"K={K:2d}  H={H:2d}  d={d:2d}  "
                    f"eta_dc={row['eta_dc']:.8f}  "
                    f"Phi_SP={row['Phi_SP']:.8f}  "
                    f"Z_node={row['Z_node']:.8f}  "
                    f"Z_edge={row['Z_edge']:.8f}  "
                    f"res={row['residual']:.2e}  "
                    f"it={row['iterations']:3d}  "
                    f"support={row['support']:3d}"
                )

    return rows


def benchmark_paper_values() -> None:
    """
    Benchmarks against Table 2 of the paper.

    Paper value:
        K=2, d=8, assortative H=5 -> Phi_SP = 0.02302

    Equivalent disassortative parameter:
        H_dis = d - H_ass + 1 = 8 - 5 + 1 = 4.
    """
    tests = [
        ("assortative", 5, "paper d=8, H_ass=5"),
        ("disassortative", 4, "equivalent d=8, H_dis=4"),
        ("assortative", 6, "paper d=8, H_ass=6"),
        ("disassortative", 3, "equivalent d=8, H_dis=3"),
    ]

    for mode, H, label in tests:
        row = run_single(
            K=2,
            d=8,
            H=H,
            mode=mode,
            eps=1e-3,
            damping=0.85,
            max_iter=500,
            tol=2e-10,
            verbose=False,
        )

        print("\n", label)
        print("mode:", mode, "H:", H)
        print("eta_dc: ", row["eta_dc"])
        print("Phi_SP:", row["Phi_SP"])
        print("Z_node:", row["Z_node"])
        print("Z_edge:", row["Z_edge"])
        print("res:   ", row["residual"])
        print("iters: ", row["iterations"])


if __name__ == "__main__":
    Ks = (2, )
    d_max = 13
    mode = "disassortative"
    eps=1e-3
    damping=0.85
    max_iter=500
    tol=2e-8
    save_dir="results/wp"

    rows = scan_K_H(
        Ks=Ks,
        d_max=d_max,
        mode=mode,
        eps=eps,
        damping=damping,
        max_iter=max_iter,
        tol=tol,
        save_dir=save_dir,
    )