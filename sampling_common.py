"""sampling_common.py

Common utilities shared by the Normal / Uniform / Weibull sampling demos.

This module is intentionally distribution-agnostic. It provides:
  - Sobol/QMC generation
  - Sliced Wasserstein distance (block-matching, Ny = y_multiple * Nx)
  - An LBFGS optimization driver with optional slice-wise standardization
    and optional caching of y-projection statistics/sorts per theta
  - Voronoi-based weight estimation in the original (x) space
  - Rank / weighted-CDF mid-point helpers for GF rearrangement
  - Generic marginal GF-discrepancy given marginal CDF values
  - Weighted resampling and a standard 2D scatter comparison plot
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# =========================================================
# 1) QMC / Sobol
# =========================================================
def sobol_net(
    n: int,
    s: int,
    skip: int,
    device: str,
    dtype: torch.dtype,
    scramble: bool = True,
    seed: int | None = None,
) -> torch.Tensor:
    """Generate Sobol points in [0,1]^s."""
    eng = torch.quasirandom.SobolEngine(dimension=s, scramble=scramble, seed=seed)
    if skip > 0:
        eng.fast_forward(skip)
    return eng.draw(n).to(device=device, dtype=dtype)


# =========================================================
# 2) Basic helpers
# =========================================================
def normalize_weights(w: torch.Tensor, eps: float = 1e-18) -> torch.Tensor:
    s = w.sum()
    if s <= eps:
        return torch.full_like(w, 1.0 / w.numel())
    return w / s


def estimate_voronoi_probabilities_in_x(
    y_fixed_x: torch.Tensor,
    rep_x: torch.Tensor,
    chunk_size: int = 8192,
) -> torch.Tensor:
    """Estimate Voronoi cell probabilities in x-space using y_fixed as measure."""
    assert y_fixed_x.dim() == 2 and rep_x.dim() == 2
    Ny, d = y_fixed_x.shape
    nrep = rep_x.size(0)
    assert rep_x.size(1) == d

    rep_norm2 = (rep_x * rep_x).sum(dim=1)
    counts = torch.zeros(nrep, device=rep_x.device, dtype=torch.long)

    for start in range(0, Ny, chunk_size):
        end = min(Ny, start + chunk_size)
        yb = y_fixed_x[start:end]
        yb_norm2 = (yb * yb).sum(dim=1, keepdim=True)
        dist2 = yb_norm2 - 2.0 * (yb @ rep_x.t()) + rep_norm2.unsqueeze(0)
        idx = torch.argmin(dist2, dim=1)
        counts += torch.bincount(idx, minlength=nrep)

    return counts.to(rep_x.dtype) / float(Ny)


def rank_to_mid_u(x: torch.Tensor) -> torch.Tensor:
    """Convert values to mid-point ranks in (0,1)."""
    n = x.numel()
    order = torch.argsort(x)
    ranks = torch.empty(n, device=x.device, dtype=torch.long)
    ranks[order] = torch.arange(n, device=x.device, dtype=torch.long)
    return (ranks.to(x.dtype) + 0.5) / n


def weighted_mid_cdf_1d(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Return weighted mid-point empirical CDF values aligned with x."""
    order = torch.argsort(x)
    w_sorted = w[order]
    cumsum = torch.cumsum(w_sorted, dim=0)
    mid = (cumsum - w_sorted) + 0.5 * w_sorted
    u = torch.empty_like(x)
    u[order] = mid
    return u


def resample_discrete(points: torch.Tensor, weights: torch.Tensor, m: int, seed: int = 0) -> torch.Tensor:
    """Multinomial resampling with replacement."""
    g = torch.Generator(device=points.device)
    g.manual_seed(seed)
    idx = torch.multinomial(normalize_weights(weights), num_samples=m, replacement=True, generator=g)
    return points[idx]


# =========================================================
# 3) Sliced Wasserstein (block match)
# =========================================================
def _make_theta(
    num_projections: int,
    d: int,
    device: torch.device,
    dtype: torch.dtype,
    eps: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    theta = torch.randn(num_projections, d, device=device, dtype=dtype, generator=generator)
    return theta / (theta.norm(dim=1, keepdim=True) + eps)


@torch.no_grad()
def build_y_cache(
    y: torch.Tensor,
    theta: torch.Tensor,
    standardize_by_y: bool = True,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cache (mu_y, inv_std_y, sorted(proj_y_std)) for a given theta."""
    proj_y = y @ theta.t()  # (Ny,K)

    if standardize_by_y:
        mu_y = proj_y.mean(dim=0, keepdim=True)  # (1,K)
        var_y = (proj_y - mu_y).pow(2).mean(dim=0, keepdim=True)  # population var
        inv_std_y = torch.rsqrt(var_y + eps)
        proj_y = (proj_y - mu_y) * inv_std_y
    else:
        K = theta.size(0)
        mu_y = torch.zeros(1, K, device=y.device, dtype=y.dtype)
        inv_std_y = torch.ones(1, K, device=y.device, dtype=y.dtype)

    proj_y_sorted, _ = torch.sort(proj_y, dim=0)
    return mu_y, inv_std_y, proj_y_sorted


def sliced_wasserstein_distance_blockmatch(
    x: torch.Tensor,  # (Nx,d)
    y: torch.Tensor,  # (Ny,d)
    y_multiple: int,
    num_projections: int = 256,
    p: int = 2,
    eps: float = 1e-8,
    block_reduce: str = "sum",
    theta: torch.Tensor | None = None,
    standardize_by_y: bool = True,
    y_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Sliced Wasserstein distance using 1D block matching (Ny = y_multiple * Nx)."""
    assert x.dim() == 2 and y.dim() == 2 and x.size(1) == y.size(1)
    Nx, d = x.shape
    Ny = y.size(0)
    assert Ny == Nx * y_multiple, f"需要 Ny == Nx * y_multiple，但得到 Ny={Ny}, Nx={Nx}, y_multiple={y_multiple}"

    if theta is None:
        theta = _make_theta(num_projections, d, x.device, x.dtype, eps)
    else:
        assert theta.dim() == 2 and theta.size(1) == d, "theta 形状需为 (K,d)"
        theta = theta.to(device=x.device, dtype=x.dtype)
        num_projections = theta.size(0)
        theta = theta / (theta.norm(dim=1, keepdim=True) + eps)

    # ---- x projections ----
    proj_x = x @ theta.t()  # (Nx,K)

    # ---- y projections (or cached) + standardization ----
    if y_cache is None:
        proj_y = y @ theta.t()  # (Ny,K)
        if standardize_by_y:
            mu_y = proj_y.mean(dim=0, keepdim=True).detach()
            var_y = (proj_y - mu_y).pow(2).mean(dim=0, keepdim=True).detach()
            inv_std_y = torch.rsqrt(var_y + eps)
            proj_x = (proj_x - mu_y) * inv_std_y
            proj_y = (proj_y - mu_y) * inv_std_y
        proj_y_sorted, _ = torch.sort(proj_y, dim=0)
    else:
        mu_y, inv_std_y, proj_y_sorted = y_cache
        mu_y = mu_y.to(device=x.device, dtype=x.dtype)
        inv_std_y = inv_std_y.to(device=x.device, dtype=x.dtype)
        proj_y_sorted = proj_y_sorted.to(device=x.device, dtype=x.dtype)
        if standardize_by_y:
            proj_x = (proj_x - mu_y) * inv_std_y

    # ---- sort x projections ----
    proj_x_sorted, _ = torch.sort(proj_x, dim=0)  # (Nx,K)

    # ---- block matching ----
    m = y_multiple
    x_rep = proj_x_sorted.repeat_interleave(m, dim=0)  # (Ny,K)
    diff = x_rep - proj_y_sorted

    if p == 1:
        cost = diff.abs()
    elif p == 2:
        cost = diff * diff
    else:
        cost = diff.abs().pow(p)

    cost_blk = cost.view(Nx, m, num_projections)

    if block_reduce == "sum":
        per_x = cost_blk.sum(dim=1)  # (Nx,K)
    elif block_reduce == "mean":
        per_x = cost_blk.mean(dim=1)  # (Nx,K)
    else:
        raise ValueError("block_reduce must be 'sum' or 'mean'")

    return per_x.mean(dim=0).mean()


# =========================================================
# 4) Optimization driver (LBFGS)
# =========================================================
def optimize_point_set_lbfgs(
    *,
    Nx: int,
    d: int,
    y_fixed: torch.Tensor,
    y_multiple: int,
    steps: int = 1000,
    lr: float = 1.0,
    num_projections: int = 512,
    p: int = 2,
    block_reduce: str = "sum",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
    log_every: int = 100,
    lbfgs_max_iter: int = 20,
    lbfgs_history_size: int = 50,
    line_search_fn: str | None = "strong_wolfe",
    theta_mode: str = "fixed",  # 'fixed' | 'per_iter'
    cache_y_projection: bool = False,
    standardize_by_y: bool = True,
    swd_eps: float = 1e-8,
    param_init: torch.Tensor | None = None,
    param_to_x: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Generic LBFGS optimizer for x-point sets.

    Parameters
    - param_to_x: maps unconstrained parameters to x (identity if None).
    - theta_mode:
        * 'fixed': theta is fixed throughout optimization.
        * 'per_iter': theta is regenerated once per outer iteration.
    - cache_y_projection:
        If True, compute (mu_y, inv_std_y, sorted proj_y) for each theta (or once for fixed theta)
        and reuse within the LBFGS closure calls.
    """
    if param_to_x is None:
        param_to_x = lambda u: u

    torch.manual_seed(seed)
    y_fixed = y_fixed.to(device=device, dtype=dtype)

    if param_init is None:
        param_init = torch.randn(Nx, d, device=device, dtype=dtype)
    else:
        param_init = param_init.to(device=device, dtype=dtype)

    u = nn.Parameter(param_init)
    opt = torch.optim.LBFGS(
        [u],
        lr=lr,
        max_iter=lbfgs_max_iter,
        history_size=lbfgs_history_size,
        line_search_fn=line_search_fn,
    )

    theta_fixed: torch.Tensor | None = None
    y_cache_fixed: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    if theta_mode == "fixed":
        theta_fixed = _make_theta(num_projections, d, y_fixed.device, y_fixed.dtype, swd_eps)
        if cache_y_projection:
            y_cache_fixed = build_y_cache(y_fixed, theta_fixed, standardize_by_y=standardize_by_y, eps=swd_eps)

    for it in range(1, steps + 1):
        if theta_mode == "fixed":
            theta_k = theta_fixed
            y_cache_k = y_cache_fixed
        elif theta_mode == "per_iter":
            theta_k = _make_theta(num_projections, d, y_fixed.device, y_fixed.dtype, swd_eps)
            y_cache_k = (
                build_y_cache(y_fixed, theta_k, standardize_by_y=standardize_by_y, eps=swd_eps)
                if cache_y_projection
                else None
            )
        else:
            raise ValueError("theta_mode must be 'fixed' or 'per_iter'")

        loss_holder: dict[str, torch.Tensor | None] = {"loss": None}

        def closure() -> torch.Tensor:
            opt.zero_grad(set_to_none=True)
            x = param_to_x(u)
            loss = sliced_wasserstein_distance_blockmatch(
                x,
                y_fixed,
                y_multiple=y_multiple,
                num_projections=num_projections,
                p=p,
                eps=swd_eps,
                block_reduce=block_reduce,
                theta=theta_k,
                standardize_by_y=standardize_by_y,
                y_cache=y_cache_k,
            )
            loss.backward()
            loss_holder["loss"] = loss.detach()
            return loss

        opt.step(closure)

        if it % log_every == 0 or it == 1:
            loss_val = loss_holder["loss"]
            loss_item = float(loss_val.item()) if loss_val is not None else float("nan")
            print(
                f"[W-LBFGS] step {it:4d} | SWD_block(p={p},{block_reduce},std_by_y={standardize_by_y})={loss_item:.6f} "
                f"| Nx={Nx}, Ny={y_fixed.size(0)}"
            )

    return param_to_x(u.detach())


# =========================================================
# 5) Generic marginal GF-discrepancy (given CDF values)
# =========================================================
def gf_discrepancy_from_cdf(
    x: torch.Tensor,
    F: torch.Tensor,
    weights: torch.Tensor | None = None,
    eps: float = 1e-15,
) -> float:
    """Compute marginal GF-discrepancy given marginal CDF values F(x).

    F must be in [0,1] with the same shape as x.
    """
    assert x.dim() == 2 and F.dim() == 2 and x.shape == F.shape
    n, d = x.shape

    if weights is None:
        w = torch.full((n,), 1.0 / n, device=x.device, dtype=x.dtype)
    else:
        w = normalize_weights(weights.to(device=x.device, dtype=x.dtype), eps=eps)

    gfd = 0.0
    for j in range(d):
        order = torch.argsort(x[:, j])
        w_sorted = w[order]
        F_sorted = F[order, j]
        cumsum = torch.cumsum(w_sorted, dim=0)
        Fn_sorted = cumsum - w_sorted
        gfd = max(gfd, (Fn_sorted - F_sorted).abs().max().item())

    return float(gfd)


# =========================================================
# 6) Plot
# =========================================================
def plot_comparison_2d(
    y: torch.Tensor,
    x_w: torch.Tensor,
    x_gf: torch.Tensor,
    save_path: str = "compare_sampling.png",
    title: str = "Wasserstein vs GF",
    marker_size: float = 16.0,
    label_y: str = "y_fixed (QMC samples)",
    label_w: str = "Wasserstein points",
    label_gf: str = "GF points",
):
    y2 = y.detach().cpu()[:, :2]
    xw2 = x_w.detach().cpu()[:, :2]
    xg2 = x_gf.detach().cpu()[:, :2]

    plt.figure(figsize=(7.2, 6.6))
    plt.scatter(y2[:, 0].numpy(), y2[:, 1].numpy(), s=marker_size, alpha=0.35, label=label_y)
    plt.scatter(xw2[:, 0].numpy(), xw2[:, 1].numpy(), s=marker_size, alpha=0.80, label=label_w)
    plt.scatter(xg2[:, 0].numpy(), xg2[:, 1].numpy(), s=marker_size, alpha=0.80, label=label_gf)
    plt.title(title)
    plt.xlabel("dim 1")
    plt.ylabel("dim 2")
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.show()


__all__ = [
    "sobol_net",
    "normalize_weights",
    "estimate_voronoi_probabilities_in_x",
    "rank_to_mid_u",
    "weighted_mid_cdf_1d",
    "resample_discrete",
    "build_y_cache",
    "sliced_wasserstein_distance_blockmatch",
    "optimize_point_set_lbfgs",
    "gf_discrepancy_from_cdf",
    "plot_comparison_2d",
]
