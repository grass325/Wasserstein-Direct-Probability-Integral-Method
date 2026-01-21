"""
Wasserstein (Sliced Wasserstein) vs GF-discrepancy（Weibull 目标分布版本）
- 目标分布：多维 Weibull（边缘 Weibull + Gaussian copula 相关结构，可选）
- y_fixed：一次性生成的低差异度（Sobol/QMC）目标样本点集

SWD：
- 使用“块匹配”版本（当 Ny = y_multiple * Nx 时），并使用 y_fixed 的投影统计做标准化

GF：
- 在 z 空间做逐维分位重排（保证边缘 N(0,1)）
- 通过 Gaussian copula 得到相关结构，再用 Weibull PPF 映射到 x 空间
- 在 x 空间用 y_fixed 的 Voronoi 剖分估计权重
- 在 z 空间做带权分位重排两次，得到最终点集

导出：
- 在同一个 CSV 中输出：QMC(y_fixed) / SWD / GF
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch

from pointset_export import export_point_sets_to_csv
from sampling_common import (
    sobol_net,
    optimize_point_set_lbfgs,
    estimate_voronoi_probabilities_in_x,
    normalize_weights,
    rank_to_mid_u,
    weighted_mid_cdf_1d,
    plot_comparison_2d,
    gf_discrepancy_from_cdf,
    sliced_wasserstein_distance_blockmatch,
)

# =========================================================
# 0) 标准正态分布相关函数（torch版）——用于 Gaussian copula
# =========================================================
_SQRT2 = math.sqrt(2.0)


def norm_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.special.erf(x / _SQRT2))


def norm_ppf(u: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    u = torch.clamp(u, eps, 1.0 - eps)
    return _SQRT2 * torch.special.erfinv(2.0 * u - 1.0)


def weibull_cdf(x: torch.Tensor, k: torch.Tensor, lam: torch.Tensor, eps: float = 1e-18) -> torch.Tensor:
    x = torch.clamp(x, min=0.0)
    z = torch.clamp(x / torch.clamp(lam, min=eps), min=0.0)
    return 1.0 - torch.exp(-(z ** k))


def weibull_ppf(u: torch.Tensor, k: torch.Tensor, lam: torch.Tensor, eps: float = 1e-18) -> torch.Tensor:
    u = torch.clamp(u, eps, 1.0 - eps)
    return lam * (-torch.log1p(-u)) ** (1.0 / k)


def corr_from_cov(Sigma: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    d = Sigma.size(0)
    diag = torch.diag(Sigma)
    std = torch.sqrt(torch.clamp(diag, min=eps))
    inv_std = 1.0 / std
    Corr = Sigma * inv_std.view(d, 1) * inv_std.view(1, d)
    Corr = 0.5 * (Corr + Corr.T)
    Corr.fill_diagonal_(1.0)
    return Corr


def gaussian_copula_u(z: torch.Tensor, Corr: torch.Tensor | None) -> torch.Tensor:
    if Corr is None:
        return norm_cdf(z)
    L = torch.linalg.cholesky(Corr)
    z_corr = z @ L.T
    return norm_cdf(z_corr)


def qmc_weibull_samples(
    n: int,
    d: int,
    device: str,
    dtype: torch.dtype,
    k: torch.Tensor,
    lam: torch.Tensor,
    Corr: torch.Tensor | None = None,
    skip: int = 2**3,
    scramble: bool = True,
    seed: int = 0,
) -> torch.Tensor:
    u = sobol_net(n=n, s=d, skip=skip, device=device, dtype=dtype, scramble=scramble, seed=seed)
    z = norm_ppf(u)
    u_c = gaussian_copula_u(z, Corr=Corr.to(device=device, dtype=dtype) if Corr is not None else None)
    k_d = k.to(device=device, dtype=dtype).view(1, d)
    lam_d = lam.to(device=device, dtype=dtype).view(1, d)
    return weibull_ppf(u_c, k=k_d, lam=lam_d)


def optimize_point_set_against_fixed_y(
    Nx: int,
    d: int,
    y_fixed: torch.Tensor,
    y_multiple: int,
    steps: int = 1000,
    lr: float = 1.0,
    num_projections: int = 512,
    p: int = 2,
    block_reduce: str = "sum",
    standardize_by_y: bool = True,
    swd_eps: float = 1e-8,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
    log_every: int = 50,
    lbfgs_max_iter: int = 20,
    lbfgs_history_size: int = 50,
    line_search_fn: str = "strong_wolfe",
) -> torch.Tensor:
    return optimize_point_set_lbfgs(
        Nx=Nx,
        d=d,
        y_fixed=y_fixed,
        y_multiple=y_multiple,
        steps=steps,
        lr=lr,
        num_projections=num_projections,
        p=p,
        block_reduce=block_reduce,
        standardize_by_y=standardize_by_y,
        swd_eps=swd_eps,
        device=device,
        dtype=dtype,
        seed=seed,
        log_every=log_every,
        lbfgs_max_iter=lbfgs_max_iter,
        lbfgs_history_size=lbfgs_history_size,
        line_search_fn=line_search_fn,
        theta_mode="per_iter",
        cache_y_projection=True,
        param_init=None,
        param_to_x=lambda u: u,
    )


@dataclass
class GFResultWeibull:
    z_points: torch.Tensor
    x_points: torch.Tensor
    weights: torch.Tensor


def select_points_gf_weibull_using_y_fixed(
    n: int,
    d: int,
    y_fixed: torch.Tensor,
    k: torch.Tensor,
    lam: torch.Tensor,
    Corr: torch.Tensor | None = None,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    seed: int = 0,
    chunk_size: int = 8192,
    qmc_skip: int = 2**3,
    qmc_scramble: bool = True,
    prob_eps: float = 1e-12,
) -> GFResultWeibull:
    torch.manual_seed(seed)

    u0 = sobol_net(n=n, s=d, skip=qmc_skip, device=device, dtype=dtype, scramble=qmc_scramble, seed=seed)
    z0 = norm_ppf(u0)

    z1 = torch.empty_like(z0)
    for j in range(d):
        z1[:, j] = norm_ppf(rank_to_mid_u(z0[:, j]))

    k_d = k.to(device=device, dtype=dtype).view(1, d)
    lam_d = lam.to(device=device, dtype=dtype).view(1, d)
    Corr_d = Corr.to(device=device, dtype=dtype) if Corr is not None else None

    u1 = gaussian_copula_u(z1, Corr=Corr_d)
    x1 = weibull_ppf(u1, k=k_d, lam=lam_d)

    yx = y_fixed.to(device=device, dtype=dtype)
    Pq1 = estimate_voronoi_probabilities_in_x(yx, x1, chunk_size=chunk_size)
    Pq = normalize_weights(Pq1 + prob_eps)

    def rearrange_once(z_in: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        u = torch.empty_like(z_in)
        for j in range(d):
            u[:, j] = weighted_mid_cdf_1d(z_in[:, j], w)
        return norm_ppf(u)

    z3 = rearrange_once(z1, Pq)
    z3 = rearrange_once(z3, Pq)

    u3 = gaussian_copula_u(z3, Corr=Corr_d)
    x3 = weibull_ppf(u3, k=k_d, lam=lam_d)

    return GFResultWeibull(z_points=z3, x_points=x3, weights=Pq)


def gf_discrepancy_weibull_marginals(
    x: torch.Tensor,
    k: torch.Tensor,
    lam: torch.Tensor,
    weights: torch.Tensor | None = None,
    eps: float = 1e-15,
) -> float:
    k = k.to(device=x.device, dtype=x.dtype).view(1, x.size(1))
    lam = lam.to(device=x.device, dtype=x.dtype).view(1, x.size(1))
    F = weibull_cdf(x, k=k, lam=lam)
    return gf_discrepancy_from_cdf(x, F, weights=weights, eps=eps)


if __name__ == "__main__":
    device_w = "cuda" if torch.cuda.is_available() else "cpu"
    device_g = "cpu"

    d = 2
    Nx = 128
    steps = 10
    y_multiple = 16
    Ny = Nx * y_multiple

    k = torch.tensor([1.6, 2.2], dtype=torch.float32)
    lam = torch.tensor([1.2, 0.8], dtype=torch.float32)

    Sigma = torch.tensor([[1.2, 0.6], [0.6, 0.8]], dtype=torch.float32)
    Corr = corr_from_cov(Sigma)

    y_fixed = qmc_weibull_samples(
        n=Ny, d=d, device=device_w, dtype=torch.float32,
        k=k, lam=lam, Corr=Corr,
        skip=2**3, scramble=True, seed=0
    )

    Xw = optimize_point_set_against_fixed_y(
        Nx=Nx, d=d, y_fixed=y_fixed, y_multiple=y_multiple,
        steps=steps, lr=1.0, num_projections=512, p=2, block_reduce="mean",
        standardize_by_y=True, swd_eps=1e-8,
        device=device_w, dtype=torch.float32,
        seed=0, log_every=1, lbfgs_max_iter=20, line_search_fn="strong_wolfe"
    )

    Ww = estimate_voronoi_probabilities_in_x(y_fixed, Xw.to(device_w), chunk_size=8192)
    Ww = normalize_weights(Ww)

    gf_res = select_points_gf_weibull_using_y_fixed(
        n=Nx, d=d, y_fixed=y_fixed,
        k=k, lam=lam, Corr=Corr,
        device=device_g, dtype=torch.float32, seed=0,
        chunk_size=8192, qmc_skip=2**3, qmc_scramble=True, prob_eps=1e-12
    )
    Xg = gf_res.x_points.to(device_w)
    Wg = gf_res.weights.to(device_w)

    with torch.no_grad():
        swd_w = sliced_wasserstein_distance_blockmatch(
            Xw, y_fixed, y_multiple=y_multiple, num_projections=1024,
            p=2, block_reduce="sum", standardize_by_y=True
        ).item()
        swd_g = sliced_wasserstein_distance_blockmatch(
            Xg, y_fixed, y_multiple=y_multiple, num_projections=1024,
            p=2, block_reduce="sum", standardize_by_y=True
        ).item()

    gfd_w_wt = gf_discrepancy_weibull_marginals(Xw, k, lam, weights=Ww)
    gfd_g_wt = gf_discrepancy_weibull_marginals(Xg, k, lam, weights=Wg)
    gfd_w_unw = gf_discrepancy_weibull_marginals(Xw, k, lam, weights=None)
    gfd_g_unw = gf_discrepancy_weibull_marginals(Xg, k, lam, weights=None)

    std_w = float(Ww.std(unbiased=False).item())
    std_g = float(Wg.std(unbiased=False).item())

    print(f"[Metric] SWD (Xw vs y_fixed) = {swd_w:.6f}")
    print(f"[Metric] SWD (Xg vs y_fixed) = {swd_g:.6f}")

    print(f"[Metric] GF-disc weighted   (Xw, Voronoi) = {gfd_w_wt:.6f}")
    print(f"[Metric] GF-disc weighted   (Xg, Voronoi) = {gfd_g_wt:.6f}")
    print(f"[Metric] GF-disc unweighted (Xw, equal)   = {gfd_w_unw:.6f}")
    print(f"[Metric] GF-disc unweighted (Xg, equal)   = {gfd_g_unw:.6f}")

    print(f"[Metric] std(weights) Wasserstein(Voronoi) = {std_w:.6e}")
    print(f"[Metric] std(weights) GF(Voronoi)         = {std_g:.6e}")

    plot_comparison_2d(
        y=y_fixed.detach().cpu(),
        x_w=Xw.detach().cpu(),
        x_gf=Xg.detach().cpu(),
        save_path="compare_sampling.png",
        title="Wasserstein vs GF (Weibull target)",
        marker_size=16.0,
    )

    out_csv = Path(__file__).resolve().parent / "Weibull.csv"
    export_point_sets_to_csv(
        out_csv,
        y_qmc=y_fixed.detach().cpu(),
        w_qmc=None,
        x_swd=Xw.detach().cpu(),
        w_swd=None,
        x_gf=Xg.detach().cpu(),
        w_gf=Wg.detach().cpu(),
        method_qmc="QMC",
        method_swd="SWD",
        method_gf="GF",
    )
    print(f"[Export] Wrote: {out_csv}")
