"""
Wasserstein (Sliced Wasserstein) vs GF-discrepancy（仅正态变量点选）
- 目标分布：多维高斯 N(mu, Sigma)
- y_fixed：一次性生成的低差异度（Sobol/QMC）目标样本点集

SWD：
- 使用“块匹配”版本（当 Ny = y_multiple * Nx 时）
  对每个投影方向：
    0) 用 y_fixed 在该方向投影估计 mean/std，并对 proj_x/proj_y 做标准化
    1) sort(proj_x) -> proj_x_sorted (Nx)
    2) sort(proj_y) -> proj_y_sorted (Ny)
    3) 令 m = Ny/Nx = y_multiple
       将 proj_x_sorted[i] 与 proj_y_sorted[i*m : (i+1)*m] 配对，计算距离并聚合

GF：
- 在 z 空间做逐维分位重排（保证边缘 N(0,1)）
- 变换到 x 空间后在“原空间 x”上，用 y_fixed 做 Voronoi 剖分近似目标测度，得到 Voronoi 权重
- 在 z 空间做带权分位重排两次（weighted-midpoint CDF），得到最终点集

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
# 0) 标准正态分布相关函数（torch版）
# =========================================================
_SQRT2 = math.sqrt(2.0)


def norm_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.special.erf(x / _SQRT2))


def norm_ppf(u: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    u = torch.clamp(u, eps, 1.0 - eps)
    return _SQRT2 * torch.special.erfinv(2.0 * u - 1.0)


# =========================================================
# 1) 目标分布：多维高斯 N(mu, Sigma)
# =========================================================
def cholesky_affine(z: torch.Tensor, mu: torch.Tensor, Sigma: torch.Tensor) -> torch.Tensor:
    """z ~ N(0,I), return x = mu + L z, where Sigma = L L^T."""
    L = torch.linalg.cholesky(Sigma)
    return mu + z @ L.T


# =========================================================
# 2) 低差异度点集（Sobol/QMC）生成：目标 y_fixed
# =========================================================
def qmc_gaussian_samples(
    n: int,
    d: int,
    device: str,
    dtype: torch.dtype,
    mu: torch.Tensor,
    Sigma: torch.Tensor,
    skip: int = 2**3,
    scramble: bool = True,
    seed: int = 0,
) -> torch.Tensor:
    """Sobol([0,1]^d) -> Φ^{-1} -> z ~ N(0,1) -> affine to N(mu,Sigma)."""
    u = sobol_net(n=n, s=d, skip=skip, device=device, dtype=dtype, scramble=scramble, seed=seed)
    z = norm_ppf(u)
    return cholesky_affine(z, mu.to(device=device, dtype=dtype), Sigma.to(device=device, dtype=dtype))


# =========================================================
# 3) Wasserstein：用 LBFGS 优化 X (Nx,d) 使其 SWD 接近 y_fixed
# =========================================================
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
    standardize_by_qmc: bool = True,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
    log_every: int = 50,
    fix_projections: bool = True,
    lbfgs_max_iter: int = 20,
    lbfgs_history_size: int = 50,
    line_search_fn: str = "strong_wolfe",
) -> torch.Tensor:
    theta_mode = "fixed" if fix_projections else "per_iter"
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
        standardize_by_y=standardize_by_qmc,
        swd_eps=1e-8,
        device=device,
        dtype=dtype,
        seed=seed,
        log_every=log_every,
        lbfgs_max_iter=lbfgs_max_iter,
        lbfgs_history_size=lbfgs_history_size,
        line_search_fn=line_search_fn,
        theta_mode=theta_mode,
        cache_y_projection=False,
        param_init=None,
        param_to_x=lambda u: u,
    )


# =========================================================
# 4) GF 方法：点选 + 原空间 Voronoi 权重 + 重排
# =========================================================
@dataclass
class GFResultNormal:
    z_points: torch.Tensor
    x_points: torch.Tensor
    weights: torch.Tensor


def select_points_gf_normal_using_y_fixed(
    n: int,
    d: int,
    y_fixed: torch.Tensor,
    mu: torch.Tensor,
    Sigma: torch.Tensor,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    seed: int = 0,
    chunk_size: int = 8192,
    qmc_skip: int = 2**3,
    qmc_scramble: bool = True,
    prob_eps: float = 1e-12,
) -> GFResultNormal:
    """
    GF 点选（正态变量）：
    - z 空间：Sobol(scramble=True) -> Φ^{-1}，再逐维分位重排
    - x 空间：基于 y_fixed 的 Voronoi 剖分估计权重
    - z 空间：带权分位重排两次
    """
    torch.manual_seed(seed)

    u0 = sobol_net(n=n, s=d, skip=qmc_skip, device=device, dtype=dtype, scramble=qmc_scramble, seed=seed)
    z0 = norm_ppf(u0)

    z1 = torch.empty_like(z0)
    for j in range(d):
        z1[:, j] = norm_ppf(rank_to_mid_u(z0[:, j]))

    mu_d = mu.to(device=device, dtype=dtype)
    Sigma_d = Sigma.to(device=device, dtype=dtype)
    x1 = cholesky_affine(z1, mu_d, Sigma_d)

    yx = y_fixed.to(device=device, dtype=dtype)
    Pq1 = estimate_voronoi_probabilities_in_x(yx, x1, chunk_size=chunk_size)

    # 避免 0 权重导致退化（保持点数为 n）
    Pq = normalize_weights(Pq1 + prob_eps)

    def rearrange_once(z_in: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        u = torch.empty_like(z_in)
        for j in range(d):
            u[:, j] = weighted_mid_cdf_1d(z_in[:, j], w)
        return norm_ppf(u)

    z3 = rearrange_once(z1, Pq)
    z3 = rearrange_once(z3, Pq)

    x3 = cholesky_affine(z3, mu_d, Sigma_d)
    return GFResultNormal(z_points=z3, x_points=x3, weights=Pq)


# =========================================================
# 5) GF-discrepancy（目标空间边缘正态）
# =========================================================
def gf_discrepancy_normal_marginals(
    x: torch.Tensor,
    mu: torch.Tensor,
    Sigma: torch.Tensor,
    weights: torch.Tensor | None = None,
    eps: float = 1e-15,
) -> float:
    """
    逐维边缘 GF-discrepancy（正态目标边缘）：
      F_ij  = Φ((x_ij - mu_j)/sqrt(Sigma_jj))
      Fn_ij = Σ_{k: x_kj < x_ij} w_k
    返回 max_{i,j} |Fn_ij - F_ij|
    """
    assert x.dim() == 2
    n, d = x.shape

    mu = mu.to(device=x.device, dtype=x.dtype).view(1, d)
    diag = torch.diag(Sigma.to(device=x.device, dtype=x.dtype)).view(1, d)
    std = torch.sqrt(torch.clamp(diag, min=eps))

    z = (x - mu) / std
    F = norm_cdf(z)
    return gf_discrepancy_from_cdf(x, F, weights=weights, eps=eps)


# =========================================================
# 6) Demo
# =========================================================
if __name__ == "__main__":
    device_w = "cuda" if torch.cuda.is_available() else "cpu"
    device_g = "cpu"

    d = 2
    Nx = 128
    steps = 10
    y_multiple = 16
    Ny = Nx * y_multiple

    mu = torch.tensor([1.0, -0.5], dtype=torch.float32, device=device_w)
    Sigma = torch.tensor([[1.2, 0.6], [0.6, 0.8]], dtype=torch.float32, device=device_w)

    # A) 固定目标点集 y_fixed（scramble=True）
    y_fixed = qmc_gaussian_samples(
        n=Ny, d=d, device=device_w, dtype=torch.float32,
        mu=mu, Sigma=Sigma, skip=2**3, scramble=True, seed=0
    )

    # B) Wasserstein：优化得到点集 Xw（块匹配 SWD）—— LBFGS
    Xw = optimize_point_set_against_fixed_y(
        Nx=Nx, d=d, y_fixed=y_fixed, y_multiple=y_multiple,
        steps=steps, lr=1.0, num_projections=512, p=2, block_reduce="mean",
        standardize_by_qmc=True, device=device_w, dtype=torch.float32,
        seed=0, log_every=1, fix_projections=True,
        lbfgs_max_iter=20, line_search_fn="strong_wolfe"
    )

    # C) SWD 权重：Voronoi 重分配（用于加权 GF-discrepancy 等）
    Ww = estimate_voronoi_probabilities_in_x(y_fixed, Xw.to(device_w), chunk_size=8192)
    Ww = normalize_weights(Ww)

    # D) GF：点选 + Voronoi 权重
    gf_res = select_points_gf_normal_using_y_fixed(
        n=Nx, d=d, y_fixed=y_fixed, mu=mu.to(device_g), Sigma=Sigma.to(device_g),
        device=device_g, dtype=torch.float32, seed=0,
        chunk_size=8192, qmc_skip=2**3, qmc_scramble=True, prob_eps=1e-12
    )
    Xg = gf_res.x_points.to(device_w)
    Wg = gf_res.weights.to(device_w)

    # E) 指标：GF-discrepancy（加权 + 不加权）、SWD（不加权）、权重标准差
    with torch.no_grad():
        swd_w = sliced_wasserstein_distance_blockmatch(
            Xw, y_fixed, y_multiple=y_multiple,
            num_projections=1024, p=2, block_reduce="sum", standardize_by_y=True
        ).item()
        swd_g = sliced_wasserstein_distance_blockmatch(
            Xg, y_fixed, y_multiple=y_multiple,
            num_projections=1024, p=2, block_reduce="sum", standardize_by_y=True
        ).item()

    gfd_w_wt = gf_discrepancy_normal_marginals(Xw, mu, Sigma, weights=Ww)
    gfd_g_wt = gf_discrepancy_normal_marginals(Xg, mu, Sigma, weights=Wg)

    gfd_w_unw = gf_discrepancy_normal_marginals(Xw, mu, Sigma, weights=None)
    gfd_g_unw = gf_discrepancy_normal_marginals(Xg, mu, Sigma, weights=None)

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

    # F) 画图（点大小统一）
    plot_comparison_2d(
        y=y_fixed.detach().cpu(),
        x_w=Xw.detach().cpu(),
        x_gf=Xg.detach().cpu(),
        save_path="compare_sampling.png",
        title="Wasserstein vs GF (Normal target)",
        marker_size=16.0,
    )

    # G) 导出 CSV：同一个文件内输出 QMC / SWD / GF
    out_csv = Path(__file__).resolve().parent / "Normal.csv"
    export_point_sets_to_csv(
        out_csv,
        y_qmc=y_fixed.detach().cpu(),
        w_qmc=None,
        x_swd=Xw.detach().cpu(),
        w_swd=None,  # SWD 视为等权
        x_gf=Xg.detach().cpu(),
        w_gf=Wg.detach().cpu(),  # GF 用 Voronoi 权重
        method_qmc="QMC",
        method_swd="SWD",
        method_gf="GF",
    )
    print(f"[Export] Wrote: {out_csv}")
