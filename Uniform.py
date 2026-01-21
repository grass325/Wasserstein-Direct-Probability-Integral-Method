"""
Wasserstein (Sliced Wasserstein) vs GF-discrepancy（仅均匀变量点选）
- 目标分布：多维均匀分布 Unif([low, high]^d)
- y_fixed：一次性生成的低差异度（Sobol/QMC）目标样本点集（映射到 [low, high]^d）

SWD：
- 使用“块匹配”版本（当 Ny = y_multiple * Nx 时），并使用 y_fixed 的投影统计做标准化

GF：
- 在 x 空间逐维分位重排以满足边缘均匀
- 在 x 空间用 y_fixed 的 Voronoi 剖分估计权重
- 在 x 空间做带权分位重排两次

导出：
- 在同一个 CSV 中输出：QMC(y_fixed) / SWD / GF
"""

from __future__ import annotations

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


def uniform_ppf(u: torch.Tensor, low: torch.Tensor, high: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    u = torch.clamp(u, eps, 1.0 - eps)
    return low + (high - low) * u


def uniform_cdf(x: torch.Tensor, low: torch.Tensor, high: torch.Tensor, eps: float = 1e-15) -> torch.Tensor:
    denom = torch.clamp(high - low, min=eps)
    u = (x - low) / denom
    return torch.clamp(u, 0.0, 1.0)


def qmc_uniform_samples(
    n: int,
    d: int,
    device: str,
    dtype: torch.dtype,
    low: torch.Tensor,
    high: torch.Tensor,
    skip: int = 2**3,
    scramble: bool = True,
    seed: int = 0,
) -> torch.Tensor:
    low = low.to(device=device, dtype=dtype).view(1, d)
    high = high.to(device=device, dtype=dtype).view(1, d)
    u = sobol_net(n=n, s=d, skip=skip, device=device, dtype=dtype, scramble=scramble, seed=seed)
    return uniform_ppf(u, low, high)


def optimize_point_set_against_fixed_y(
    Nx: int,
    d: int,
    y_fixed: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
    y_multiple: int,
    steps: int = 1000,
    lbfgs_lr: float = 1.0,
    num_projections: int = 512,
    p: int = 2,
    block_reduce: str = "sum",
    qmc_standardize: bool = True,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
    log_every: int = 50,
    lbfgs_max_iter: int = 20,
    lbfgs_history_size: int = 50,
    line_search_fn: str = "strong_wolfe",
) -> torch.Tensor:
    low_d = low.to(device=device, dtype=dtype).view(1, d)
    high_d = high.to(device=device, dtype=dtype).view(1, d)

    def param_to_x(u: torch.Tensor) -> torch.Tensor:
        return uniform_ppf(torch.sigmoid(u), low_d, high_d)

    return optimize_point_set_lbfgs(
        Nx=Nx,
        d=d,
        y_fixed=y_fixed,
        y_multiple=y_multiple,
        steps=steps,
        lr=lbfgs_lr,
        num_projections=num_projections,
        p=p,
        block_reduce=block_reduce,
        standardize_by_y=qmc_standardize,
        swd_eps=1e-8,
        device=device,
        dtype=dtype,
        seed=seed,
        log_every=log_every,
        lbfgs_max_iter=lbfgs_max_iter,
        lbfgs_history_size=lbfgs_history_size,
        line_search_fn=line_search_fn,
        theta_mode="per_iter",
        cache_y_projection=False,
        param_init=None,
        param_to_x=param_to_x,
    )


@dataclass
class GFResultUniform:
    x_points: torch.Tensor
    weights: torch.Tensor


def select_points_gf_uniform_using_y_fixed(
    n: int,
    d: int,
    y_fixed: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    seed: int = 0,
    chunk_size: int = 8192,
    qmc_skip: int = 2**3,
    qmc_scramble: bool = True,
    prob_eps: float = 1e-12,
) -> GFResultUniform:
    torch.manual_seed(seed)

    low_d = low.to(device=device, dtype=dtype).view(1, d)
    high_d = high.to(device=device, dtype=dtype).view(1, d)

    u0 = sobol_net(n=n, s=d, skip=qmc_skip, device=device, dtype=dtype, scramble=qmc_scramble, seed=seed)
    x0 = uniform_ppf(u0, low_d, high_d)

    x1 = torch.empty_like(x0)
    for j in range(d):
        uj = rank_to_mid_u(x0[:, j])
        x1[:, j] = uniform_ppf(uj, low_d[:, j], high_d[:, j])

    yx = y_fixed.to(device=device, dtype=dtype)
    Pq1 = estimate_voronoi_probabilities_in_x(yx, x1, chunk_size=chunk_size)
    Pq = normalize_weights(Pq1 + prob_eps)

    def rearrange_once(x_in: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        x_out = torch.empty_like(x_in)
        for j in range(d):
            u = weighted_mid_cdf_1d(x_in[:, j], w)
            x_out[:, j] = uniform_ppf(u, low_d[:, j], high_d[:, j])
        return x_out

    x3 = rearrange_once(x1, Pq)
    x3 = rearrange_once(x3, Pq)

    return GFResultUniform(x_points=x3, weights=Pq)


def gf_discrepancy_uniform_marginals(
    x: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
    weights: torch.Tensor | None = None,
    eps: float = 1e-15,
) -> float:
    assert x.dim() == 2
    n, d = x.shape
    low = low.to(device=x.device, dtype=x.dtype).view(1, d)
    high = high.to(device=x.device, dtype=x.dtype).view(1, d)
    F = uniform_cdf(x, low, high, eps=eps)
    return gf_discrepancy_from_cdf(x, F, weights=weights, eps=eps)


if __name__ == "__main__":
    device_w = "cuda" if torch.cuda.is_available() else "cpu"
    device_g = "cpu"

    d = 2
    Nx = 128
    steps = 10
    y_multiple = 16
    Ny = Nx * y_multiple

    low = torch.tensor([0.0, 0.0], dtype=torch.float32, device=device_w)
    high = torch.tensor([1.0, 1.0], dtype=torch.float32, device=device_w)

    y_fixed = qmc_uniform_samples(
        n=Ny, d=d, device=device_w, dtype=torch.float32,
        low=low, high=high, skip=2**3, scramble=True, seed=0
    )

    Xw = optimize_point_set_against_fixed_y(
        Nx=Nx, d=d, y_fixed=y_fixed, low=low, high=high, y_multiple=y_multiple,
        steps=steps, lbfgs_lr=1.0, num_projections=512, p=2, block_reduce="mean",
        qmc_standardize=True, device=device_w, dtype=torch.float32,
        seed=0, log_every=1, lbfgs_max_iter=20, line_search_fn="strong_wolfe"
    )

    Ww = estimate_voronoi_probabilities_in_x(y_fixed, Xw.to(device_w), chunk_size=8192)
    Ww = normalize_weights(Ww)

    gf_res = select_points_gf_uniform_using_y_fixed(
        n=Nx, d=d, y_fixed=y_fixed, low=low.to(device_g), high=high.to(device_g),
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

    gfd_w_wt = gf_discrepancy_uniform_marginals(Xw, low, high, weights=Ww)
    gfd_g_wt = gf_discrepancy_uniform_marginals(Xg, low, high, weights=Wg)
    gfd_w_unw = gf_discrepancy_uniform_marginals(Xw, low, high, weights=None)
    gfd_g_unw = gf_discrepancy_uniform_marginals(Xg, low, high, weights=None)

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
        title="Wasserstein vs GF (Uniform target)",
        marker_size=16.0,
    )

    out_csv = Path(__file__).resolve().parent / "Uniform.csv"
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
