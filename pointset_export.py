from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional, Union

import torch


def _normalize_weights(w: torch.Tensor, eps: float = 1e-18) -> torch.Tensor:
    """Normalize weights to sum to 1. If the sum is too small, fall back to uniform."""
    w = w.detach().flatten()
    s = w.sum()
    if s <= eps:
        return torch.full_like(w, 1.0 / w.numel())
    return w / s


def _to_cpu_2d(x: torch.Tensor, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    x = x.detach()
    if x.dim() == 1:
        x = x.view(-1, 1)
    assert x.dim() == 2, "点集必须是二维张量 (N,d)"
    return x.to(device="cpu", dtype=dtype)


def export_point_sets_to_csv(
    csv_path: Union[str, Path],
    *,
    y_qmc: Optional[torch.Tensor] = None,
    w_qmc: Optional[torch.Tensor] = None,
    x_swd: Optional[torch.Tensor] = None,
    w_swd: Optional[torch.Tensor] = None,
    x_gf: Optional[torch.Tensor] = None,
    w_gf: Optional[torch.Tensor] = None,
    method_qmc: str = "QMC",
    method_swd: str = "SWD",
    method_gf: str = "GF",
    float_fmt: str = "{:.10g}",
    weight_eps: float = 1e-18,
) -> Path:
    """Export multiple point sets into ONE CSV.

    Each row:
        method, index, weight, x1, x2, ..., xd

    Notes:
      - If the corresponding weights are None, uniform weights are used.
      - SWD is commonly treated as equal-weight points, so w_swd can be None.
      - QMC (y_fixed) is exported as method_qmc.
      - All tensors will be moved to CPU and stored as float64 in the file.
    """
    csv_path = Path(csv_path)

    sets = []
    if y_qmc is not None:
        sets.append((method_qmc, y_qmc, w_qmc))
    if x_swd is not None:
        sets.append((method_swd, x_swd, w_swd))
    if x_gf is not None:
        sets.append((method_gf, x_gf, w_gf))
    if not sets:
        raise ValueError("至少需要提供一个点集（y_qmc / x_swd / x_gf）")

    first_x = _to_cpu_2d(sets[0][1])
    d = first_x.size(1)

    prepared = []
    for method, x, w in sets:
        x_c = _to_cpu_2d(x)
        assert x_c.size(1) == d, "不同点集的维度 d 必须一致"
        n = x_c.size(0)

        if w is None:
            w_c = torch.full((n,), 1.0 / n, dtype=torch.float64)
        else:
            w_c = _normalize_weights(w.to(device="cpu", dtype=torch.float64), eps=weight_eps)
            assert w_c.numel() == n, f"{method} 权重长度必须与点数一致"

        prepared.append((method, x_c, w_c))

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    header = ["method", "index", "weight"] + [f"x{j+1}" for j in range(d)]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for method, x_c, w_c in prepared:
            n = x_c.size(0)
            for i in range(n):
                row = [method, i, float_fmt.format(float(w_c[i].item()))]
                row += [float_fmt.format(float(x_c[i, j].item())) for j in range(d)]
                writer.writerow(row)

    return csv_path
