import numpy as _np
import math as _math
from typing import Sequence
import torch

def _safe_np(x):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    elif isinstance(x, (list, tuple)):
        x = _np.array(list(map(_tofloat, x)))
    elif isinstance(x, (float, int)):
        x = _np.array([x], dtype=_np.float32)
    return x

def _acc_pair(a: Sequence[float], b: Sequence[float]) -> float:
    a = _safe_np(a); b = _safe_np(b)
    if len(a)==0 or len(b)==0: return _np.nan
    return float(_np.mean(a < b))

def _triplet_ok(e_gt, e_pred, e_noisy) -> float:
    e_gt = _safe_np(e_gt); e_pred=_safe_np(e_pred); e_noisy=_safe_np(e_noisy)
    n = min(len(e_gt), len(e_pred), len(e_noisy))
    if n==0: return _np.nan
    return float(_np.mean((e_gt[:n] < e_pred[:n]) & (e_pred[:n] < e_noisy[:n])))

def _margin_ok(a, b, m: float) -> float:
    a = _safe_np(a); b = _safe_np(b)
    n = min(len(a), len(b))
    if n==0: return _np.nan
    return float(_np.mean((b[:n] - a[:n]) >= m))

def _spearman(x, y) -> float:
    x = _safe_np(x); y = _safe_np(y)
    n = min(len(x), len(y))
    if n < 2: return _np.nan
    rx = _np.argsort(_np.argsort(x[:n]))  # 0..n-1
    ry = _np.argsort(_np.argsort(y[:n]))
    # 皮尔逊相关系数 of ranks
    x0 = rx - rx.mean(); y0 = ry - ry.mean()
    denom = (_np.linalg.norm(x0)*_np.linalg.norm(y0) + 1e-12)
    return float((x0@y0)/denom)

def _pearson(x, y) -> float:
    x = _safe_np(x); y = _safe_np(y)
    n = min(len(x), len(y))
    if n < 2: return _np.nan
    x0 = x[:n] - x[:n].mean(); y0 = y[:n] - y[:n].mean()
    denom = (_np.linalg.norm(x0)*_np.linalg.norm(y0) + 1e-12)
    return float((x0@y0)/denom)

def _topk_mean(vals, k_list=(1,5,10)):
    v = _safe_np(vals)
    if len(v)==0: return {f"k{k}": _np.nan for k in k_list}
    idx = _np.argsort(v)  # 升序（小=好）
    out = {}
    for k in k_list:
        kk = min(k, len(v))
        out[f"k{k}"] = float(v[idx[:kk]].mean()) if kk>0 else _np.nan
    return out

def _add_hist(writer, tag, vec, step):
    v = _safe_np(vec)
    if len(v) == 0: return
    # 限制样本数，避免超大 batch 直方图撑爆事件文件
    if len(v) > 4096:
        v = v[_np.random.choice(len(v), 4096, replace=False)]
    writer.add_histogram(tag, v, step)

def _add_scatter(writer, tag, xs, ys, step, every_n:int, current:int):
    if current % every_n != 0: return
    import matplotlib.pyplot as _plt
    x = _safe_np(xs);   y = _safe_np(ys)
    if min(len(x), len(y)) == 0: return
    # 采样一些点，避免图太大
    n = min(len(x), len(y))
    if n > 2000:
        sel = _np.random.choice(n, 2000, replace=False)
        x, y = x[sel], y[sel]
    fig = _plt.figure()
    _plt.scatter(x, y, s=6, alpha=0.5)
    _plt.xlabel("RMSD"); _plt.ylabel("Energy")
    writer.add_figure(tag, fig, step)
    _plt.close(fig)


def _tofloat(x):
    return float(x.detach().item() if torch.is_tensor(x) else x)



