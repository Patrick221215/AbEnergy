#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
根据 npz 内的 antibody_str_seq / antibody_cdr_def / antibody_chain_ids
自动判断每个 npz 使用的是 IMGT 还是 Chothia 编号。

用法示例：
  python guess_npz_numbering.py --npz_dir /path/to/npz_dir

可选：
  python guess_npz_numbering.py --npz_dir /path/to/npz_dir --patch

如果加上 --patch，会把判断得到的 numbering_scheme 写回 npz 文件里：
  np.savez(..., numbering_scheme="imgt" 或 "chothia")
"""

import os
import sys
import argparse
import pathlib
import numpy as np

# ---- 把项目根目录加到 sys.path，保持跟你现有脚本一致 ----
ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from abx.preprocess.numbering import renumber_ab_seq, get_ab_regions


def _get_chain_seq_and_cdr(data, chain_id_value):
    """
    从 npz 中取出某条链的序列 & cdr_def。
    chain_id_value: 0 = heavy, 1 = light（与你的 merge_chains 约定一致）
    返回: (seq_str, cdr_def_array) 或 (None, None)
    """
    if "antibody_str_seq" not in data or \
       "antibody_cdr_def" not in data or \
       "antibody_chain_ids" not in data:
        return None, None

    seq_all = data["antibody_str_seq"]
    if isinstance(seq_all, np.ndarray):
        seq_all = seq_all.item()
    seq_all = str(seq_all)

    cdr_def_all = np.asarray(data["antibody_cdr_def"]).reshape(-1)
    chain_ids = np.asarray(data["antibody_chain_ids"]).reshape(-1)

    if len(seq_all) != len(cdr_def_all) or len(seq_all) != len(chain_ids):
        # 数据不一致，放弃这个样本
        return None, None

    mask = (chain_ids == chain_id_value)
    if not np.any(mask):
        return None, None

    seq_chain = "".join([aa for aa, m in zip(seq_all, mask) if m])
    cdr_chain = cdr_def_all[mask]

    if len(seq_chain) == 0:
        return None, None

    return seq_chain, cdr_chain


def _cdr_def_with_scheme(seq, scheme, chain_type):
    """
    使用指定 scheme（imgt / chothia）对单条链 seq 重新编号，
    返回 get_ab_regions 得到的 cdr_def 数组。
    chain_type: "H" 或 "L"
    """
    allow = ["H"] if chain_type == "H" else ["K", "L"]

    try:
        anarci_res = renumber_ab_seq(seq, allow=allow, scheme=scheme)
    except Exception as e:
        print(f"[WARN] renumber_ab_seq failed (scheme={scheme}, chain={chain_type}): {e}")
        return None

    domain_numbering = anarci_res.get("domain_numbering", None)
    if domain_numbering is None:
        return None

    try:
        cdr_def = get_ab_regions(domain_numbering, chain_id=chain_type)
    except Exception as e:
        print(f"[WARN] get_ab_regions failed (scheme={scheme}, chain={chain_type}): {e}")
        return None

    return np.asarray(cdr_def)


def _mismatch_ratio(cdr_old, cdr_new):
    """
    计算两条 cdr_def 序列的 mismatch 比例。
    长度不一致则返回 +inf。
    """
    if cdr_new is None or len(cdr_old) != len(cdr_new):
        return float("inf")
    mism = np.sum(cdr_old != cdr_new)
    return mism / float(len(cdr_old)) if len(cdr_old) > 0 else float("inf")


def guess_scheme_for_npz(npz_path, max_mismatch_ratio=0.05):
    """
    对单个 npz 文件判断使用的编号方案（IMGT / Chothia）。
    返回:
      - "imgt" / "chothia" / None(无法可靠判断)
      - 辅助信息字典（方便 debug）
    """
    data = np.load(npz_path, allow_pickle=True)

    # 先看看是不是已经有字段
    if "numbering_scheme" in data:
        scheme = data["numbering_scheme"]
        if isinstance(scheme, np.ndarray):
            scheme = scheme.item()
        scheme = str(scheme).lower()
        return scheme, {"from_field": True}

    # 仅用 Heavy / Light 两条链做判断
    info = {
        "H": {},
        "L": {},
    }

    # Heavy
    seq_H, cdr_H_old = _get_chain_seq_and_cdr(data, chain_id_value=0)
    if seq_H is not None:
        for scheme in ["imgt", "chothia"]:
            cdr_new = _cdr_def_with_scheme(seq_H, scheme, chain_type="H")
            info["H"][scheme] = _mismatch_ratio(cdr_H_old, cdr_new)

    # Light
    seq_L, cdr_L_old = _get_chain_seq_and_cdr(data, chain_id_value=1)
    if seq_L is not None:
        for scheme in ["imgt", "chothia"]:
            cdr_new = _cdr_def_with_scheme(seq_L, scheme, chain_type="L")
            info["L"][scheme] = _mismatch_ratio(cdr_L_old, cdr_new)

    # 根据已有链综合判断
    scores = {"imgt": [], "chothia": []}

    for chain_type in ["H", "L"]:
        for scheme in ["imgt", "chothia"]:
            if scheme in info[chain_type]:
                scores[scheme].append(info[chain_type][scheme])

    # 如果两条链都拿不到信息，放弃
    if len(scores["imgt"]) == 0 and len(scores["chothia"]) == 0:
        return None, info

    # 对每个 scheme 取平均 mismatch
    avg_scores = {}
    for scheme in ["imgt", "chothia"]:
        if len(scores[scheme]) == 0:
            avg_scores[scheme] = float("inf")
        else:
            avg_scores[scheme] = float(np.mean(scores[scheme]))

    best_scheme = min(avg_scores, key=avg_scores.get)
    best_mismatch = avg_scores[best_scheme]

    info["avg_scores"] = avg_scores
    info["best_scheme"] = best_scheme
    info["best_mismatch"] = best_mismatch

    # mismatch 太大则认为无法可靠判断
    if best_mismatch > max_mismatch_ratio:
        return None, info

    return best_scheme, info


def main():
    parser = argparse.ArgumentParser(
        description="根据 npz 内容自动判断 IMGT / Chothia 编号方案"
    )
    parser.add_argument(
        "--npz_dir",
        type=str,
        required=True,
        help="包含 .npz 文件的目录",
    )
    parser.add_argument(
        "--patch",
        action="store_true",
        help="如指定，则将判断结果写回 npz（增加 numbering_scheme 字段）",
    )
    parser.add_argument(
        "--max_mismatch_ratio",
        type=float,
        default=0.05,
        help="允许的最大 mismatch 比例，超过则视为无法判断（默认 0.05）",
    )
    args = parser.parse_args()

    npz_dir = os.path.abspath(args.npz_dir)
    print("[INFO] npz_dir =", npz_dir)
    print("[INFO] patch   =", args.patch)
    print("[INFO] max_mismatch_ratio =", args.max_mismatch_ratio)

    files = sorted([f for f in os.listdir(npz_dir) if f.endswith(".npz")])
    print(f"[INFO] 找到 npz 文件 {len(files)} 个")

    cnt_imgt = 0
    cnt_chothia = 0
    cnt_unknown = 0

    for fname in files:
        path = os.path.join(npz_dir, fname)
        scheme, info = guess_scheme_for_npz(path, max_mismatch_ratio=args.max_mismatch_ratio)

        if scheme is None:
            cnt_unknown += 1
            print(f"[UNKN] {fname} -> 无法可靠判断, info={info.get('avg_scores', info)}")
            continue

        if scheme == "imgt":
            cnt_imgt += 1
        elif scheme == "chothia":
            cnt_chothia += 1

        avg_scores = info.get("avg_scores", {})
        print(
            f"[OK]   {fname} -> {scheme.upper()} "
            f"(imgt_mismatch={avg_scores.get('imgt', 'NA'):.4f}, "
            f"chothia_mismatch={avg_scores.get('chothia', 'NA'):.4f})"
        )

        # 如需写回 npz
        if args.patch:
            data = np.load(path, allow_pickle=True)
            feature = {k: data[k] for k in data.files}
            feature["numbering_scheme"] = scheme
            np.savez(path, **feature)

    print("\n[SUMMARY]")
    print(f"  IMGT     : {cnt_imgt}")
    print(f"  Chothia  : {cnt_chothia}")
    print(f"  Unknown  : {cnt_unknown}")


if __name__ == "__main__":
    main()
