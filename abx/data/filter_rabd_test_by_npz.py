#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
根据现有 npz_dir，过滤 dyMEAN 的 RAbD_test.idx，只保留真正存在 .npz 的样本。

用法示例：
  python filter_rabd_test_by_npz.py \
      --npz_dir /path/to/your/npz_dir \
      --rabd_idx_dymean ./project/dyMEAN/all_data/RAbD/RAbD_test.idx \
      --out_idx ./project/abx_splits/RAbD_test_abx.idx
"""

import os
import argparse


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--npz_dir", type=str, required=True,
                   help="当前 AbX 预处理生成的 .npz 目录")
    p.add_argument("--rabd_idx_dymean", type=str, required=True,
                   help="dyMEAN 的 RAbD_test.idx 路径")
    p.add_argument("--out_idx", type=str, required=True,
                   help="过滤后的 idx 输出路径（例如 RAbD_test_abx.idx）")
    return p.parse_args()


def main():
    args = parse_args()

    npz_dir = os.path.abspath(args.npz_dir)
    rabd_idx_path = os.path.abspath(args.rabd_idx_dymean)
    out_path = os.path.abspath(args.out_idx)

    # 1) 收集所有现有的 npz 名（去掉后缀）
    npz_names = {
        f[:-4]
        for f in os.listdir(npz_dir)
        if f.endswith(".npz")
    }

    print(f"[INFO] npz_dir = {npz_dir}")
    print(f"[INFO] 发现 npz 数量 = {len(npz_names)}")

    # 2) 读取 dyMEAN 的 RAbD_test.idx
    with open(rabd_idx_path) as f:
        lines = [l.strip() for l in f if l.strip()]

    print(f"[INFO] dyMEAN RAbD_test.idx 条数 = {len(lines)}")

    exist = []
    missing = []

    for name in lines:
        # dyMEAN 的名字格式和我们 npz 完全一致，比如 "4fqj_H_L_A"
        if name in npz_names:
            exist.append(name)
        else:
            missing.append(name)

    # 3) 写出只包含“存在 npz”的新 idx
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for n in exist:
            f.write(n + "\n")

    print(f"[INFO] 新的 RAbD_test_abx.idx 写到: {out_path}")
    print(f"[INFO] 可用 test 样本数 = {len(exist)}")
    print(f"[INFO] 缺失样本数 = {len(missing)}")

    if missing:
        print("[WARN] 下列样本在 npz_dir 中找不到 .npz：")
        for n in missing:
            print("   -", n)


if __name__ == "__main__":
    main()
