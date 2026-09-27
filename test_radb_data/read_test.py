#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
根据 dyMEAN 的 RAbD test.json，生成 ABx 使用的 RAbD_test.idx

输入:  dyMEAN 格式的 test.json
  每行类似:
  {"pdb": "4fqj",
   "heavy_chain": "H",
   "light_chain": "L",
   "antigen_chains": ["A"],
   ...}

输出: ABx 格式的 idx 文件 (每行一个 npz 基名)
  例如:
    4fqj_H_L_A
    8tp5_E_F_C
    ...

注意:
- AG 部分使用 ''.join(antigen_chains)，即 ["A","B"] -> "AB"
- pdb 一律转为小写，和你现在的 npz 命名保持一致
"""

import os
import json
import argparse


def parse_args():
    p = argparse.ArgumentParser(
        description="将 dyMEAN RAbD test.json 转成 ABx 的 RAbD_test.idx"
    )
    p.add_argument(
        "--input",
        type=str,
        required=True,
        help="dyMEAN 格式的 test.json 路径，比如 ./project/dyMEAN/all_data/RAbD/test.json",
    )
    p.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出的 idx 路径，比如 ./project/ABx/data/RAbD_test.idx",
    )
    return p.parse_args()


def load_lines(json_path):
    with open(json_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield line


def main():
    args = parse_args()

    in_path = os.path.abspath(args.input)
    out_path = os.path.abspath(args.output)

    print(f"[INFO] 读取 dyMEAN test.json: {in_path}")
    print(f"[INFO] 输出 ABx RAbD_test.idx: {out_path}")

    names = []
    pdb_seen = set()

    for line in load_lines(in_path):
        item = json.loads(line)

        # 1) 取出字段
        pdb = str(item["pdb"]).strip().lower()          # 和 ABx 一致，使用小写
        h = str(item["heavy_chain"]).strip()
        l = str(item["light_chain"]).strip()
        ag_chains = item.get("antigen_chains", [])

        # 支持字符串/列表两种格式，防止意外
        if isinstance(ag_chains, str):
            # 兼容 "A,B" / "A|B" 等写法
            if "|" in ag_chains:
                ag_chains = [s.strip() for s in ag_chains.split("|") if s.strip()]
            elif "," in ag_chains:
                ag_chains = [s.strip() for s in ag_chains.split(",") if s.strip()]
            else:
                ag_chains = [ag_chains.strip()] if ag_chains.strip() else []

        if not ag_chains:
            print(f"[WARN] {pdb} 没有 antigen_chains，跳过这一条")
            continue

        ag = "".join(ag_chains)  # 和 ABx 保存 npz 时的 antigen_chain_id 命名一致

        name = f"{pdb}_{h}_{l}_{ag}"
        names.append(name)
        pdb_seen.add(pdb)

    # 写出 idx
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for n in names:
            f.write(n + "\n")

    print(f"[INFO] 共写出 {len(names)} 条到 {out_path}")
    print(f"[INFO] 涉及的 PDB 数量 = {len(pdb_seen)}")
    if len(names) != len(pdb_seen):
        print(
            f"[INFO] 注意: 同一 PDB 可能有多条样本 "
            f"(条目数 {len(names)} vs PDB 数 {len(pdb_seen)})"
        )


if __name__ == "__main__":
    main()
