#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
构建 train.idx / val.idx，固定 RAbD_test.idx 为测试集。

- RAbD_test 作为 benchmark/test 集
- 其余样本 + RAbD 一起做 CDR-H3 (H3) 的 mmseqs 聚类 (--min-seq-id 0.5)
- 含 benchmark 样本的簇视为 benchmark_clusters
- 其他簇视为 other_clusters，从中按簇随机划分 train / valid
- train.idx / val.idx 只包含 other_clusters 的样本名 (pdb_H_L_AG)

要求:
- npz_dir: 预处理生成的 *.npz 路径
- rabd_idx: 现有的 RAbD_test.idx 文件 (每行一个 pdb_H_L_AG)
- 系统 PATH 里有 mmseqs 可执行文件
"""

import os
import argparse
import shutil
import subprocess
from collections import defaultdict
import sys
import pathlib

# ---- 关键：把项目根目录加到 sys.path ----
ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    
from abx.common import residue_constants

import numpy as np

MMSEQS_BIN = "./anaconda3/envs/dymean0/bin/mmseqs"  # 顶部加一行全局常量


def run_cmd(cmd, cwd=None):
    print(f"[CMD] {' '.join(cmd)}")
    res = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    print(res.stdout)
    if res.returncode != 0:
        raise RuntimeError(
            f"Command failed with code {res.returncode}: {' '.join(cmd)}"
        )
    return res.stdout


def load_rabd_info(rabd_idx_path, all_npz_names):
    """
    读取 RAbD_test.idx，返回：
      - exist_names: 在 npz_dir 中确实存在的测试样本名集合
      - rabd_pdbs:   RAbD_test.idx 中涉及到的所有 PDB 编码集合
    """
    with open(rabd_idx_path) as f:
        names = [l.strip() for l in f if l.strip()]

    names_set = set(names)
    npz_set = set(all_npz_names)

    exist = sorted(list(names_set & npz_set))
    missing = sorted(list(names_set - npz_set))

    print(f"[INFO] RAbD_test.idx 总条目 = {len(names_set)}")
    print(f"[INFO] 在 npz_dir 中实际存在的 RAbD 条目 = {len(exist)}")
    if missing:
        print(f"[WARN] 下列 {len(missing)} 个 RAbD 样本在 npz_dir 下找不到 .npz：")
        for n in missing[:20]:
            print("   -", n)

    # ★ 关键：按 PDB 维度记录所有 test 用到的 pdb 编码
    rabd_pdbs = {n.split("_", 1)[0] for n in names_set}
    print(f"[INFO] RAbD_test 涉及的 PDB 数量 = {len(rabd_pdbs)}")

    return set(exist), rabd_pdbs



def extract_h3_seq_from_npz(npz_dir, names, cdr_name="H3"):
    """
    从 .npz 中用 antibody_str_seq + antibody_cdr_def 提取 CDR-H3 序列。
    返回 items: list[dict], 每个 dict:
      {
        'name': 'pdb_H_L_AG',
        'pdb': 'pdb',                 # 新增：PDB ID（从 name 里解析）
        'seq': 'CDRH3SEQ',
        'is_benchmark': True/False
      }
    """
    h3_enum = residue_constants.cdr_str_to_enum[cdr_name]  # 'H3' -> 枚举值

    items = []
    skipped_no_h3 = 0

    for name, is_benchmark in names:
        path = os.path.join(npz_dir, name + ".npz")
        try:
            data = np.load(path, allow_pickle=True)
        except Exception as e:
            print(f"[WARN] 加载 {path} 失败，跳过: {e}")
            continue

        if (
            "antibody_str_seq" not in data
            or "antibody_cdr_def" not in data
            or "antibody_chain_ids" not in data
        ):
            print(f"[WARN] {name}.npz 缺少 antibody_str_seq / antibody_cdr_def / antibody_chain_ids，跳过")
            continue

        ab_seq = data["antibody_str_seq"]
        if isinstance(ab_seq, np.ndarray):
            ab_seq = ab_seq.item()
        ab_seq = str(ab_seq)

        cdr_def = np.asarray(data["antibody_cdr_def"]).reshape(-1)
        chain_ids = np.asarray(data["antibody_chain_ids"]).reshape(-1)

        if len(cdr_def) != len(ab_seq) or len(chain_ids) != len(ab_seq):
            print(
                f"[WARN] {name}.npz: cdr_def/chain_ids 长度与抗体序列不一致，跳过 "
                f"({len(cdr_def)}, {len(chain_ids)} vs {len(ab_seq)})"
            )
            continue

        # heavy chain id = 0, 参见 make_ab_data_from_mmcif.merge_chains 的实现
        mask = (cdr_def == h3_enum) & (chain_ids == 0)
        idxs = np.nonzero(mask)[0]
        if len(idxs) == 0:
            skipped_no_h3 += 1
            continue

        h3_seq = "".join(ab_seq[i] for i in idxs)
        if len(h3_seq) == 0:
            skipped_no_h3 += 1
            continue

        # 新增：从 name 里解析 pdb id
        # 假设你的 .npz 命名是 pdb_H_L_AG 这种，下划线前一段就是 pdb
        pdb_id = name.split("_", 1)[0]

        items.append(
            {
                "name": name,
                "pdb": pdb_id,          # 新字段
                "seq": h3_seq,
                "is_benchmark": is_benchmark,
            }
        )

    print(f"[INFO] 共构造 {len(items)} 个样本的 H3 序列，"
          f"无 H3 / 异常而被跳过 {skipped_no_h3} 个")

    if len(items) == 0:
        raise RuntimeError("没有任何可用样本，检查 .npz 内容或 cdr_name 是否正确")

    return items


def mmseqs_cluster(items, tmp_dir, min_seq_id=0.5):
    """
    使用 mmseqs 对 items 里的 H3 序列做聚类（PDB 粒度）。
    items: list[dict]，每个 dict 至少包含:
      - 'name': 'pdb_H_L_AG'
      - 'pdb': 'pdb'
      - 'seq': 'CDRH3SEQ'
      - 'is_benchmark': bool

    返回:
      clu2idx: dict[cluster_id] -> list[item_index]
        这里的 item_index 是 items 的下标（0..len(items)-1），
        每个簇包含所有属于簇内 PDB 的样本（可能多个 npz）。
    """
    if os.path.exists(tmp_dir):
        raise RuntimeError(
            f"临时目录 {tmp_dir} 已存在，为避免误删不继续执行，请手动删除"
        )
    os.makedirs(tmp_dir, exist_ok=False)

    fasta = os.path.join(tmp_dir, "seq.fasta")
    db = os.path.join(tmp_dir, "DB")
    db_clu = os.path.join(tmp_dir, "DB_clu")
    tsv = os.path.join(tmp_dir, "DB_clu.tsv")

    # 1) 聚合到 PDB 级别：一个 pdb 一条序列（代表 H3）
    pdb2seq = {}
    pdb2idxs = defaultdict(list)

    for idx, it in enumerate(items):
        pdb_id = it["pdb"]
        pdb2idxs[pdb_id].append(idx)
        # 如果同一个 pdb 有多条，只取第一条作为聚类代表即可
        if pdb_id not in pdb2seq:
            pdb2seq[pdb_id] = it["seq"]

    # 写 FASTA：header = pdb（和 dyMEAN 对齐）
    with open(fasta, "w") as f:
        for pdb_id, seq in pdb2seq.items():
            f.write(f">{pdb_id}\n{seq}\n")

    # 2) 运行 mmseqs
    run_cmd([MMSEQS_BIN, "createdb", fasta, db])
    run_cmd(
        [
            MMSEQS_BIN,
            "cluster",
            db,
            db_clu,
            tmp_dir,
            "--min-seq-id",
            str(min_seq_id),
        ]
    )
    run_cmd([MMSEQS_BIN, "createtsv", db, db, db_clu, tsv])

    # 3) 解析 tsv: cluster_id \t pdb_id
    with open(tsv) as fin:
        lines = [l.strip() for l in fin if l.strip()]

    # cluster -> pdb 列表
    clu2pdbs = defaultdict(list)
    for line in lines:
        clu, pdb_id = line.split("\t")
        clu2pdbs[clu].append(pdb_id)

    # 4) cluster -> item_index（把同一 pdb 下的所有样本 index 都放进来）
    clu2idx = defaultdict(list)
    for clu, pdb_list in clu2pdbs.items():
        for pdb_id in pdb_list:
            if pdb_id not in pdb2idxs:
                print(f"[WARN] PDB {pdb_id} 在 pdb2idxs 中找不到，跳过")
                continue
            clu2idx[clu].extend(pdb2idxs[pdb_id])

    sizes = [len(v) for v in clu2idx.values()]
    print(
        f"[INFO] 聚类完成: 簇数={len(clu2idx)}, "
        f"成员数 mean={np.mean(sizes):.2f}, min={np.min(sizes)}, max={np.max(sizes)}"
    )

    shutil.rmtree(tmp_dir)
    return clu2idx

def split_with_benchmark(items, clu2idx, valid_ratio=0.1, seed=2024):
    """
    - benchmark_clusters: 含有任意 is_benchmark=True 的簇
    - other_clusters: 其他簇
    - 在 other_clusters 上按 valid_ratio 划分 train / valid
    """
    np.random.seed(seed)

    is_benchmark = [it["is_benchmark"] for it in items]

    benchmark_clusters = []
    other_clusters = []

    for c, idx_list in clu2idx.items():
        flag = any(is_benchmark[i] for i in idx_list)
        if flag:
            benchmark_clusters.append(c)
        else:
            other_clusters.append(c)

    print(
        f"[INFO] benchmark_clusters={len(benchmark_clusters)} "
        f"(含 RAbD_test 样本的簇), other_clusters={len(other_clusters)}"
    )

    # 在 other_clusters 上做 train/valid
    np.random.shuffle(other_clusters)
    n_other = len(other_clusters)
    n_valid = int(n_other * valid_ratio)
    n_train = n_other - n_valid
    if n_train <= 0:
        raise ValueError(
            f"train 簇数 <= 0，检查 valid_ratio={valid_ratio}, n_other={n_other}"
        )

    train_clusters = other_clusters[:n_train]
    valid_clusters = other_clusters[n_train:]

    print(
        f"[INFO] train_clusters={len(train_clusters)}, "
        f"valid_clusters={len(valid_clusters)}"
    )

    splits = {
        "train": train_clusters,
        "valid": valid_clusters,
        "benchmark": benchmark_clusters,
    }
    return splits


def write_train_val_idx(items, clu2idx, splits, out_dir):
    """
    写出 train.idx / val.idx。
    注意：只写 non-benchmark 样本，benchmark 用现有的 RAbD_test.idx。
    """
    os.makedirs(out_dir, exist_ok=True)

    # 建一个索引到 (name, is_benchmark) 的映射
    names = [it["name"] for it in items]
    is_benchmark = [it["is_benchmark"] for it in items]

    # train
    train_file = os.path.join(out_dir, "train.idx")
    cnt_train = 0
    with open(train_file, "w") as f:
        for c in splits["train"]:
            for i in clu2idx[c]:
                if is_benchmark[i]:
                    continue
                f.write(names[i] + "\n")
                cnt_train += 1
    print(f"[INFO] 写出 {train_file}, 样本数={cnt_train}")

    # val
    val_file = os.path.join(out_dir, "val.idx")
    cnt_val = 0
    with open(val_file, "w") as f:
        for c in splits["valid"]:
            for i in clu2idx[c]:
                if is_benchmark[i]:
                    continue
                f.write(names[i] + "\n")
                cnt_val += 1
    print(f"[INFO] 写出 {val_file}, 样本数={cnt_val}")

    # benchmark 簇信息只做 debug，不写 idx（你已有 RAbD_test.idx）
    total_benchmark = 0
    for c in splits["benchmark"]:
        for i in clu2idx[c]:
            if is_benchmark[i]:
                total_benchmark += 1
    print(f"[INFO] benchmark 簇中实际 RAbD_test 样本数={total_benchmark}")


def parse_args():
    p = argparse.ArgumentParser(
        description="在 构建 train.idx / val.idx，固定 RAbD_test.idx 为测试集"
    )
    p.add_argument(
        "--npz_dir",
        type=str,
        required=True,
        help="预处理生成的 .npz 目录",
    )
    p.add_argument(
        "--rabd_idx",
        type=str,
        required=True,
        help="现有 RAbD_test.idx 文件路径（每行一个 pdb_H_L_AG）",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="输出 train.idx / val.idx 的目录",
    )
    p.add_argument(
        "--tmp_dir",
        type=str,
        default="./tmp_mmseqs_abx_rabd",
        help="mmseqs 使用的临时目录（默认 ./tmp_mmseqs_abx_rabd，需不存在）",
    )
    p.add_argument(
        "--valid_ratio",
        type=float,
        default=0.1,
        help="验证集簇比例 (default 0.1)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    p.add_argument(
        "--min_seq_id",
        type=float,
        default=0.4,
        help="mmseqs 聚类的 min-seq-id，默认 0.5",
    )
    return p.parse_args()


def main():
    args = parse_args()

    npz_dir = os.path.abspath(args.npz_dir)
    out_dir = os.path.abspath(args.out_dir)
    tmp_dir = os.path.abspath(args.tmp_dir)

    print("[INFO] 参数：")
    print(f"  npz_dir     = {npz_dir}")
    print(f"  rabd_idx    = {args.rabd_idx}")
    print(f"  out_dir     = {out_dir}")
    print(f"  tmp_dir     = {tmp_dir}")
    print(f"  valid_ratio = {args.valid_ratio}")
    print(f"  seed        = {args.seed}")
    print(f"  min_seq_id  = {args.min_seq_id}")

    # 1) 列出所有 npz 名称
    all_npz = sorted(
        [
            f[:-4]
            for f in os.listdir(npz_dir)
            if f.endswith(".npz")
        ]
    )
    print(f"[INFO] 在 npz_dir 下发现 .npz 样本数 = {len(all_npz)}")

    # 2) 读取 RAbD_test.idx，确定：
    #    - 哪些 test 样本名真实存在
    #    - RAbD_test 里一共有哪些 PDB 编码
    rabd_exist_names, rabd_pdbs = load_rabd_info(args.rabd_idx, all_npz)

    # 构造 (name, is_benchmark) 列表：
    # ★ 关键：按 PDB 维度标 benchmark，只要这个 PDB 在 RAbD_test 里出现过，
    #         该 PDB 的任何 npz（不管 H/L/Ag 怎么组合）都标记为 is_benchmark=True
    names_flag = []
    for n in all_npz:
        pdb_id = n.split("_", 1)[0]
        is_benchmark = (pdb_id in rabd_pdbs)
        names_flag.append((n, is_benchmark))

    # 3) 从 .npz 提取 CDR-H3 序列
    items = extract_h3_seq_from_npz(npz_dir, names_flag, cdr_name="H3")

    # 4) mmseqs 聚类
    clu2idx = mmseqs_cluster(items, tmp_dir=tmp_dir, min_seq_id=args.min_seq_id)

    # 5) 按 benchmark 逻辑划分簇 -> train / valid
    splits = split_with_benchmark(
        items, clu2idx, valid_ratio=args.valid_ratio, seed=args.seed
    )

    # 6) 写出 train.idx / val.idx
    write_train_val_idx(items, clu2idx, splits, out_dir=out_dir)

    print("[INFO] 完成!")


if __name__ == "__main__":
    main()
