# cacl_energy_ddg.py (支持 relaxed / raw；output 默认写回 pdb_dir)

import os
import re
import argparse
import multiprocessing as mp
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

import pyrosetta
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover


import hashlib
_WORKER_INIT = False
_SCORE_FXN = None

def _stable_seed_from_str(s: str, base: int = 20250101) -> int:
    """Stable 32-bit seed from string (reproducible across runs/machines)."""
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return (int(h[:8], 16) ^ base) & 0x7fffffff

def _set_rosetta_seed(seed: int):
    """
    Re-seed Rosetta RNG per task to make packer deterministic even in multiprocessing.
    """
    try:
        pyrosetta.rosetta.basic.random.init_random_generators(seed, "mt19937")
    except Exception as e:
        logging.warning(f"[Seed] init_random_generators failed: {type(e).__name__}: {e}")

def _init_pyrosetta_worker(init_opts: str, base_seed: int):
    """
    Called once per spawned worker process.
    """
    global _WORKER_INIT, _SCORE_FXN
    if _WORKER_INIT:
        return
    pyrosetta.init(init_opts)
    _SCORE_FXN = pyrosetta.create_score_function("ref2015")
    _WORKER_INIT = True


# --- 1. 初始化 PyRosetta ---
def _pyrosetta_interface_energy(pdb_path: str) -> float:
    try:
        interface = _parse_interface_from_filename(pdb_path)
        if interface is None:
            logging.warning(f"[Skip] Cannot parse interface from filename: {os.path.basename(pdb_path)}")
            return np.nan

        # ===== [ADD] per-task deterministic seed =====
        # Use pdb_path as identity; if you want gen/ref share same seed, use basename only.
        seed = _stable_seed_from_str(os.path.abspath(pdb_path))
        _set_rosetta_seed(seed)
        # ===== [ADD END] =====

        pose = pyrosetta.pose_from_pdb(pdb_path)

        mover = InterfaceAnalyzerMover()
        mover.set_interface(interface)

        # ===== [CHANGE] reuse per-worker scorefxn if available =====
        scorefxn = _SCORE_FXN if _SCORE_FXN is not None else pyrosetta.create_score_function("ref2015")
        mover.set_scorefunction(scorefxn)
        # ===== [CHANGE END] =====

        mover.set_pack_separated(True)
        mover.apply(pose)

        return float(pose.scores.get("dG_separated", np.nan))
    except Exception as e:
        logging.warning(f"Energy calculation failed for {os.path.basename(pdb_path)}: {e}")
        return np.nan



# --- 2. 任务封装 ---
@dataclass
class EnergyTask:
    pred_path: str
    ref_path: str
    name: str
    run_folder: str
    variant: str  # "relaxed" or "raw"
    scores: Dict[str, Any] = field(default_factory=dict)


def _strip_relaxed_suffix(stem: str) -> str:
    # 统一去掉 relaxed 后缀（兼容 _relaxed / -relaxed 等）
    if stem.endswith("_relaxed"):
        return stem[: -len("_relaxed")]
    return stem


def _parse_interface_from_filename(pdb_path: str) -> Optional[str]:
    """
    从文件名解析 interface: heavy+light _ antigen_ids
    约定：{code}_{H}_{L}_{AG}.pdb 或 {code}_{H}_{L}_{AG}_relaxed.pdb
    """
    stem = os.path.splitext(os.path.basename(pdb_path))[0]
    stem = _strip_relaxed_suffix(stem)
    parts = stem.split("_")
    if len(parts) < 4:
        return None
    # 只取最后三段更稳（防止 code 内含 '_' 的极端情况）
    heavy_id, light_id, antigen_ids = parts[-3], parts[-2], parts[-1]
    antibody = f"{heavy_id}{light_id}"
    return f"{antibody}_{antigen_ids}"


def calculate_ddG(task: EnergyTask) -> EnergyTask:
    dG_gen = _pyrosetta_interface_energy(task.pred_path)
    dG_ref = _pyrosetta_interface_energy(task.ref_path)
    ddG = dG_gen - dG_ref if not (np.isnan(dG_gen) or np.isnan(dG_ref)) else np.nan

    task.scores.update({
        "dG_gen": dG_gen,
        "dG_ref": dG_ref,
        "ddG": ddG,
    })
    return task


def _load_name_set(name_idx_path: str) -> set:
    df = pd.read_csv(name_idx_path, names=["name"], header=None)
    return set(df["name"].astype(str).tolist())


def _choose_pred_file(run_folder_path: str, name: str, mode: str) -> List[Tuple[str, str]]:
    """
    返回 [(pred_path, variant), ...]
    mode:
      - "auto": 优先 relaxed，找不到用 raw
      - "relaxed": 只用 relaxed
      - "raw": 只用 raw
      - "both": 两者都算（如果存在）
    """
    relaxed_path = os.path.join(run_folder_path, f"{name}_relaxed.pdb")
    raw_path = os.path.join(run_folder_path, f"{name}.pdb")

    out = []
    if mode == "relaxed":
        if os.path.exists(relaxed_path):
            out.append((relaxed_path, "relaxed"))
        return out

    if mode == "raw":
        if os.path.exists(raw_path):
            out.append((raw_path, "raw"))
        return out

    if mode == "both":
        if os.path.exists(relaxed_path):
            out.append((relaxed_path, "relaxed"))
        if os.path.exists(raw_path):
            out.append((raw_path, "raw"))
        return out

    # auto
    if os.path.exists(relaxed_path):
        out.append((relaxed_path, "relaxed"))
    elif os.path.exists(raw_path):
        out.append((raw_path, "raw"))
    return out


def prepare_tasks(pred_root_dir: str, name_idx_path: str, mode: str) -> List[EnergyTask]:
    """
    pred_root_dir:
      design/
        reference/
          {name}.pdb
        0000/
          {name}.pdb or {name}_relaxed.pdb
        0001/
          ...
    """
    tasks: List[EnergyTask] = []
    pred_root_dir = os.path.abspath(pred_root_dir)

    reference_dir = os.path.join(pred_root_dir, "reference")
    if not os.path.isdir(reference_dir):
        raise FileNotFoundError(f"Reference directory not found: {reference_dir}")

    target_names = _load_name_set(name_idx_path)

    all_subdirs = [d for d in os.listdir(pred_root_dir) if os.path.isdir(os.path.join(pred_root_dir, d))]
    run_folders = sorted([d for d in all_subdirs if d != "reference"])
    logging.info(f"Found {len(run_folders)} run folders in '{pred_root_dir}'.")

    # 加速：先把 reference 里存在的 name 过滤掉，避免无意义匹配
    ref_exist = set()
    for n in target_names:
        if os.path.exists(os.path.join(reference_dir, f"{n}.pdb")):
            ref_exist.add(n)
    if not ref_exist:
        logging.warning("No reference pdb matched any name in idx.")
        return []

    for run_folder in run_folders:
        run_folder_path = os.path.join(pred_root_dir, run_folder)

        # 加速：扫描该目录所有 pdb，推回 name 候选
        # 支持 name.pdb 与 name_relaxed.pdb
        present_names = set()
        for fn in os.listdir(run_folder_path):
            if not fn.endswith(".pdb"):
                continue
            stem = os.path.splitext(fn)[0]
            stem = _strip_relaxed_suffix(stem)
            present_names.add(stem)

        common = (present_names & ref_exist)
        if not common:
            continue

        for name in sorted(common):
            ref_file_path = os.path.join(reference_dir, f"{name}.pdb")
            pred_candidates = _choose_pred_file(run_folder_path, name, mode)

            for pred_path, variant in pred_candidates:
                if os.path.exists(pred_path) and os.path.exists(ref_file_path):
                    tasks.append(EnergyTask(
                        pred_path=pred_path,
                        ref_path=ref_file_path,
                        name=name,
                        run_folder=run_folder,
                        variant=variant,
                    ))

    return tasks


def _summarize_imp(df: pd.DataFrame):
    if "ddG" not in df.columns:
        return
    df_clean = df.dropna(subset=["ddG"])
    if df_clean.empty:
        return

    # 同时按 run_folder 和 variant 分组，避免混在一起误读
    grp = df_clean.groupby(["run_folder", "variant"])
    imp = grp.apply(lambda x: (x["ddG"] < 0).mean() * 100.0).reset_index(name="IMP(%)")

    print("\n--- IMP (%) per Run Folder & Variant ---")
    # 更可读
    for _, row in imp.iterrows():
        print(f"{row['run_folder']}\t{row['variant']}\t{row['IMP(%)']:.2f}")

    overall_imp = (df_clean["ddG"] < 0).mean() * 100.0
    logging.info(f"FINAL_IMP_SCORE (overall, all variants): {overall_imp:.2f}%")


def main(args):
    pdb_dir = os.path.abspath(args.pdb_dir)
    output_dir = os.path.abspath(args.output_dir or args.pdb_dir)  # ✅ 默认 output = input

    os.makedirs(output_dir, exist_ok=True)

    logging.info(f"Scanning in '{pdb_dir}' with idx '{args.name_idx}' (mode={args.mode})")
    tasks = prepare_tasks(pdb_dir, args.name_idx, args.mode)

    if not tasks:
        logging.info("No valid (prediction, reference) file pairs found.")
        return

    logging.info(f"Found {len(tasks)} tasks. Using {args.cpus} CPUs.")

    if args.cpus <= 1:
        # Single process: init here (safe)
        init_opts = (
            f"-use_input_sc -input_ab_scheme AHo_Scheme -ignore_unrecognized_res "
            f"-ignore_zero_occupancy false -load_PDB_components false -no_fconfig -mute all "
            f"-multithreading:total_threads 1 "
            f"-constant_seed -jran {args.seed} "
        )
        try:
            pyrosetta.init(init_opts)
        except RuntimeError:
            pass

        final_results = [calculate_ddG(t) for t in tqdm(tasks, desc="Calculating ddG")]
    else:
        # Multi-process: use spawn + per-worker init (avoid fork issues)
        init_opts = (
            "-use_input_sc -input_ab_scheme AHo_Scheme -ignore_unrecognized_res "
            "-ignore_zero_occupancy false -load_PDB_components false -no_fconfig -mute all "
            "-multithreading:total_threads 1 "
        )

        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=args.cpus,
            initializer=_init_pyrosetta_worker,
            initargs=(init_opts, args.seed),
        ) as pool:
            final_results = list(
                tqdm(pool.imap(calculate_ddG, tasks, chunksize=1),
                     total=len(tasks), desc="Calculating ddG")
            )

    report_data = []
    for t in final_results:
        report_data.append({
            "name": t.name,
            "run_folder": t.run_folder,
            "variant": t.variant,
            "pred_path": t.pred_path,
            "ref_path": t.ref_path,
            **t.scores
        })

    df = pd.DataFrame(report_data)
    _summarize_imp(df)

    out_csv = os.path.join(output_dir, args.csv_name)
    df.to_csv(out_csv, index=False, float_format="%.6f")
    logging.info(f"Saved: {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate interface ddG/IMP for relaxed and/or raw PDBs.")
    parser.add_argument("--pdb_dir", type=str, required=True,
                        help="Root directory (e.g., .../design) containing 'reference' and run subfolders.")
    parser.add_argument("--name_idx", type=str, required=True,
                        help="Path to .idx file with target names (one per line).")

    # ✅ output 默认等于 pdb_dir
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to write csv (default: same as pdb_dir).")
    parser.add_argument("--csv_name", type=str, default="ddG_imp_results.csv",
                        help="CSV filename (default: ddG_imp_results.csv).")

    # ✅ relaxed/raw 选择
    parser.add_argument("--mode", type=str, default="auto",
                        choices=["auto", "relaxed", "raw", "both"],
                        help="auto: prefer *_relaxed.pdb else raw; relaxed/raw: only that; both: compute both if present.")

    parser.add_argument("-c", "--cpus", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42,
                        help="Base seed for deterministic Rosetta packing.")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main(args)
