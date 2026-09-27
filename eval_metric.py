import os
import argparse
import functools
import multiprocessing as mp
import logging
import re
import pandas as pd
import traceback
from tqdm import tqdm

from abx.metric import eval_metric as _eval_metric


def parse_list(data_dir: str, include_relaxed: bool) -> list:
    pdb_files = []
    pdb_pattern = re.compile(r'\.pdb$')
    relax_pattern = re.compile(r'_relaxed\.pdb$')

    reference_dir = os.path.abspath(os.path.join(data_dir, 'reference'))

    for root, _, files in os.walk(data_dir):
        if os.path.abspath(root).startswith(reference_dir):
            continue

        for fname in files:
            if not pdb_pattern.search(fname):
                continue
            if os.path.getsize(os.path.join(root, fname)) == 0:
                continue

            is_relaxed = bool(relax_pattern.search(fname))
            if include_relaxed:
                if not is_relaxed:
                    continue
            else:
                if is_relaxed:
                    continue

            pdb_files.append(os.path.join(root, fname))

    return pdb_files


# ---------- diffab: 扫描 + 配对 ----------
def _pick_best_regions(region_names):
    """
    同一 base（如 H_CDR3）只保留最大 O-step（H_CDR3-O2 优先于 H_CDR3）
    避免你遇到的 100x2=200 重复任务。
    """
    best = {}
    for rn in region_names:
        if "-O" in rn:
            base, step_str = rn.split("-O", 1)
            try:
                step = int(step_str)
            except ValueError:
                step = 0
        else:
            base, step = rn, 0
        cur = best.get(base)
        if cur is None or step > cur[0]:
            best[base] = (step, rn)
    return [v[1] for v in best.values()]


def build_tasks_legacy(data_dir: str, include_relaxed: bool):
    pred_files = parse_list(data_dir, include_relaxed=include_relaxed)
    reference_dir = os.path.join(data_dir, 'reference')

    tasks = []
    for pred_file in pred_files:
        stem = os.path.splitext(os.path.basename(pred_file))[0].split('@')[0]
        if stem.endswith('_relaxed'):
            stem = stem[:-len('_relaxed')]
        ref_file = os.path.join(reference_dir, f"{stem}.pdb")
        if os.path.exists(ref_file):
            tasks.append((pred_file, ref_file, {}))
        else:
            logging.warning(f"No reference file found for prediction: {os.path.basename(pred_file)}")
    return tasks


def build_tasks_diffab(data_dir: str, include_relaxed: bool):
    """
    diffab 输入结构：
      data_dir/
        0001_1n8z_B_A_C_2026_01_13__15_22_41/
          H_CDR3/ or H_CDR3-O2/
            0000.pdb ... 0099.pdb
            REF1.pdb
    """
    tasks = []
    data_dir = os.path.abspath(data_dir)

    job_dir_re = re.compile(
    r'^(?P<idx>\d{4})_'
    r'(?P<pdbid>[^_]+)_'
    r'(?P<chains>.+?)_'
    r'(?P<year>\d{4})_(?P<month>\d{2})_(?P<day>\d{2})__'
    r'(?P<h>\d{2})_(?P<m>\d{2})_(?P<s>\d{2})$'
)
    region_dir_re = re.compile(r'^[HL]_CDR[123](?:-O\d+)?$')

    if include_relaxed:
        sample_re = re.compile(r'^\d{4}_relaxed\.pdb$')
    else:
        sample_re = re.compile(r'^\d{4}\.pdb$')

    ref_re = re.compile(r'^REF\d+\.pdb$', re.IGNORECASE)

    def parse_job(job_name: str):
        m = job_dir_re.match(job_name)
        if not m:
            return None

        target = m.group("pdbid")

        # chains 段例如 "D_C_B"（也可能更长），过滤掉空 token
        chain_tokens = [t for t in m.group("chains").split("_") if t]
        if len(chain_tokens) < 2:
            return None

        heavy = chain_tokens[0]
        light = chain_tokens[1]
        antigen_tokens = chain_tokens[2:]
        antigen = "".join(antigen_tokens) if antigen_tokens else ""

        return {
            "target": target,
            "heavy": heavy,
            "light": light,
            "antigen": antigen,
            "pdbname": f"{target}_{heavy}_{light}_{antigen}" if antigen else f"{target}_{heavy}_{light}_"
        }

    for job_name in sorted(os.listdir(data_dir)):
        job_path = os.path.join(data_dir, job_name)
        if not os.path.isdir(job_path) or not job_dir_re.match(job_name):
            continue

        meta_job = parse_job(job_name)
        if meta_job is None:
            logging.warning(f"Skip job (cannot parse): {job_name}")
            continue

        region_names = []
        for rn in sorted(os.listdir(job_path)):
            rp = os.path.join(job_path, rn)
            if os.path.isdir(rp) and region_dir_re.match(rn):
                region_names.append(rn)

        region_names = _pick_best_regions(region_names)

        for region_name in region_names:
            region_path = os.path.join(job_path, region_name)

            # 找参考 REF*.pdb（优先 REF1）
            ref_file = None
            ref_candidates = [f for f in sorted(os.listdir(region_path)) if ref_re.match(f)]
            if not ref_candidates:
                logging.warning(f"No REF*.pdb under {region_path}, skip.")
                continue
            if "REF1.pdb" in ref_candidates:
                ref_file = os.path.join(region_path, "REF1.pdb")
            else:
                ref_file = os.path.join(region_path, ref_candidates[0])

            # 扫样本
            for fn in sorted(os.listdir(region_path)):
                if not sample_re.match(fn):
                    continue
                pred_file = os.path.join(region_path, fn)

                sample_id = fn.split('_')[0].split('.')[0]  # 0000 or 0000_relaxed -> 0000
                meta = {
                    "mode": "diffab",
                    "job": job_name,
                    "region": region_name,
                    "sample_id": sample_id,
                    "target": meta_job["target"],
                    "heavy": meta_job["heavy"],
                    "light": meta_job["light"],
                    "antigen": meta_job["antigen"],
                }
                tasks.append((pred_file, ref_file, meta))

    return tasks


def eval_metric_with_meta(pred_file: str, ref_file: str, meta: dict, args: argparse.Namespace):
    r = _eval_metric(pred_file, ref_file, args)
    if r is None:
        return None
    if meta:
        r.update(meta)
    return r


def main(args):
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s [%(processName)s] %(levelname)s - %(message)s'
    )

    try:
        logging.info(f"[mode={args.mode}] Scanning for evaluation tasks in '{args.data_dir}'...")

        include_relaxed = args.data_dir.rstrip('/').endswith('_relaxed')

        if args.mode == "legacy":
            tasks = build_tasks_legacy(args.data_dir, include_relaxed=include_relaxed)
        else:
            tasks = build_tasks_diffab(args.data_dir, include_relaxed=include_relaxed)

        if not tasks:
            logging.info("No valid prediction-reference pairs found to process.")
            return

        logging.info(f"Found {len(tasks)} tasks. Starting evaluation with {args.cpus} CPUs...")

        func = functools.partial(eval_metric_with_meta, args=args)

        with mp.Pool(processes=args.cpus) as pool:
            all_results = list(tqdm(pool.starmap(func, tasks), total=len(tasks)))

        results = [r for r in all_results if r is not None]
        if not results:
            logging.warning("No files were processed successfully. Please check logs for errors.")
            return

        logging.info("Evaluation complete. Aggregating results...")
        df = pd.DataFrame(results)

        avg_metrics = [col for col in df.columns if 'RMSD' in col or 'AAR' in col]
        if avg_metrics:
            print("\n" + "-" * 21)
            print("Average Results for each Metric")
            print("-" * 21)
            print(df[avg_metrics].mean().to_string())

        output_path = os.path.join(args.data_dir, 'evaluation_results.csv')
        df.to_csv(output_path, index=False, float_format='%.4f')
        logging.info(f"Full results saved to {output_path}")

    except Exception as e:
        logging.error(f"A critical error occurred in the main pipeline: {e}")
        traceback.print_exc()


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--data_dir', type=str, required=True)
    parser.add_argument('-c', '--cpus', type=int, default=1)
    parser.add_argument('-e', '--energy', action='store_true', help="Enable energy calculation.")
    parser.add_argument('-v', '--verbose', action='store_true', help="Enable verbose logging.")
    parser.add_argument('--mode', type=str, choices=['legacy', 'diffab'], default='legacy',
                        help="legacy: old folder+reference; diffab: job/region/sample+REF*.pdb")
    args = parser.parse_args()

    if not any(isinstance(h, logging.StreamHandler) for h in logging.getLogger().handlers):
        logging.getLogger().addHandler(logging.StreamHandler())

    main(args)
