import argparse
import logging
import os
import re
import traceback
from multiprocessing import Pool, cpu_count
from dataclasses import dataclass
from typing import List
import shutil
from tqdm import tqdm

from abx.relax import AntibodyRelaxer


@dataclass
class RelaxTask:
    in_path: str                 # 原始输入路径
    current_path: str            # 流水线中当前步骤的输入路径
    output_path: str             # 最终弛豫产物的期望路径
    name: str                    # PDB基础名
    status: str = 'created'
    generate_area: str = 'cdrs'

    def can_proceed(self) -> bool:
        return self.status != 'failed'

    def mark_success(self, final_path: str):
        self.status = 'success'
        self.current_path = final_path

    def mark_failure(self):
        self.status = 'failed'


# ----------------------------
# 旧模式：递归扫 input_dir 下所有 pdb（保持你现有逻辑不变）
# ----------------------------
def prepare_relax_tasks_legacy(input_dir: str, output_dir: str, generate_area: str) -> List[RelaxTask]:
    tasks = []
    pdb_pattern = re.compile(r'\.pdb$')
    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)

    for root, _, files in os.walk(input_dir):
        if 'reference' in root.split(os.sep) or 'relaxed' in root.split(os.sep):
            continue  # 跳过参考和已弛豫的目录

        for file in files:
            if (not pdb_pattern.search(file)) or ('_relaxed' in file):
                continue

            in_path = os.path.join(root, file)

            # 构造输出路径（保持相对路径结构）
            relative_path = os.path.relpath(in_path, input_dir)
            output_path = os.path.join(output_dir, relative_path).replace('.pdb', '_relaxed.pdb')

            if os.path.exists(output_path):
                continue

            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            tasks.append(RelaxTask(
                in_path=in_path,
                current_path=in_path,
                output_path=output_path,
                name=os.path.splitext(file)[0],
                generate_area=generate_area
            ))
    return tasks


# ----------------------------
# diffab 模式：结构化扫描（只认 job_dir / region_dir / 0000-0099.pdb，并拷贝 REF*.pdb）
# ----------------------------
def prepare_relax_tasks_diffab(input_dir: str, output_dir: str, generate_area: str) -> List[RelaxTask]:
    """
    期望结构（示例）：
    input_dir/
      0000_3rkd_D_C_B_2026_01_13__15_19_52/
        H_CDR3/           或 H_CDR3-O2
          0000.pdb ... 0099.pdb
          REF1.pdb
    """
    tasks = []
    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)

    # job 目录：0000_..._YYYY_MM_DD__HH_MM_SS
    job_dir_re = re.compile(r'^\d{4}_.+_\d{4}_\d{2}_\d{2}__\d{2}_\d{2}_\d{2}$')
    # 区域目录：H_CDR3 或 H_CDR3-O2（更一般：H/L_CDR{1,2,3}(-O\d+)?）
    region_dir_re = re.compile(r'^[HL]_CDR[123](?:-O\d+)?$')
    # 采样文件：0000.pdb ~ 0099.pdb（只认 4 位数字命名）
    sample_pdb_re = re.compile(r'^\d{4}\.pdb$')
    # 参考结构：REF1.pdb / REF2.pdb ...
    ref_pdb_re = re.compile(r'^REF\d+\.pdb$', re.IGNORECASE)

    # 只扫描 input_dir 的一级子目录作为 job_dir
    try:
        entries = sorted(os.listdir(input_dir))
    except FileNotFoundError:
        raise FileNotFoundError(f"input_dir not found: {input_dir}")

    for job_name in entries:
        job_path = os.path.join(input_dir, job_name)
        if not os.path.isdir(job_path):
            continue
        if not job_dir_re.match(job_name):
            # 不匹配 diffab job 目录命名就跳过（避免扫到别的垃圾目录）
            continue

        # 扫描 job_dir 下的 region 目录（一级）
        for region_name in sorted(os.listdir(job_path)):
            region_path = os.path.join(job_path, region_name)
            if not os.path.isdir(region_path):
                continue
            if not region_dir_re.match(region_name):
                continue

            # 先把 REF*.pdb 复制到输出目录（不 relax）
            for fn in sorted(os.listdir(region_path)):
                if ref_pdb_re.match(fn):
                    src = os.path.join(region_path, fn)
                    rel = os.path.relpath(src, input_dir)
                    dst = os.path.join(output_dir, rel)
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    if not os.path.exists(dst):
                        shutil.copy2(src, dst)

            # 再收集 0000-0099.pdb relax 任务
            for fn in sorted(os.listdir(region_path)):
                if not sample_pdb_re.match(fn):
                    continue
                in_path = os.path.join(region_path, fn)

                relative_path = os.path.relpath(in_path, input_dir)
                output_path = os.path.join(output_dir, relative_path).replace('.pdb', '_relaxed.pdb')

                if os.path.exists(output_path):
                    continue

                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                tasks.append(RelaxTask(
                    in_path=in_path,
                    current_path=in_path,
                    output_path=output_path,
                    name=os.path.splitext(fn)[0],
                    generate_area=generate_area
                ))

    return tasks


# --- 工作者函数 ---
def worker_task(task: RelaxTask) -> RelaxTask:
    if not task.can_proceed():
        return task

    try:
        relaxer = AntibodyRelaxer(generate_area=task.generate_area)
        relaxer.run(task.in_path, task.output_path)
        task.mark_success(task.output_path)
    except Exception as e:
        logging.error(f"Worker failed on {task.name}: {e}\n{traceback.format_exc()}")
        task.mark_failure()

    return task


def main():
    parser = argparse.ArgumentParser(description="A robust relaxation pipeline (legacy + diffab mode).")
    parser.add_argument('-i', '--input_dir', type=str, required=True,
                        help="Root directory containing PDB files to be relaxed.")
    parser.add_argument('-c', '--cpus', type=int, default=max(1, cpu_count() - 2),
                        help="Number of CPU cores for parallel processing.")
    parser.add_argument('-g', '--generate_area', type=str, choices=['cdrs', 'H3'], default='cdrs',
                        help="Region to apply flexibility (passed to AntibodyRelaxer).")

    # ✅ 新增：模式选择（默认 legacy，不显式指定就按旧行为跑）
    parser.add_argument('--mode', type=str, choices=['legacy', 'diffab'], default='legacy',
                        help="Task discovery mode. legacy=recursive scan; diffab=DiffAb-style structured scan.")

    args = parser.parse_args()

    # 自动输出目录：input_dir + "_relaxed"（避免重复加）
    in_dir = os.path.abspath(args.input_dir.rstrip('/'))
    out_dir = in_dir if in_dir.endswith('_relaxed') else (in_dir + '_relaxed')
    args.input_dir = in_dir
    args.output_dir = out_dir
    os.makedirs(args.output_dir, exist_ok=True)

    # legacy：复制 reference 文件夹
    # diffab：REF*.pdb 是分散在 region 目录里，已在 diffab 扫描函数中复制
    if args.mode == 'legacy':
        src_ref = os.path.join(args.input_dir, "reference")
        dst_ref = os.path.join(args.output_dir, "reference")
        if os.path.isdir(src_ref) and not os.path.exists(dst_ref):
            shutil.copytree(src_ref, dst_ref)

    log_file = os.path.join(args.output_dir, 'relax_pipeline.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(processName)s] %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )

    logging.info(f"[mode={args.mode}] Scanning for relaxation tasks in '{args.input_dir}'...")

    if args.mode == 'legacy':
        tasks = prepare_relax_tasks_legacy(args.input_dir, args.output_dir, args.generate_area)
    else:
        tasks = prepare_relax_tasks_diffab(args.input_dir, args.output_dir, args.generate_area)

    if not tasks:
        logging.info("No new PDB files found to process.")
        return

    logging.info(f"Found {len(tasks)} tasks. Starting relaxation with {args.cpus} CPUs...")

    with Pool(processes=args.cpus) as pool:
        results = list(tqdm(pool.imap(worker_task, tasks),
                            total=len(tasks),
                            desc="Relaxing structures"))

    success_count = sum(1 for r in results if r.status == 'success')
    logging.info(f"Relaxation complete. Total tasks: {len(tasks)}. Successful: {success_count}.")


if __name__ == '__main__':
    main()
