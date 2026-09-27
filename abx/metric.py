import os
import argparse
import traceback
from collections import OrderedDict

import numpy as np
import pandas as pd

from abx.common.ab_utils import calc_ab_metrics
from abx.common import residue_constants

from Bio.SeqUtils import seq1
from Bio.PDB.PDBParser import PDBParser
import logging
from abx.preprocess.numbering import renumber_ab_seq, get_ab_regions

from Bio import pairwise2
from Bio.Align import substitution_matrices


# --- Lazy PyRosetta init: only when energy is needed ---
_PYROSETTA_READY = False

import re

def _chain_seq_from_model(model, chain_id: str) -> str:
    from Bio.SeqUtils import seq1
    try:
        residues = list(model[chain_id].get_residues())
    except KeyError:
        return ""
    aa = []
    for r in residues:
        try:
            aa.append(seq1(r.get_resname()))
        except Exception:
            continue
    return "".join(aa)

def _autodetect_heavy_light(model):
    chain_ids = [c.id for c in model.get_chains()]
    heavy_cands, light_cands = [], []
    for cid in chain_ids:
        seq = _chain_seq_from_model(model, cid)
        if not seq:
            continue
        try:
            h = renumber_ab_seq(seq, allow=['H'], scheme='imgt')
            if h.get('domain_numbering') is not None:
                heavy_cands.append((cid, len(seq)))
        except Exception:
            pass
        try:
            l = renumber_ab_seq(seq, allow=['K', 'L'], scheme='imgt')
            if l.get('domain_numbering') is not None:
                light_cands.append((cid, len(seq)))
        except Exception:
            pass

    heavy = max(heavy_cands, key=lambda x: x[1])[0] if heavy_cands else None
    light = max(light_cands, key=lambda x: x[1])[0] if light_cands else None
    return heavy, light, chain_ids

def _infer_hl_from_name_or_path(pdb_file: str, model=None):
    """
    返回: (heavy_id, light_id)
    1) 旧模式：<code>_<H>_<L>_<Ag>.pdb
    2) diffab：.../<jobdir>/<region>/<0000.pdb>
       jobdir: 0001_1n8z_B_A_C_2026_...  -> heavy=B, light=A
    3) 结构自动识别（需要 model）
    """
    base = os.path.splitext(os.path.basename(pdb_file))[0].split('@')[0]
    if base.endswith('_relaxed'):
        base = base[:-len('_relaxed')]
    parts = base.split('_')
    if len(parts) >= 3:
        return parts[1], parts[2]

    # diffab: 从 jobdir 解析（两级父目录）
    jobdir = os.path.basename(os.path.dirname(os.path.dirname(pdb_file)))
    jparts = jobdir.split('_')
    year_idx = None
    for i, p in enumerate(jparts):
        if re.fullmatch(r'\d{4}', p):
            year_idx = i
            break
    if year_idx is not None and year_idx >= 4:
        chain_tokens = jparts[2:year_idx]
        if len(chain_tokens) >= 2:
            return chain_tokens[0], chain_tokens[1]

    # 最后：结构识别
    if model is not None:
        h, l, _ = _autodetect_heavy_light(model)
        return h, l

    return None, None


def _ensure_pyrosetta_ready(mute_banner: bool = True):
    """
    在当前进程内确保 PyRosetta 只初始化一次。
    spawn 多进程下每个子进程都会跑到这里一次，但不会重复 init。
    """
    global _PYROSETTA_READY
    if _PYROSETTA_READY:
        return

    if mute_banner:
        import contextlib, io
        buf_out, buf_err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
            import pyrosetta
            pyrosetta.init(
                "-use_input_sc -input_ab_scheme AHo_Scheme -ignore_unrecognized_res "
                "-ignore_zero_occupancy false -load_PDB_components true "
                "-relax:default_repeats 2 -no_fconfig -mute all"
            )
    else:
        import pyrosetta
        pyrosetta.init(
            "-use_input_sc -input_ab_scheme AHo_Scheme -ignore_unrecognized_res "
            "-ignore_zero_occupancy false -load_PDB_components true "
            "-relax:default_repeats 2 -no_fconfig"
        )

    _PYROSETTA_READY = True
    
def pyrosetta_interface_energy(pdb_path, interface):
    _ensure_pyrosetta_ready(mute_banner=True)
    import pyrosetta
    from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
    from pyrosetta import create_score_function

    pose = pyrosetta.pose_from_pdb(pdb_path)
    mover = InterfaceAnalyzerMover()
    mover.set_interface(interface)
    mover.set_scorefunction(create_score_function('ref2015'))
    mover.apply(pose)
    return pose.scores['dG_separated']


def InterfaceEnergy(pdb_file):
    try:
        parser = PDBParser(QUIET=True)
        model = parser.get_structure("s", pdb_file)[0]

        heavy_chain_id, light_chain_id = _infer_hl_from_name_or_path(pdb_file, model=model)
        if not heavy_chain_id or not light_chain_id:
            raise ValueError(f"Cannot infer heavy/light for {os.path.basename(pdb_file)}")

        all_chain_ids = [c.id for c in model.get_chains()]
        antigen_chain_ids = [c for c in all_chain_ids if c not in {heavy_chain_id, light_chain_id}]
        if not antigen_chain_ids:
            return np.nan

        antibody_chains_str = f"{heavy_chain_id}{light_chain_id}"
        antigen_str = "".join(antigen_chain_ids)
        interface = f"{antibody_chains_str}_{antigen_str}"

        dG = pyrosetta_interface_energy(pdb_file, interface)
        return dG
    except Exception as e:
        logging.error(f"Failed to calculate InterfaceEnergy for {pdb_file}: {e}")
        return np.nan



def get_aligned_ab_data(pdb_file: str) -> dict:
    parser = PDBParser(QUIET=1)
    model = parser.get_structure('s', pdb_file)[0]
    base_name = os.path.splitext(os.path.basename(pdb_file))[0].split('@')[0]
    if base_name.endswith('_relaxed'):
        base_name = base_name[:-len('_relaxed')]

    heavy_id, light_id = _infer_hl_from_name_or_path(pdb_file, model=model)
    if not heavy_id or not light_id:
        raise ValueError(f"Cannot infer heavy/light chains for {os.path.basename(pdb_file)}")


    # _, heavy_id, light_id, _ = base_name.split('_')

    heavy_residues_full = list(model[heavy_id].get_residues())
    light_residues_full = list(model[light_id].get_residues())
    
    heavy_str_seq_full = "".join([seq1(r.get_resname()) for r in heavy_residues_full])
    light_str_seq_full = "".join([seq1(r.get_resname()) for r in light_residues_full])
    
    heavy_anarci_res = renumber_ab_seq(heavy_str_seq_full, allow=['H'], scheme='imgt')
    light_anarci_res = renumber_ab_seq(light_str_seq_full, allow=['K', 'L'], scheme='imgt')
    
    h_domain_num_raw = heavy_anarci_res.get('domain_numbering')
    l_domain_num_raw = light_anarci_res.get('domain_numbering')

    if h_domain_num_raw is None or l_domain_num_raw is None:
        raise ValueError(f"ANARCI failed to process chains in {os.path.basename(pdb_file)}")

    h_valid_mask = [num is not None for num in h_domain_num_raw]
    l_valid_mask = [num is not None for num in l_domain_num_raw]

    heavy_residues_aligned = [res for res, is_valid in zip(heavy_residues_full, h_valid_mask) if is_valid]
    light_residues_aligned = [res for res, is_valid in zip(light_residues_full, l_valid_mask) if is_valid]
    
    all_residues_aligned = heavy_residues_aligned + light_residues_aligned
    
    h_domain_num_aligned = [num for num in h_domain_num_raw if num is not None]
    l_domain_num_aligned = [num for num in l_domain_num_raw if num is not None]
    
    aligned_str_seq = "".join([seq1(r.get_resname()) for r in all_residues_aligned])

    aligned_coords = np.zeros((len(all_residues_aligned), 3))
    for i, r in enumerate(all_residues_aligned):
        aligned_coords[i] = r['CA'].get_coord()

    heavy_cdr_def_aligned = get_ab_regions(h_domain_num_aligned, chain_id='H')
    light_cdr_def_aligned = get_ab_regions(l_domain_num_aligned, chain_id='L')
    aligned_cdr_def = np.concatenate([heavy_cdr_def_aligned, light_cdr_def_aligned], axis=0)

    if not (aligned_coords.shape[0] == len(aligned_str_seq) == aligned_cdr_def.shape[0]):
        raise AssertionError(
            f"Internal logic error in get_aligned_ab_data for {os.path.basename(pdb_file)}: "
            f"Coords({aligned_coords.shape[0]}), Seq({len(aligned_str_seq)}), CDRDef({aligned_cdr_def.shape[0]})"
        )

    return {
        'coords': aligned_coords,
        'str_seq': aligned_str_seq,
        'cdr_def': aligned_cdr_def
    }


def eval_metric(pred_file: str, ref_file: str, args: argparse.Namespace):
    """
    评估单个预测文件，对比其参考文件。
    该版本能够处理预测和参考之间真实的序列长度差异 (Indels)。
    """
    pdb_name = os.path.splitext(os.path.basename(pred_file))[0].split('@')[0]
    base_result = {'code': pdb_name, 'file_path': pred_file}

    try:
        # 1. 对预测文件和参考文件都使用统一的函数获取对齐后的Fv区数据
        pred_data = get_aligned_ab_data(pred_file)
        ref_data = get_aligned_ab_data(ref_file) # 按需加载参考数据

        gt_ab_ca = ref_data['coords']
        gt_ab_str_seq = ref_data['str_seq']
        cdr_def_ref = ref_data['cdr_def']
        
        pred_ab_ca = pred_data['coords']
        pred_ab_str_seq = pred_data['str_seq']
        
        # 处理Fv区之间的长度不一致 (Indels)
        if len(gt_ab_str_seq) != len(pred_ab_str_seq):
            logging.warning(f"Fv domain length mismatch for {pdb_name}: "
                            f"Ref({len(gt_ab_str_seq)}) vs Pred({len(pred_ab_str_seq)}). Performing alignment.")
            
            alignments = pairwise2.align.globalds(
                gt_ab_str_seq, pred_ab_str_seq, 
                substitution_matrices.load("BLOSUM62"), -10, -0.5
            )
            if not alignments:
                raise ValueError("Pairwise alignment failed.")
            
            best_aln = alignments[0]
            aligned_gt_seq, aligned_pred_seq, _, _, _ = best_aln
            
            aligned_gt_indices = []
            aligned_pred_indices = []
            gt_idx, pred_idx = 0, 0
            for gt_char, pred_char in zip(aligned_gt_seq, aligned_pred_seq):
                if gt_char != '-' and pred_char != '-':
                    aligned_gt_indices.append(gt_idx)
                    aligned_pred_indices.append(pred_idx)
                if gt_char != '-':
                    gt_idx += 1
                if pred_char != '-':
                    pred_idx += 1

            if not aligned_gt_indices:
                raise ValueError("Alignment resulted in no common residues.")

            gt_ab_ca_aligned = gt_ab_ca[aligned_gt_indices]
            pred_ab_ca_aligned = pred_ab_ca[aligned_pred_indices]
            cdr_def_aligned = cdr_def_ref[aligned_gt_indices]
            gt_ab_str_seq_aligned = "".join(np.array(list(gt_ab_str_seq))[aligned_gt_indices])
            pred_ab_str_seq_aligned = "".join(np.array(list(pred_ab_str_seq))[aligned_pred_indices])
        else:
            gt_ab_ca_aligned, pred_ab_ca_aligned = gt_ab_ca, pred_ab_ca
            cdr_def_aligned = cdr_def_ref
            gt_ab_str_seq_aligned, pred_ab_str_seq_aligned = gt_ab_str_seq, pred_ab_str_seq

        # 3. 在对齐后的数据上，安全地进行计算
        ab_metrics = calc_ab_metrics(
            gt_ab_ca_aligned, 
            pred_ab_ca_aligned, 
            cdr_def_aligned, 
            gt_ab_str_seq_aligned, 
            pred_ab_str_seq_aligned
        )
        base_result.update(ab_metrics)
        
        # 4. 能量计算部分保持不变 (在完整的、未裁剪的结构上计算)
        #import ipdb; ipdb.set_trace()
        if args.energy:
            pred_dG = InterfaceEnergy(pred_file)
            ref_dG = InterfaceEnergy(ref_file)
            base_result.update({
                'dG_gen': pred_dG,
                'dG_ref': ref_dG,
                'ddG': pred_dG - ref_dG if not (np.isnan(pred_dG) or np.isnan(ref_dG)) else np.nan
            })
            
        return base_result

    except Exception as e:
        logging.error(f"An unexpected error occurred while processing {os.path.basename(pred_file)}. "
                      f"Skipping. Error: {traceback.format_exc()}")
        return None