import logging
import re
import os
from Bio.PDB import PDBParser

from abx.preprocess.numbering import renumber_ab_seq, get_ab_regions

from pyrosetta import *
from pyrosetta.rosetta.core.pack.task import TaskFactory, operation
from pyrosetta.rosetta.core.select import residue_selector as selections
from pyrosetta.rosetta.core.select.movemap import MoveMapFactory, move_map_action
from pyrosetta.rosetta.protocols.relax import FastRelax

# 初始化 PyRosetta
try:
    init('-use_input_sc -input_ab_scheme AHo_Scheme -ignore_unrecognized_res \
        -ignore_zero_occupancy false -load_PDB_components false -relax:default_repeats 2 -no_fconfig -mute all')
except RuntimeError:
    pass

three_to_one = { 'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V' }


class AntibodyRelaxer:
    def __init__(self, generate_area='cdrs'):
        self.generate_area = generate_area
        self.scorefxn = create_score_function('ref2015') 
        self.parser = PDBParser(QUIET=True)
        logging.info(f"AntibodyRelaxer instance created for relaxing '{self.generate_area}' area.")

    def _find_all_indices(self, input_list, value_to_find):
        return [index for index, value in enumerate(input_list) if value == value_to_find]

    def _get_seqres_from_pdb(self, structure, chain_id):
        try:
            model = structure[0]
            chain = model[chain_id]
            seq = "".join([
                three_to_one.get(res.get_resname().strip())
                for res in chain.get_unpacked_list()
                if res.get_resname().strip() in three_to_one
            ])
            return seq
        except KeyError:
            return ""

    def _autodetect_heavy_light(self, structure):
        """
        从 PDB 结构自动识别哪条链像 heavy / light：
        - heavy: renumber_ab_seq(allow=['H']) 能成功
        - light: renumber_ab_seq(allow=['K','L']) 能成功
        """
        model = structure[0]
        chain_ids = [c.id for c in model.get_chains()]

        heavy_cands = []
        light_cands = []

        for cid in chain_ids:
            seq = self._get_seqres_from_pdb(structure, cid)
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

        # 选最长的（更稳一点）
        heavy_id = max(heavy_cands, key=lambda x: x[1])[0] if heavy_cands else None
        light_id = max(light_cands, key=lambda x: x[1])[0] if light_cands else None
        return heavy_id, light_id, chain_ids

    def _parse_chain_ids_from_jobdir(self, pdb_file_path: str):
        """
        diffab 路径：
        .../0001_1n8z_B_A_C_2026_01_13__15_22_41/H_CDR3/0099.pdb
        jobdir = 0001_1n8z_B_A_C_2026_01_13__15_22_41
        解析出 heavy=B, light=A（antigen 可能多 token，不需要）
        """
        # jobdir 是两级父目录：region_dir 的父目录
        jobdir = os.path.basename(os.path.dirname(os.path.dirname(pdb_file_path)))
        parts = jobdir.split('_')
        if len(parts) < 6:
            return None, None

        # 找到 year token（四位数字）
        year_idx = None
        for i, p in enumerate(parts):
            if re.fullmatch(r'\d{4}', p):
                year_idx = i
                break
        if year_idx is None:
            return None, None

        chain_tokens = parts[2:year_idx]  # ['B','A','C'] 或 ['B','A','ANTIGEN',...]
        if len(chain_tokens) < 2:
            return None, None
        return chain_tokens[0], chain_tokens[1]

    def _identify_cdr_regions(self, pdb_file_path):
        def _make_domain(feature, chain_type):
            allow = ['H'] if chain_type == 'H' else ['K', 'L']
            anarci_res = renumber_ab_seq(feature['str_seq'], allow=allow, scheme='imgt')
            domain_numbering = anarci_res.get('domain_numbering')
            if domain_numbering is None:
                return {}
            cdr_def = get_ab_regions(domain_numbering, chain_id=chain_type)
            cdr_indices = {}
            if chain_type == 'H':
                for name, num in [('CDR_H1', 1), ('CDR_H2', 3), ('CDR_H3', 5)]:
                    indices = self._find_all_indices(cdr_def, num)
                    if indices:
                        cdr_indices[name] = [min(indices) + 1, max(indices) + 1]  # 1-based seq position
            else:
                for name, num in [('CDR_L1', 8), ('CDR_L2', 10), ('CDR_L3', 12)]:
                    indices = self._find_all_indices(cdr_def, num)
                    if indices:
                        cdr_indices[name] = [min(indices) + 1, max(indices) + 1]
            return cdr_indices

        filename = os.path.basename(pdb_file_path)
        name_part = filename.split('@')[0] if '@' in filename else os.path.splitext(filename)[0]

        # 先把结构读出来（后面 fallback 需要）
        try:
            structure = self.parser.get_structure('s', pdb_file_path)
        except Exception as e:
            raise ValueError(f'PDB parsing failed for {filename}: {e}')

        heavy_chain_id = None
        light_chain_id = None

        # 1) 旧模式：文件名里有 pdbid_H_L_Ag
        parts = name_part.split('_')
        if len(parts) >= 4:
            heavy_chain_id, light_chain_id = parts[1], parts[2]

        # 2) diffab：从 jobdir 解析
        if not heavy_chain_id or not light_chain_id:
            h2, l2 = self._parse_chain_ids_from_jobdir(pdb_file_path)
            heavy_chain_id = heavy_chain_id or h2
            light_chain_id = light_chain_id or l2

        # 3) 仍然拿不到：结构自动识别
        all_chain_ids = None
        if not heavy_chain_id or not light_chain_id:
            h3, l3, all_chain_ids = self._autodetect_heavy_light(structure)
            heavy_chain_id = heavy_chain_id or h3
            light_chain_id = light_chain_id or l3

        if not heavy_chain_id or not light_chain_id:
            if all_chain_ids is None:
                model = structure[0]
                all_chain_ids = [c.id for c in model.get_chains()]
            raise ValueError(
                f"Could not determine heavy/light chains for {filename}. "
                f"Available chains in PDB: {all_chain_ids}"
            )

        # 真正做 CDR 定位
        all_cdr_indices = {}
        heavy_seq = self._get_seqres_from_pdb(structure, heavy_chain_id)
        light_seq = self._get_seqres_from_pdb(structure, light_chain_id)

        if heavy_seq:
            all_cdr_indices.update(_make_domain({'str_seq': heavy_seq}, 'H'))
        if light_seq:
            all_cdr_indices.update(_make_domain({'str_seq': light_seq}, 'L'))

        logging.info(f"Detected chains for {filename}: heavy={heavy_chain_id}, light={light_chain_id}")
        return all_cdr_indices, heavy_chain_id, light_chain_id


    def run(self, input_path: str, output_path: str):
        logging.info(f"Relaxing: {os.path.basename(input_path)} -> {os.path.basename(output_path)}")
        cdr_dict, heavy_id, light_id = self._identify_cdr_regions(input_path)
        if not cdr_dict:
            raise ValueError(f"Could not determine CDRs for {input_path}.")

        pose = pose_from_pdb(input_path)
        if self.generate_area == 'H3' and 'CDR_H3' in cdr_dict:
            cdr_dict = {'CDR_H3': cdr_dict['CDR_H3']}

        # --- gen_selector 和 nbr_selector 的定义保持不变 ---
        flexible_selectors = []
        for cdr_name, indices in cdr_dict.items():
            chain_id = heavy_id if 'H' in cdr_name else light_id
            start_pose = pose.pdb_info().pdb2pose(chain_id, indices[0])
            end_pose = pose.pdb_info().pdb2pose(chain_id, indices[1])
            if start_pose == 0 or end_pose == 0: continue
            selector = selections.ResidueIndexSelector()
            selector.set_index_range(start_pose, end_pose)
            flexible_selectors.append(selector)

        if not flexible_selectors:
            raise ValueError(f"No valid flexible regions found for {input_path}.")

        if len(flexible_selectors) == 1:
            gen_selector = flexible_selectors[0]
        else:
            gen_selector = selections.OrResidueSelector(flexible_selectors[0], flexible_selectors[1])
            for i in range(2, len(flexible_selectors)):
                gen_selector = selections.OrResidueSelector(gen_selector, flexible_selectors[i])
        
        nbr_selector = selections.NeighborhoodResidueSelector(gen_selector, 8.0, True)

        tf = TaskFactory()
        tf.push_back(operation.InitializeFromCommandline())
        
        # 1. 允许重排 (但会被下一步覆盖)
        tf.push_back(operation.RestrictToRepacking())
        
        # 2. 立即又全局禁止重排
        tf.push_back(operation.PreventRepacking())
        
        # 3. 对 CDR 区域的【外部】，再次施加“禁止重排”
        prevent_repacking_rlt = operation.PreventRepackingRLT()
        prevent_subset_repacking = operation.OperateOnResidueSubset(
            prevent_repacking_rlt, 
            gen_selector,  # 严格遵循 traj_evaluate.py，使用 gen_selector
            flip_subset=True,
        )
        tf.push_back(prevent_subset_repacking)

        # --- MoveMap 的定义保持不变，它允许骨架和侧链在最小化时移动 ---
        mmf = MoveMapFactory()
        mmf.add_bb_action(move_map_action.mm_enable, gen_selector)
        mmf.add_chi_action(move_map_action.mm_enable, nbr_selector) # 即使不能重排，也允许侧链移动
        mm = mmf.create_movemap_from_pose(pose)

        # --- FastRelax 调用保持不变 ---
        fastrelax = FastRelax()
        fastrelax.set_scorefxn(self.scorefxn)
        fastrelax.set_task_factory(tf)
        fastrelax.set_movemap(mm)
        fastrelax.apply(pose)
        
        pose.dump_pdb(output_path)
        logging.info(f"Successfully relaxed and saved: {os.path.basename(output_path)}")