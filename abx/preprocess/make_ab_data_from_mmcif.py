import os
import argparse
import functools
import multiprocessing as mp
import logging
import json
import pandas as pd
import traceback
import numpy as np

from Bio.PDB.PDBExceptions import PDBConstructionException
from Bio.PDB.PDBParser import PDBParser
from Bio.SeqUtils import seq1

from abx.common import residue_constants
from abx.data.mmcif_parsing import parse as mmcif_parse
from abx.preprocess.numbering import renumber_ab_seq, get_ab_regions



# 接触距离阈值：dyMEAN config.CONTACT_DIST = 6.6
CONTACT_DIST = 6.6  # Å

# NEW: 是否强制要求 CDR-H3 与抗原有接触（默认 True，保持现在行为）
REQUIRE_H3_CONTACT = True


logger = logging.getLogger(__name__)

# ======================
# RAbD / dyMEAN 测试集中 3 个“错标记”的 PDB 特殊处理
# 我们不改真实结构里用到的链，只是把 npz 文件名改成 dyMEAN 的格式：
#   3h3b -> 3h3b_D_E_B
#   2ghw -> 2ghw_B_E_A
#   3uzq -> 3uzq_A_C_B
# 匹配用 (pdb, 抗原链字符串)，因为这几个 PDB 里抗原链刚好不一样。
RABD_TEST_NAME_MAP = {
    ("3h3b", "B"): ("D", "E", "B"),   # 3h3b_*_*_B  -> 3h3b_D_E_B
    ("2ghw", "A"): ("B", "E", "A"),   # 2ghw_*_*_A  -> 2ghw_B_E_A
    ("3uzq", "B"): ("A", "C", "B"),   # 3uzq_*_*_B  -> 3uzq_A_C_B
}

# ======================
# 1. 全局配置（尽量和 dyMEAN / DiffAb 对齐）
# ======================

# DiffAb / dyMEAN 对 SAbDab 的抗原类型要求：只要 protein / peptide
AG_TYPES = ["protein", "peptide"]

# 分辨率过滤：DiffAb 使用 4.0 Å
RESOLUTION_THRESHOLD = 4.0

# 接触距离阈值：dyMEAN config.CONTACT_DIST = 6.6
CONTACT_DIST = 6.6  # Å

# NEW: 全局编号方案（通过命令行设置）
NUMBERING_SCHEME = "imgt"  # "imgt" or "chothia"

# NEW: IMGT / Chothia 的保守位点（用 one-letter）
#    来自 dyMEAN configs.IMGT / Chothia 里的 Hconserve/Lconserve
#    这里只对这些位置做“如果出现，就必须匹配指定氨基酸”的检查。
CONSERVE_RULES = {
    "imgt": {
        "H": {
            23: ["C"],  # CYS
            41: ["W"],  # TRP
            104: ["C"],  # CYS
        },
        "L": {
            23: ["C"],
            41: ["W"],
            104: ["C"],
        },
    },
    "chothia": {
        "H": {
            92: ["C"],  # CYS
        },
        "L": {
            88: ["C"],  # CYS
        },
    },
}


def _parse_resolution(val):
    """把 SAbDab 的 resolution 字段转成 float，解析不了就返回 None。"""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None

    s = str(val).strip()
    if s.upper() == "NOT" or s == "":
        return None

    # 处理类似 "2.50, 2.80" 的情况，取第一个
    if "," in s:
        s = s.split(",")[0].strip()

    try:
        return float(s)
    except Exception:
        return None


def parse_list(path):
    """
    从 sabdab_summary*.tsv 读取，并做 “DiffAb + dyMEAN 风格” 的过滤，
    然后生成给 process/code, chain_ids/ 用的列表：

    返回: list[(code, chain_ids)]
      - code: pdb id (小写)
      - chain_ids: list[ (Hchain, Lchain, cleaned_antigen_chain_string) ]
    """
    df = pd.read_csv(path, sep="\t")
    logger.info(f"[parse_list] raw entries from summary: {df.shape[0]}")

    # 1) method 过滤：我们保守地只保留 X-RAY / EM
    if "method" in df.columns:
        df = df[df["method"].isin(["X-RAY DIFFRACTION", "ELECTRON MICROSCOPY"])]
        logger.info(f"[parse_list] after method filter: {df.shape[0]}")
    else:
        logger.warning("[parse_list] no 'method' column, skip method filter")

    # 2) model == 0
    if "model" in df.columns:
        df = df[df["model"] == 0]
        logger.info(f"[parse_list] after model == 0 filter: {df.shape[0]}")
    else:
        logger.warning("[parse_list] no 'model' column, skip model filter")

    # 3) 必须同时有 Hchain 和 Lchain（排除 nanobody）
    df = df.fillna({"Hchain": "", "Lchain": ""})
    df = df[(df["Hchain"] != "") & (df["Lchain"] != "")]
    logger.info(f"[parse_list] after requiring Hchain & Lchain: {df.shape[0]}")

    # 4) 必须有 antigen_chain
    if "antigen_chain" in df.columns:
        df["antigen_chain"] = df["antigen_chain"].fillna("")
        df = df[df["antigen_chain"] != ""]
        logger.info(f"[parse_list] after requiring antigen_chain: {df.shape[0]}")
    else:
        logger.warning("[parse_list] no 'antigen_chain' column, return empty.")
        return []

    # 5) 分辨率过滤: <= 4.0 Å
    if "resolution" in df.columns:
        df["resolution_val"] = df["resolution"].apply(_parse_resolution)
        before = df.shape[0]
        df = df[df["resolution_val"].notna()]
        df = df[df["resolution_val"] <= RESOLUTION_THRESHOLD]
        logger.info(
            f"[parse_list] after resolution <= {RESOLUTION_THRESHOLD}Å: "
            f"{df.shape[0]} (from {before})"
        )
    else:
        logger.warning("[parse_list] no 'resolution' column, skip resolution filter")

    # 6) antigen_type + antigen_chain 逐链过滤
    if "antigen_type" not in df.columns:
        logger.warning(
            "[parse_list] no 'antigen_type' column, "
            "cannot apply dyMEAN-style AG_TYPES filter exactly."
        )
        df["antigen_type"] = ""
    else:
        df["antigen_type"] = df["antigen_type"].fillna("")

    pairs = []
    dropped_all_ag = 0

    if "pdb" not in df.columns:
        raise ValueError("summary file must contain 'pdb' column")

    for _, row in df.iterrows():
        code = str(row["pdb"]).strip().lower()
        heavy = str(row["Hchain"]).strip()
        light = str(row["Lchain"]).strip()
        ag_raw = str(row["antigen_chain"]).strip()
        ag_type_raw = str(row["antigen_type"]).strip()

        if ag_raw == "":
            continue

        # sabdab 里常见格式: 'A|B|C' / 'A' 等
        ag_chains = [s.strip() for s in ag_raw.split("|")]
        ag_types = [s.strip() for s in ag_type_raw.split("|")] if ag_type_raw else []

        # 若 antigen_type 信息缺失或长度不匹配，保守做法：直接跳过
        if not ag_types or len(ag_types) != len(ag_chains):
            logger.warning(
                f"[parse_list] {code}: len(antigen_type)={len(ag_types)} "
                f"!= len(antigen_chain)={len(ag_chains)}, skip."
            )
            continue

        # cleaned_chains = []
        # for t, c in zip(ag_types, ag_chains):
        #     # 只保留 antigen_type 在 AG_TYPES 中的那些链
        #     if t not in AG_TYPES:
        #         continue
        #     # dyMEAN: 不允许 antigen_chain 和 H/L 链重合
        #     if c == heavy or c == light:
        #         continue
        #     cleaned_chains.append(c)

        # if not cleaned_chains:
        #     dropped_all_ag += 1
        #     continue
                # 2) 统一大小写：H、L、抗原链全部转大写
        heavy = heavy.upper()
        light = light.upper()

        cleaned_chains = []
        for t, c in zip(ag_types, ag_chains):
            c = c.upper()   # ★ 这里把 antigen 链也统一大写

            if t not in AG_TYPES:
                continue
            if c == heavy or c == light:
                continue
            cleaned_chains.append(c)

        if not cleaned_chains:
            print(f"[DROP_ALL_AG] pdb={code}, H={heavy}, L={light}, raw_ag={ag_raw}, raw_type={ag_type_raw}")
            dropped_all_ag += 1
            continue


        cleaned_ag = "|".join(cleaned_chains)
        chain_ids = [(heavy, light, cleaned_ag)]
        pairs.append((code, chain_ids))

    logger.info(
        f"[parse_list] final pairs: {len(pairs)}, "
        f"dropped (no valid antigen chain after AG_TYPES filter) = {dropped_all_ag}"
    )

    return pairs


# ======================
# 2. 结构处理工具函数（和原 AbX 基本一致）
# ======================


def make_chain_feature(chain):
    residues = list(chain.get_residues())
    residues = [
        r for r in residues if r.get_resname() in residue_constants.restype_3to1.keys()
    ]

    str_seq = [
        seq1(r.get_resname())
        for r in residues
        if r.get_resname() in residue_constants.restype_3to1.keys()
    ]
    N = len(str_seq)

    coords = np.zeros((N, 14, 3), dtype=np.float32)
    coord_mask = np.zeros((N, 14), dtype=bool)

    for jj, residue in enumerate(residues):
        if residue.get_resname() in residue_constants.restype_3to1.keys():
            res_atom14_list = residue_constants.restype_name_to_atom14_names[
                residue.resname
            ]
            for atom in residue.get_atoms():
                if atom.id not in res_atom14_list:
                    continue
                atom14idx = res_atom14_list.index(atom.id)
                coords[jj, atom14idx] = atom.get_coord()
                coord_mask[jj, atom14idx] = True

    feature = dict(
        str_seq="".join(str_seq),
        coords=coords,
        coord_mask=coord_mask,
    )
    return feature


def make_feature(str_seq, seq2struc, structure):
    n = len(str_seq)
    assert n > 0
    coords = np.zeros((n, 14, 3), dtype=np.float32)
    coord_mask = np.zeros((n, 14), dtype=bool)

    for seq_idx, residue_at_position in seq2struc.items():
        if not residue_at_position.is_missing and residue_at_position.hetflag == " ":
            residue_id = (
                residue_at_position.hetflag,
                residue_at_position.position.residue_number,
                residue_at_position.position.insertion_code,
            )

            residue = structure[residue_id]

            if residue.resname not in residue_constants.restype_name_to_atom14_names:
                continue
            res_atom14_list = residue_constants.restype_name_to_atom14_names[
                residue.resname
            ]
            for atom in residue.get_atoms():
                if atom.id not in res_atom14_list:
                    continue
                atom14idx = res_atom14_list.index(atom.id)
                coords[seq_idx, atom14idx] = atom.get_coord()
                coord_mask[seq_idx, atom14idx] = True

    feature = dict(str_seq=str_seq, coords=coords, coord_mask=coord_mask)
    return feature


def merge_chains(features):
    """
    与原 AbX 保持一致：
    - antibody 链（有 cdr_def 的）chain_id 从 0,1,... 开始
    - antigen 链 chain_id 从 2,3,... 开始，cdr_def 填 14
    """
    for i, data in enumerate(features):
        if "cdr_def" in data:
            chain_flag = 0
            prefix = "antibody"
        else:
            chain_flag = 2
            prefix = "antigen"

        chain_id = np.full((len(data["str_seq"])), i + chain_flag, dtype=np.int32)
        residx = np.arange(0, len(data["str_seq"]), dtype=np.int32)

        if prefix == "antibody" and i > 0:
            residx += residue_constants.residue_chain_index_offset
        if prefix == "antigen":
            cdr_def = np.full_like(chain_id, fill_value=14, dtype=np.int32)
            data.update(dict(cdr_def=cdr_def))

        data.update(dict(residx=residx))
        data.update(dict(chain_id=chain_id))

    chain_ids = np.concatenate([data["chain_id"] for data in features], axis=0)
    residx = np.concatenate([data["residx"] for data in features], axis=0)
    str_seq = "".join([data["str_seq"] for data in features])
    coord_mask = np.concatenate([data["coord_mask"] for data in features], axis=0)
    coords = np.concatenate([data["coords"] for data in features], axis=0)
    cdr_def = np.concatenate([data["cdr_def"] for data in features], axis=0)

    merge_features = dict(
        str_seq=str_seq,
        coords=coords,
        coord_mask=coord_mask,
        chain_ids=chain_ids,
        residx=residx,
        cdr_def=cdr_def,
    )
    merge_features = {f"{prefix}_{k}": v for k, v in merge_features.items()}

    return merge_features


# ======================
# 3. 关键：CDR-H3 是否接触抗原的过滤
# ======================


def _check_h3_contacts(features):
    """
    根据 AbX 的 npz 特征，检查：
    - heavy CDR-H3 中是否至少有一个残基的 CA 原子
    - 与任意抗原残基 CA 原子的距离 < CONTACT_DIST

    若满足：返回 True
    若不满足：返回 False
    """
    # antibody
    ab_coords = features["antibody_coords"]  # (N_ab, 14, 3)
    ab_mask = features["antibody_coord_mask"]  # (N_ab, 14)
    ab_cdr_def = features["antibody_cdr_def"]  # (N_ab,)
    ab_chain_ids = features["antibody_chain_ids"]  # (N_ab,)

    # antigen
    ag_coords = features["antigen_coords"]  # (N_ag, 14, 3)
    ag_mask = features["antigen_coord_mask"]  # (N_ag, 14)

    if ab_coords.shape[0] == 0 or ag_coords.shape[0] == 0:
        return False

    ca_idx = residue_constants.atom_order["CA"]

    # 对应 numbering.get_ab_regions 中定义：
    # heavy: fr1/cdr1/fr2/cdr2/fr3/cdr3/fr4 -> 0,1,2,3,4,5,6
    # light: fr1/cdr1/... -> 7..13
    # 所以 heavy CDR-H3 的标号 = 5
    H3_CODE = 5

    h3_mask = (ab_chain_ids == 0) & (ab_cdr_def == H3_CODE)
    if not np.any(h3_mask):
        # 理论上不应该发生（ANARCI 认不出来 H3），保守丢掉
        return False

    h3_ca_mask = h3_mask & ab_mask[:, ca_idx]
    if not np.any(h3_ca_mask):
        return False

    ag_ca_mask = ag_mask[:, ca_idx]
    if not np.any(ag_ca_mask):
        return False

    h3_ca = ab_coords[h3_ca_mask, ca_idx, :]  # (Nh3, 3)
    ag_ca = ag_coords[ag_ca_mask, ca_idx, :]  # (Nag, 3)

    # 计算 pairwise 距离
    diff = h3_ca[:, None, :] - ag_ca[None, :, :]
    dist2 = np.sum(diff * diff, axis=-1)  # (Nh3, Nag)
    min_dist = np.sqrt(dist2.min())

    return bool(min_dist < CONTACT_DIST)


# ======================
# 4. 生成 npz（mmCIF / PDB 两套管线）
# ======================

# NEW: 根据 IMGT / Chothia 保守位点过滤“非典型”抗体
def _check_conserved_sites(domain_seq, domain_numbering, chain_id):
    """
    domain_seq:  已经截到 variable domain 的序列 (str)
    domain_numbering: ANARCI 返回的 domain_numbering 列表（长度与 domain_seq 一致）
    chain_id: "H" 或 "L"
    使用全局 NUMBERING_SCHEME 和 CONSERVE_RULES。
    """
    scheme = NUMBERING_SCHEME.lower()
    if scheme not in CONSERVE_RULES:
        return True  # 未知 scheme，直接放过（按理只有 imgt/chothia）

    rules_for_scheme = CONSERVE_RULES[scheme]
    if chain_id not in rules_for_scheme:
        return True

    rules = rules_for_scheme[chain_id]  # dict: pos -> [allowed_aas]

    # domain_numbering 里每个元素一般是 (number, insertion_code, ...)
    # 我们只关心第一个数字部分
    for idx, num_info in enumerate(domain_numbering):
        try:
            pos = int(num_info[0])
        except Exception:
            continue

        if pos not in rules:
            continue

        aa = domain_seq[idx]
        if aa not in rules[pos]:
            logger.debug(
                f"[conserve] scheme={scheme}, chain={chain_id}, "
                f"pos={pos}, aa={aa}, allowed={rules[pos]} -> FAIL"
            )
            return False

    return True


def _check_cdr_complete(cdr_def, chain_id):
    """
    检查给定链（H 或 L）的 CDR1/2/3 是否都存在。
    依赖 get_ab_regions 的编码约定：
      - heavy: FR1/CDR1/FR2/CDR2/FR3/CDR3/FR4 -> 0,1,2,3,4,5,6
               => CDR-H1/H2/H3 的 code 分别为 1,3,5
      - light: FR1/CDR1/FR2/CDR2/FR3/CDR3/FR4 -> 7,8,9,10,11,12,13
               => CDR-L1/L2/L3 的 code 分别为 8,10,12
    """
    if chain_id == "H":
        cdr_codes = [1, 3, 5]
        cdr_names = ["H1", "H2", "H3"]
    else:
        # 这里默认 chain_id == "L"
        cdr_codes = [8, 10, 12]
        cdr_names = ["L1", "L2", "L3"]

    for code, name in zip(cdr_codes, cdr_names):
        if not np.any(cdr_def == code):
            raise ValueError(f"CDR-{name} not found in chain {chain_id}")


def _make_domain(feature, chain_id):
    """
    用 ANARCI(IMGT/Chothia) 把抗体链切到 variable domain，并得到 cdr_def，
    同时根据 IMGT/Chothia 的 Hconserve/Lconserve 过滤掉“非典型”抗体。
    """
    allow = ["H"] if chain_id == "H" else ["K", "L"]

    # 使用全局 NUMBERING_SCHEME
    anarci_res = renumber_ab_seq(
        feature["str_seq"], allow=allow, scheme=NUMBERING_SCHEME
    )
    domain_numbering, domain_start, domain_end = map(
        anarci_res.get, ["domain_numbering", "start", "end"]
    )

    if domain_numbering is None:
        raise ValueError(
            f"ANARCI failed for chain {chain_id} (len={len(feature['str_seq'])}), "
            "drop this chain / complex."
        )

    # 截取 variable domain 的序列
    domain_seq = feature["str_seq"][domain_start:domain_end]

    # NEW: 保守位点过滤
    if not _check_conserved_sites(domain_seq, domain_numbering, chain_id):
        raise ValueError(
            f"Conserved positions mismatch for chain {chain_id} under {NUMBERING_SCHEME}"
        )

    # 计算 CDR/FW 区域标签
    cdr_def = get_ab_regions(domain_numbering, chain_id=chain_id)

    # NEW: CDR1/2/3 完整性检查（模仿 dyMEAN 的 assert 逻辑）
    _check_cdr_complete(cdr_def, chain_id)

    updated_feature = {k: v[domain_start:domain_end] for k, v in feature.items()}
    domain_numbering_str = ",".join(
        ["".join([str(xx) for xx in x]).strip() for x in domain_numbering]
    )

    updated_feature.update(dict(cdr_def=cdr_def, numbering=domain_numbering_str))
    return updated_feature


def make_pdb_npz(struc, chain_ids, heavy_chain_id, light_chain_id, antigen_chain_id):
    all_chain = list(struc.get_chains())

    def _get_chain_info(chain_id):
        for chain in all_chain:
            if chain.id == chain_id:
                return chain
        return None

    antibody_feature = []
    features = {}

    if heavy_chain_id:
        heavy_feature = make_chain_feature(_get_chain_info(heavy_chain_id))
        heavy_feature = _make_domain(heavy_feature, "H")
        antibody_feature.append(heavy_feature)

    if light_chain_id:
        light_feature = make_chain_feature(_get_chain_info(light_chain_id))
        light_feature = _make_domain(light_feature, "L")
        antibody_feature.append(light_feature)

    features.update(merge_chains(antibody_feature))

    if antigen_chain_id:
        antigen_features = []
        for antigen_chain in antigen_chain_id:
            if antigen_chain not in chain_ids:
                continue
            antigen_feature = make_chain_feature(_get_chain_info(antigen_chain))
            antigen_features.append(antigen_feature)
        features.update(merge_chains(antigen_features))

    # ====== 关键过滤：CDR-H3 与抗原接触（可通过参数关闭） ======
    if REQUIRE_H3_CONTACT and not _check_h3_contacts(features):
        raise ValueError("CDR-H3 has no contact with antigen under 6.6 Å")

    return features


def make_npz(heavy_data, light_data, antigen_data):
    antibody_feature = []
    features = {}

    if heavy_data:
        str_seq, seq2struc, struc = map(
            heavy_data.get, ["str_seq", "seqres_to_structure", "struc"]
        )
        heavy_feature = make_feature(str_seq, seq2struc, struc)
        heavy_feature = _make_domain(heavy_feature, "H")
        antibody_feature.append(heavy_feature)

    if light_data:
        str_seq, seq2struc, struc = map(
            light_data.get, ["str_seq", "seqres_to_structure", "struc"]
        )
        light_feature = make_feature(str_seq, seq2struc, struc)
        light_feature = _make_domain(light_feature, "L")
        antibody_feature.append(light_feature)

    features.update(merge_chains(antibody_feature))

    if antigen_data:
        antigen_features = []
        for antigen_item in antigen_data:
            str_seq, seq2struc, struc = map(
                antigen_item.get, ["str_seq", "seqres_to_structure", "struc"]
            )
            antigen_feature = make_feature(str_seq, seq2struc, struc)
            antigen_features.append(antigen_feature)
        features.update(merge_chains(antigen_features))

    # ====== 关键过滤：CDR-H3 与抗原接触（可通过参数关闭） ======
    if REQUIRE_H3_CONTACT and not _check_h3_contacts(features):
        raise ValueError("CDR-H3 has no contact with antigen under 6.6 Å")

    return features


def save_feature(feature, code, heavy_chain_id, light_chain_id, antigen_chain_ids, output_dir):
    """
    feature: 合并好的特征
    code:    pdb id（parse_list 里给的是小写，比如 '2ghw'）
    heavy_chain_id / light_chain_id:  parse_list 传进来的原始 H/L 字符
    antigen_chain_ids: list，比如 ['A'] 或 ['A', 'B']
    """
    # 抗原链字符串，用于匹配 RAbD 特殊 case
    antigen_chain_str = "".join(antigen_chain_ids)

    # 默认：文件名就用当前的 (H, L, 抗原链串)
    out_H, out_L, out_A = heavy_chain_id, light_chain_id, antigen_chain_str

    # 如果是那 3 个 RAbD 里的特殊 pdb，就把文件名改成 dyMEAN 的那套
    key = (code.lower(), antigen_chain_str)
    if key in RABD_TEST_NAME_MAP:
        mapped_H, mapped_L, mapped_A = RABD_TEST_NAME_MAP[key]
        # 只改“文件名里的字母”，不改真正用来取结构的链 ID
        out_H, out_L, out_A = mapped_H, mapped_L, mapped_A

    fname = f"{code}_{out_H}_{out_L}_{out_A}.npz"
    np.savez(os.path.join(output_dir, fname), **feature)
    return



def save_header(header, file_path):
    with open(file_path, "w") as fw:
        json.dump(header, fw)


# ======================
# 5. mmCIF / PDB 两种处理入口
# ======================


def process(code, chain_ids, args):
    logger.info(f"processing {code}, {','.join(['_'.join(x) for x in chain_ids])}")
    mmcif_file = os.path.join(args.data_dir, f"{code}.cif")
    try:
        parsing_result = mmcif_parse(file_id=code, mmcif_file=mmcif_file)
    except PDBConstructionException as e:
        logger.warning("mmcif_parse: %s {%s}", mmcif_file, str(e))
        return
    except Exception as e:
        logger.warning("mmcif_parse: %s {%s}", mmcif_file, str(e))
        return

    if not parsing_result.mmcif_object:
        return

    struc = parsing_result.mmcif_object.structure

    def _parse_chain_id(heavy_chain_id, light_chain_id):
        if heavy_chain_id.islower() and heavy_chain_id.upper() == light_chain_id:
            heavy_chain_id = heavy_chain_id.upper()
        elif light_chain_id.islower() and light_chain_id.upper() == heavy_chain_id:
            light_chain_id = light_chain_id.upper()
        return heavy_chain_id, light_chain_id

    for orig_heavy_chain_id, orig_light_chain_id, orig_antigen_chain_id in chain_ids:
        antigen_chain_ids = orig_antigen_chain_id.split("|")
        antigen_chain_ids = [s.replace(" ", "") for s in antigen_chain_ids]

        heavy_chain_id, light_chain_id = _parse_chain_id(
            orig_heavy_chain_id, orig_light_chain_id
        )

        if (
            (heavy_chain_id and heavy_chain_id not in parsing_result.mmcif_object.chain_to_seqres)
            or (light_chain_id and light_chain_id not in parsing_result.mmcif_object.chain_to_seqres)
        ):
            logger.warning(
                "%s %s %s: chain ids not exist.",
                code,
                heavy_chain_id,
                light_chain_id,
            )
            continue

        # 确保所有 antigen 链在结构中存在
        flag = 0
        for antigen_chain_id in antigen_chain_ids:
            if antigen_chain_id not in parsing_result.mmcif_object.chain_to_seqres:
                logger.warning("antigen id: %s not exist", antigen_chain_id)
                flag += 1
        if flag > 0:
            continue

        if heavy_chain_id:
            heavy_data = dict(
                str_seq=parsing_result.mmcif_object.chain_to_seqres[heavy_chain_id],
                seqres_to_structure=parsing_result.mmcif_object.seqres_to_structure[
                    heavy_chain_id
                ],
                struc=struc[heavy_chain_id],
            )
        else:
            heavy_data = None

        if light_chain_id:
            light_data = dict(
                str_seq=parsing_result.mmcif_object.chain_to_seqres[light_chain_id],
                seqres_to_structure=parsing_result.mmcif_object.seqres_to_structure[
                    light_chain_id
                ],
                struc=struc[light_chain_id],
            )
        else:
            light_data = None

        antigen_data = []
        for antigen_chain_id in antigen_chain_ids:
            if antigen_chain_id not in parsing_result.mmcif_object.chain_to_seqres:
                continue
            antigen_data.append(
                dict(
                    str_seq=parsing_result.mmcif_object.chain_to_seqres[antigen_chain_id],
                    seqres_to_structure=parsing_result.mmcif_object.seqres_to_structure[
                        antigen_chain_id
                    ],
                    struc=struc[antigen_chain_id],
                )
            )

        try:
            feature = make_npz(heavy_data, light_data, antigen_data)
            save_feature(
                feature,
                code,
                orig_heavy_chain_id,
                orig_light_chain_id,
                antigen_chain_ids,
                args.output_dir,
            )
            logger.info(
                "succeed: %s %s %s", mmcif_file, orig_heavy_chain_id, orig_light_chain_id
            )
        except Exception as e:
            traceback.print_exc()
            logger.error(
                "make structure: %s %s %s {%s}",
                mmcif_file,
                orig_heavy_chain_id,
                orig_light_chain_id,
                str(e),
            )


def process_pdb(code, chain_ids, args):
    logger.info(f"processing {code}, {','.join(['_'.join(x) for x in chain_ids])}")
    pdb_file = os.path.join(args.data_dir, f"{code}.pdb")
    try:
        parser = PDBParser()
        struc = parser.get_structure("model", pdb_file)
    except PDBConstructionException as e:
        logger.warning("PDB parse: %s {%s}", pdb_file, str(e))
        return
    except Exception as e:
        logger.warning("PDB parse: %s {%s}", pdb_file, str(e))
        return

    pdb_chain_id = [c.id for c in list(struc.get_chains())]

    def _parse_chain_id(heavy_chain_id, light_chain_id):
        if heavy_chain_id.islower() and heavy_chain_id.upper() == light_chain_id:
            heavy_chain_id = heavy_chain_id.upper()
        elif light_chain_id.islower() and light_chain_id.upper() == heavy_chain_id:
            light_chain_id = light_chain_id.upper()
        return heavy_chain_id, light_chain_id

    for orig_heavy_chain_id, orig_light_chain_id, orig_antigen_chain_id in chain_ids:
        antigen_chain_ids = orig_antigen_chain_id.split("|")
        antigen_chain_ids = [s.replace(" ", "") for s in antigen_chain_ids]
        heavy_chain_id, light_chain_id = _parse_chain_id(
            orig_heavy_chain_id, orig_light_chain_id
        )

        if (
            (heavy_chain_id and heavy_chain_id not in pdb_chain_id)
            or (light_chain_id and light_chain_id not in pdb_chain_id)
        ):
            logger.warning(
                "%s %s %s: chain ids not exist.",
                code,
                heavy_chain_id,
                light_chain_id,
            )
            continue

        flag = 0
        for antigen_chain_id in antigen_chain_ids:
            if antigen_chain_id not in pdb_chain_id:
                logger.warning("antigen id: %s not exist", antigen_chain_id)
                flag += 1
        if flag > 0:
            continue

        try:
            feature = make_pdb_npz(
                struc, pdb_chain_id, heavy_chain_id, light_chain_id, antigen_chain_ids
            )
            save_feature(
                feature,
                code,
                orig_heavy_chain_id,
                orig_light_chain_id,
                antigen_chain_ids,
                args.output_dir,
            )
            logger.info(
                "succeed: %s %s %s", pdb_file, orig_heavy_chain_id, orig_light_chain_id
            )
        except Exception as e:
            traceback.print_exc()
            logger.error(
                "make structure: %s %s %s {%s}",
                pdb_file,
                orig_heavy_chain_id,
                orig_light_chain_id,
                str(e),
            )


# ======================
# 6. main
# ======================


def main():
    global NUMBERING_SCHEME, REQUIRE_H3_CONTACT  # NEW: 加上 REQUIRE_H3_CONTACT
    parser = argparse.ArgumentParser()

    parser.add_argument("--cpus", type=int, default=1)
    parser.add_argument("--summary_file", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--verbose", action="store_true", help="verbose")
    parser.add_argument("--data_mode", type=str, required=True, help="pdb or mmcif")
        # NEW: H3 接触开关
    parser.add_argument(
        "--disable_h3_contact",
        action="store_true",
        help="If set, DO NOT require CDR-H3 to contact antigen (default: require contact).",
    )

    # NEW: 编号方案
    parser.add_argument(
        "--numbering_scheme",
        type=str,
        default="imgt",
        choices=["imgt", "chothia"],
        help="Antibody numbering scheme used by ANARCI (imgt or chothia)",
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 设置全局编号方案
    NUMBERING_SCHEME = args.numbering_scheme.lower()

    # 设置 H3 接触过滤开关（默认 True；加上 --disable_h3_contact 就关闭）
    REQUIRE_H3_CONTACT = not args.disable_h3_contact
    
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s:%(name)s:%(message)s",
    )

    if args.data_mode == "mmcif":
        func = functools.partial(process, args=args)
    else:
        func = functools.partial(process_pdb, args=args)

    items = parse_list(args.summary_file)
    logger.info(f"[main] total items to process: {len(items)}")

    with mp.Pool(args.cpus) as p:
        p.starmap(func, items)


if __name__ == "__main__":
    main()
