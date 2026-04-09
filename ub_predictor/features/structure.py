# ------------------------------------------------------------
# features/structure.py
#
# extracts secondary structure annotations for each lysine
# site from alphafold cif files.
#
# secondary structure describes the local backbone conformation:
#   H - alpha helix
#   E - beta sheet (extended strand)
#   C - coil (everything else - loops, turns, unstructured)
#
# a lysine in a helix sits in a rigid, regular environment.
# a lysine in a coil region is more flexible and potentially
# more accessible to e3 ligase machinery. these distinctions
# are biologically relevant and complement the spatial and
# rasa features.
#
# secondary structure is read directly from the alphafold cif
# file using biopython's DSSP-derived annotations - no external
# tool required.
#
# input : sites dataframe with previous features added
#         cif files in data/structures/
# output: same dataframe with secondary structure columns added
# ------------------------------------------------------------

import numpy as np
import pandas as pd
import warnings
import logging
from pathlib import Path

from Bio.PDB import MMCIFParser

log = logging.getLogger(__name__)

# window of residues either side of the lysine for
# secondary structure composition calculation
SS_WINDOW = 5

# secondary structure code mappings from alphafold cif files
# all annotations live in _struct_conf regardless of type
SS_CODE_MAP = {
    "HELX_RH_AL_P" : "H",   # alpha helix
    "HELX_RH_3T_P" : "H",   # 3-10 helix
    "HELX_RH_PI_P" : "H",   # pi helix
    "HELX_LH_PP_P" : "H",   # polyproline helix
    "STRN"         : "E",   # beta strand
    "TURN_TY1_P"   : "T",   # turn
    "BEND"         : "T",   # bend - structurally similar to turn
}
# anything not in SS_CODE_MAP defaults to coil ("C")

# ------------------------------------------------------------
# step 1 | extract secondary structure from cif file
# ------------------------------------------------------------

def read_secondary_structure(protein_id, cif_dir):
    """
    Read per-residue secondary structure from an alphafold cif file.

    alphafold stores all secondary structure annotations in the
    _struct_conf record - helices, strands, turns and bends are
    all there. residues not covered by any annotation are coil.

    params:
        protein_id : uniprot accession e.g. "P04637"
        cif_dir    : directory containing cif files

    returns dict of {position: ss_code} where ss_code is H, E, T, or C
    """
    cif_path = Path(cif_dir) / f"{protein_id}.cif"

    if not cif_path.exists():
        log.warning(f"  {protein_id} no cif file found")
        return {}

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parser    = MMCIFParser(QUIET=True)
            structure = parser.get_structure(protein_id, str(cif_path))

        # initialise all residues as coil
        ss_map = {}
        for model in structure:
            for chain in model:
                for residue in chain:
                    ss_map[residue.get_id()[1]] = "C"

        # read secondary structure from mmcif dict
        from Bio.PDB.MMCIF2Dict import MMCIF2Dict
        mmcif_dict = MMCIF2Dict(str(cif_path))

        if "_struct_conf.conf_type_id" not in mmcif_dict:
            log.warning(f"  {protein_id} no secondary structure records in cif")
            return ss_map

        conf_types  = mmcif_dict["_struct_conf.conf_type_id"]
        beg_seq_ids = mmcif_dict["_struct_conf.beg_label_seq_id"]
        end_seq_ids = mmcif_dict["_struct_conf.end_label_seq_id"]

        # handle single entry (string) vs multiple entries (list)
        if isinstance(conf_types, str):
            conf_types  = [conf_types]
            beg_seq_ids = [beg_seq_ids]
            end_seq_ids = [end_seq_ids]

        for conf_type, beg, end in zip(conf_types, beg_seq_ids, end_seq_ids):
            ss_code = SS_CODE_MAP.get(conf_type, "C")
            for pos in range(int(beg), int(end) + 1):
                ss_map[pos] = ss_code

        return ss_map

    except Exception as e:
        log.warning(f"  {protein_id} secondary structure reading failed: {e}")
        return {}

# ------------------------------------------------------------
# step 2 | calculate features for one lysine
# ------------------------------------------------------------

def calc_structure_features(protein_id, position, ss_map, window=SS_WINDOW):
    """
    Calculate secondary structure features for one lysine site.

    params:
        protein_id : uniprot accession
        position   : 1-based lysine position
        ss_map     : dict of {position: ss_code} from read_secondary_structure
        window     : number of residues either side to consider
    
    returns dict of secondary structure features
    """
    features = {
        "protein_id"      : protein_id,
        "lysine_position" : position,
    }

    # secondary structure of the lysine itself
    features["ss_lysine"] = ss_map.get(position, "C")

    # composition of the local window
    window_ss = []
    for offset in range(-window, window + 1):
        if offset == 0:
            continue  # skip the lysine itself
        pos = position + offset
        window_ss.append(ss_map.get(pos, "C"))

    n_window = len(window_ss)

    if n_window > 0:
            features["ss_window_helix_fraction"] = round(
                window_ss.count("H") / n_window, 4
            )
            features["ss_window_sheet_fraction"] = round(
                window_ss.count("E") / n_window, 4
            )
            features["ss_window_turn_fraction"]  = round(
                window_ss.count("T") / n_window, 4
            )
            features["ss_window_coil_fraction"]  = round(
                window_ss.count("C") / n_window, 4
            )
    else:
        features["ss_window_helix_fraction"] = 0.0
        features["ss_window_sheet_fraction"] = 0.0
        features["ss_window_turn_fraction"]  = 0.0
        features["ss_window_coil_fraction"]  = 0.0

    # per-position secondary structure in window
    # gives the model more granular information than just fractions
    for offset in range(-window, window + 1):
        if offset == 0:
            continue
        pos = position + offset
        features[f"ss_k{offset:+d}"] = ss_map.get(pos, "C")

    return features


# ------------------------------------------------------------
# step 3 | run for all sites in the dataframe
# ------------------------------------------------------------

def add_structure_features(sites_df, cif_dir, window=SS_WINDOW):
    """
    Add secondary structure features to a dataframe of lysine sites.

    follows the standard feature module interface -
    takes a dataframe in, returns it with new columns added.

    params:
        sites_df : dataframe with protein_id, lysine_position columns
        cif_dir  : directory containing alphafold cif files
        window   : residues either side for composition calculation
    
    returns sites_df with secondary structure feature columns added
    """
    cif_dir     = Path(cif_dir)
    total       = len(sites_df)
    protein_ids = sites_df["protein_id"].unique().tolist()

    print(f"\n  calculating secondary structure features for {total} sites")
    print(f"  window: +-{window} residues")

    # read secondary structure once per protein - not once per site
    # avoids re-parsing the same cif file hundreds of times
    ss_maps = {}
    print(f"  reading secondary structure from {len(protein_ids)} cif files")

    for protein_id in protein_ids:
        ss_maps[protein_id] = read_secondary_structure(protein_id, cif_dir)

    # calculate features for each site
    structure_rows = []
    failed         = 0

    for i, (_, row) in enumerate(sites_df.iterrows(), 1):

        if i % 10 == 0 or i == total:
            print(f"  {i}/{total} sites processed")

        pid = row["protein_id"]
        pos = int(row["lysine_position"])

        if not ss_maps.get(pid):
            log.warning(f"  {pid} no secondary structure data - skipping")
            failed += 1
            continue

        features = calc_structure_features(pid, pos, ss_maps[pid], window)
        structure_rows.append(features)

    if failed > 0:
        print(f"  {failed} sites skipped - no secondary structure data")

    structure_df = pd.DataFrame(structure_rows)

    result = sites_df.merge(
        structure_df,
        on=["protein_id", "lysine_position"],
        how="left"
    )

    n_features = len(structure_df.columns) - 2
    print(f"  {n_features} secondary structure features added")
    print(f"  feature matrix shape: {result.shape}")

    return result