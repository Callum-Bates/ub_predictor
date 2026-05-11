# ------------------------------------------------------------
# features/rasa.py
#
# calculates relative accessible surface area (rasa) for each
# lysine site and its structural surroundings.
#
# rasa measures how exposed a residue is to solvent - a value
# of 1.0 means fully exposed (on the protein surface), 0.0
# means fully buried. for ubiquitination, surface accessibility
# matters because the e3 ligase machinery must physically reach
# the lysine sidechain to attach ubiquitin.
# many lysine ub sites are in fact buried within the protein
# (low rasa value) - there must be some opening mechanism
# for these sites.
#
# measurements from alpha carbon - most robust.
#
# features calculated:
#   rasa_lysine           - exposure of the lysine itself
#   rasa_sphere_mean      - mean exposure of residues within
#                           sphere_radius angstroms of lysine ca
#   rasa_sphere_std       - variation in exposure in that sphere
#   rasa_sphere_median    - median exposure in sphere
#   rasa_sphere_max       - maximum exposure in sphere
#   n_sphere_residues     - number of residues in sphere
#   mean_n_terminal_rasa  - mean rasa of n-terminal half of protein
#   mean_c_terminal_rasa  - mean rasa of c-terminal half of protein
#   plddt_lysine          - alphafold confidence score for lysine
#                           from b-factor column (0-100)
#
# sphere radius defaults to 8.0 angstroms - consistent with
# standard residue contact definitions in structural biology
# (pejaver et al. 2019, mol cell proteomics). parametric -
# user can adjust for benchmarking.
#
# sasa calculated using shrake-rupley algorithm via biopython.
# rasa is sasa normalised by maximum asa for that amino acid
# type (tien et al. 2013).
#
# input : sites dataframe with sequence features added
#         cif files in data/structures/
# output: same dataframe with rasa feature columns added
# ------------------------------------------------------------

import numpy as np
import pandas as pd
import warnings
import logging
from pathlib import Path

from Bio.PDB import MMCIFParser
from Bio.PDB.SASA import ShrakeRupley

log = logging.getLogger(__name__)

# maximum asa values per amino acid type (tien et al. 2013)
# used to normalise raw sasa values to rasa (0-1 range)
MAX_ASA = {
    "ALA": 121.0, "ARG": 265.0, "ASN": 187.0, "ASP": 187.0,
    "CYS": 148.0, "GLU": 214.0, "GLN": 214.0, "GLY":  97.0,
    "HIS": 216.0, "ILE": 195.0, "LEU": 191.0, "LYS": 230.0,
    "MET": 203.0, "PHE": 228.0, "PRO": 154.0, "SER": 143.0,
    "THR": 163.0, "TRP": 264.0, "TYR": 255.0, "VAL": 165.0,
}

# default sphere radius for neighbourhood rasa calculation
# 8.0 angstroms captures residues in direct contact with lysine
# consistent with standard residue contact definitions
SPHERE_RADIUS = 8.0


# ------------------------------------------------------------
# null row helper
# ------------------------------------------------------------

def _null_rasa_row(protein_id, position, sphere_radius):
    """
    Return a row of nulls for a site where calculation failed.

    params:
        protein_id    : uniprot accession
        position      : lysine position
        sphere_radius : sphere radius used
    """
    return {
        "protein_id"          : protein_id,
        "lysine_position"     : position,
        "rasa_lysine"         : None,
        "rasa_sphere_mean"    : None,
        "rasa_sphere_std"     : None,
        "rasa_sphere_median"  : None,
        "rasa_sphere_max"     : None,
        "n_sphere_residues"   : None,
        "sphere_radius_a"     : sphere_radius,
        "mean_n_terminal_rasa": None,
        "mean_c_terminal_rasa": None,
        "plddt_lysine"        : None,
    }


# ------------------------------------------------------------
# step 1 | calculate rasa for one lysine site
# ------------------------------------------------------------

def _calc_rasa_from_structure(protein_id, position, structure,
                               sphere_radius):
    """
    Calculate rasa features for one lysine using a pre-parsed structure.

    sasa must already be computed on the structure before calling this.
    called internally by add_rasa_features which handles parsing
    and sasa calculation once per protein.

    params:
        protein_id    : uniprot accession
        position      : 1-based lysine position
        structure     : biopython structure object with sasa computed
        sphere_radius : angstrom radius for neighbourhood calculation
    """
    try:
        # locate the target lysine
        target_residue = None
        target_ca      = None
        plddt_lysine   = None

        for model in structure:
            for chain in model:
                for residue in chain:
                    if (residue.get_id()[1] == position and
                            residue.get_resname() == "LYS"):
                        target_residue = residue
                        if "CA" in residue:
                            target_ca    = residue["CA"].get_coord()
                            # alphafold stores plddt in b-factor column
                            # ranges 0-100, higher = more confident
                            plddt_lysine = round(
                                float(residue["CA"].get_bfactor()), 2
                            )
                        break
                if target_residue:
                    break
            if target_residue:
                break

        if target_residue is None:
            log.warning(
                f"  {protein_id} position {position} - "
                f"lysine not found in structure"
            )
            return None

        if target_ca is None:
            log.warning(
                f"  {protein_id} position {position} - "
                f"lysine has no ca atom"
            )
            return None

        # rasa for the lysine itself
        lys_sasa = target_residue.sasa
        lys_rasa = min(lys_sasa / MAX_ASA["LYS"], 1.0)

        # rasa for residues within sphere_radius of lysine ca
        sphere_rasa_values = []

        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue == target_residue:
                        continue
                    resname = residue.get_resname()
                    if resname not in MAX_ASA:
                        continue
                    if "CA" not in residue:
                        continue
                    distance = np.linalg.norm(
                        target_ca - residue["CA"].get_coord()
                    )
                    if distance <= sphere_radius:
                        rasa = min(residue.sasa / MAX_ASA[resname], 1.0)
                        sphere_rasa_values.append(rasa)

        if sphere_rasa_values:
            sphere_mean   = round(float(np.mean(sphere_rasa_values)),   4)
            sphere_std    = round(float(np.std(sphere_rasa_values)),    4)
            sphere_median = round(float(np.median(sphere_rasa_values)), 4)
            sphere_max    = round(float(np.max(sphere_rasa_values)),    4)
            n_sphere      = len(sphere_rasa_values)
        else:
            sphere_mean   = None
            sphere_std    = None
            sphere_median = None
            sphere_max    = None
            n_sphere      = 0

        return {
            "protein_id"          : protein_id,
            "lysine_position"     : position,
            "rasa_lysine"         : round(lys_rasa, 4),
            "rasa_sphere_mean"    : sphere_mean,
            "rasa_sphere_std"     : sphere_std,
            "rasa_sphere_median"  : sphere_median,
            "rasa_sphere_max"     : sphere_max,
            "n_sphere_residues"   : n_sphere,
            "sphere_radius_a"     : sphere_radius,
            "plddt_lysine"        : plddt_lysine,
        }

    except Exception as e:
        log.warning(
            f"  {protein_id} position {position} rasa failed: {e}"
        )
        return None


# ------------------------------------------------------------
# step 2 | run for all sites in the dataframe
# ------------------------------------------------------------

def add_rasa_features(sites_df, cif_dir, sphere_radius=SPHERE_RADIUS):
    """
    Add rasa features to a dataframe of lysine sites.

    structures are parsed and sasa calculated once per protein,
    then reused for all lysines from that protein. this avoids
    re-parsing the same cif file dozens of times for proteins
    with many lysine sites.

    also calculates whole-protein rasa statistics (n-terminal
    and c-terminal mean rasa) once per protein.

    params:
        sites_df      : dataframe with protein_id, lysine_position
        cif_dir       : directory containing alphafold cif files
        sphere_radius : angstrom radius for neighbourhood rasa
    """
    cif_dir   = Path(cif_dir)
    total     = len(sites_df)
    proteins  = sites_df["protein_id"].unique()
    rasa_rows = []
    failed    = 0

    print(f"\n  calculating rasa features for {total} sites")
    print(f"  sphere radius: {sphere_radius}a")
    print(f"  processing {len(proteins)} unique proteins")

    for protein_id in proteins:

        cif_path      = cif_dir / f"{protein_id}.cif"
        protein_sites = sites_df[sites_df["protein_id"] == protein_id]

        if not cif_path.exists():
            log.warning(f"  {protein_id} no cif file found - skipping")
            for _, row in protein_sites.iterrows():
                rasa_rows.append(
                    _null_rasa_row(protein_id,
                                   int(row["lysine_position"]),
                                   sphere_radius)
                )
                failed += 1
            continue

        try:
            # parse structure and calculate sasa once for this protein
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parser    = MMCIFParser(QUIET=True)
                structure = parser.get_structure(
                    protein_id, str(cif_path)
                )

            sr = ShrakeRupley()
            sr.compute(structure, level="R")

            # calculate whole-protein rasa once per protein
            # split into n-terminal and c-terminal halves
            # used as global structural context features
            all_residues = []
            for model in structure:
                for chain in model:
                    for residue in chain:
                        resname = residue.get_resname()
                        if resname not in MAX_ASA:
                            continue
                        pos  = residue.get_id()[1]
                        rasa = min(residue.sasa / MAX_ASA[resname], 1.0)
                        all_residues.append({
                            "position": pos,
                            "rasa"    : rasa,
                        })

            if all_residues:
                all_pos  = [r["position"] for r in all_residues]
                midpoint = (min(all_pos) + max(all_pos)) / 2

                n_term_rasa = [
                    r["rasa"] for r in all_residues
                    if r["position"] <= midpoint
                ]
                c_term_rasa = [
                    r["rasa"] for r in all_residues
                    if r["position"] > midpoint
                ]

                protein_n_term_rasa = (
                    round(float(np.mean(n_term_rasa)), 4)
                    if n_term_rasa else None
                )
                protein_c_term_rasa = (
                    round(float(np.mean(c_term_rasa)), 4)
                    if c_term_rasa else None
                )
            else:
                protein_n_term_rasa = None
                protein_c_term_rasa = None

            # process each lysine site for this protein
            for _, row in protein_sites.iterrows():
                position = int(row["lysine_position"])
                result   = _calc_rasa_from_structure(
                    protein_id, position, structure, sphere_radius
                )

                if result is not None:
                    result["mean_n_terminal_rasa"] = protein_n_term_rasa
                    result["mean_c_terminal_rasa"] = protein_c_term_rasa
                    rasa_rows.append(result)
                else:
                    failed += 1
                    rasa_rows.append(
                        _null_rasa_row(protein_id, position, sphere_radius)
                    )

        except Exception as e:
            log.warning(
                f"  {protein_id} structure processing failed: {e}"
            )
            for _, row in protein_sites.iterrows():
                rasa_rows.append(
                    _null_rasa_row(protein_id,
                                   int(row["lysine_position"]),
                                   sphere_radius)
                )
                failed += 1

    if failed > 0:
        print(f"  {failed} sites failed rasa calculation - set to null")

    rasa_df = pd.DataFrame(rasa_rows)

    result = sites_df.merge(
        rasa_df,
        on  = ["protein_id", "lysine_position"],
        how = "left"
    )

    print(f"  rasa features added")
    print(f"  feature matrix shape: {result.shape}")

    return result