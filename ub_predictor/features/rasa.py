# ------------------------------------------------------------
# features/rasa.py
#
# calculates relative accessible surface area (rasa) for each
# lysine site and its structural srurroundings.
#
# rasa measures how exposed a residue is to solvent - a value
# of 1.0 means fully exposed (on the protein surface), 0.0
# means fully buried. for ubiquitination, surface accessibility
# matters because the e3 ligase machinery must physically reach
# the lysine sidechain to attach ubiquitin.
# I did however find that many lysine Ub sites are infact buried
# within the protein (low sa value) and thus there must be some 
#sort of opening mechanism for these sites.
#
# Measurements from alpha carbon - most robust 
#   
# two features are calculated:
#   rasa_lysine      - exposure of the lysine itself
#   rasa_sphere_mean - mean exposure of residues within
#                      sphere_radius angstroms of the lysine ca
#   rasa_sphere_std  - variation in exposure in that sphere
#   n_sphere_residues - number of residues in the sphere
#
# the sphere radius defaults to 8.0 angstroms - consistent with
# standard residue contact definitions in structural biology
# (pejaver et al. 2019, mol cell proteomics). this can be
# adjusted as a parameter for benchmarking.
# User will have freedom to change this parameter depending on goals/
# results.
#
# sasa is calculated using the shrake-rupley algorithm via
# biopython. rasa is sasa normalised by the maximum asa for
# that amino acid type (tien et al. 2013).
#
# input : sites_structured.csv with sequence features added
#         cif files in data/structures/
# (this is the second feature set added)
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
# 8.0 angstroms captures residues in direct contact with the lysine
# consistent with standard residue contact definitions
SPHERE_RADIUS = 5.0


# ------------------------------------------------------------
# step 1 | calculate rasa for one lysine site
# ------------------------------------------------------------

def calc_rasa_for_site(protein_id, position, cif_dir, sphere_radius=SPHERE_RADIUS):
    """
    Calculate rasa features for a single lysine site.

    opens the cif file, runs shrake-rupley sasa calculation,
    then extracts rasa for the lysine and its sphere neighbours.

    params:
        protein_id    : uniprot accession e.g. "P04637"
        position      : 1-based lysine position
        cif_dir       : directory containing cif files
        sphere_radius : angstrom radius for neighbourhood calculation
    
    returns dict of rasa features, or None if calculation fails
    """
    cif_path = Path(cif_dir) / f"{protein_id}.cif"

    if not cif_path.exists():
        log.warning(f"  {protein_id} no cif file found at {cif_path}")
        return None

    try:
        # parse the structure - QUIET suppresses biopython warnings
        # about non-standard residues which are common in alphafold files
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parser    = MMCIFParser(QUIET=True)
            structure = parser.get_structure(protein_id, str(cif_path))

        # calculate sasa for all residues using shrake-rupley
        # level="R" means per-residue (not per-atom)
        sr = ShrakeRupley()
        sr.compute(structure, level="R")

        # locate the target lysine residue
        target_residue = None
        target_ca      = None

        for model in structure:
            for chain in model:
                for residue in chain:
                    if (residue.get_id()[1] == position and
                            residue.get_resname() == "LYS"):
                        target_residue = residue
                        if "CA" in residue:
                            target_ca = residue["CA"].get_coord()
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

        # calculate rasa for the lysine itself
        lys_sasa = target_residue.sasa
        lys_rasa = min(lys_sasa / MAX_ASA["LYS"], 1.0)

        # find all residues within sphere_radius of the lysine ca
        # and calculate their rasa values
        sphere_rasa_values = []

        for model in structure:
            for chain in model:
                for residue in chain:

                    # skip the lysine itself
                    if residue == target_residue:
                        continue

                    resname = residue.get_resname()

                    # skip non-standard residues
                    if resname not in MAX_ASA:
                        continue

                    # skip residues with no ca atom
                    if "CA" not in residue:
                        continue

                    # calculate distance from lysine ca to this residue ca
                    ca_coord = residue["CA"].get_coord()
                    distance = np.linalg.norm(target_ca - ca_coord)

                    if distance <= sphere_radius:
                        rasa = min(residue.sasa / MAX_ASA[resname], 1.0)
                        sphere_rasa_values.append(rasa)

        # summarise sphere neighbourhood
        if sphere_rasa_values:
            sphere_mean = round(float(np.mean(sphere_rasa_values)), 4)
            sphere_std  = round(float(np.std(sphere_rasa_values)), 4)
            n_sphere    = len(sphere_rasa_values)
        else:
            sphere_mean = None
            sphere_std  = None
            n_sphere    = 0

        return {
            "protein_id"       : protein_id,
            "lysine_position"  : position,
            "rasa_lysine"      : round(lys_rasa, 4),
            "rasa_sphere_mean" : sphere_mean,
            "rasa_sphere_std"  : sphere_std,
            "n_sphere_residues": n_sphere,
            "sphere_radius_a"  : sphere_radius,
        }

    except Exception as e:
        log.warning(
            f"  {protein_id} position {position} rasa calculation failed: {e}"
        )
        return None


# ------------------------------------------------------------
# step 2 | run for all sites in the dataframe
# ------------------------------------------------------------

def add_rasa_features(sites_df, cif_dir, sphere_radius=SPHERE_RADIUS):
    """
    Add rasa features to a dataframe of lysine sites.

    follows the standard feature module interface -
    takes a dataframe in, returns it with new columns added.

    params:
        sites_df      : dataframe with protein_id, lysine_position columns
        cif_dir       : directory containing alphafold cif files
        sphere_radius : angstrom radius for neighbourhood rasa -
                        defaults to 8.0a, adjust for benchmarking
    
    returns sites_df with rasa feature columns added
    """
    cif_dir = Path(cif_dir)
    total   = len(sites_df)

    print(f"\n  calculating rasa features for {total} sites")
    print(f"  sphere radius: {sphere_radius}a")

    rasa_rows = []
    failed    = 0

    for i, (_, row) in enumerate(sites_df.iterrows(), 1):

        if i % 10 == 0 or i == total:
            print(f"  {i}/{total} sites processed")

        pid    = row["protein_id"]
        pos    = int(row["lysine_position"])
        result = calc_rasa_for_site(pid, pos, cif_dir, sphere_radius)

        if result is None:
            failed += 1
            # add a row of nulls so we can still merge cleanly
            rasa_rows.append({
                "protein_id"       : pid,
                "lysine_position"  : pos,
                "rasa_lysine"      : None,
                "rasa_sphere_mean" : None,
                "rasa_sphere_std"  : None,
                "n_sphere_residues": None,
                "sphere_radius_a"  : sphere_radius,
            })
        else:
            rasa_rows.append(result)

    if failed > 0:
        print(f"  {failed} sites failed rasa calculation - set to null")

    rasa_df = pd.DataFrame(rasa_rows)

    # merge back onto original dataframe to preserve all columns
    result = sites_df.merge(
        rasa_df,
        on=["protein_id", "lysine_position"],
        how="left"
    )

    print(f"  rasa features added")
    print(f"  feature matrix shape: {result.shape}")

    return result