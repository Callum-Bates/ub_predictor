# ------------------------------------------------------------
# features/spatial.py
#
# calculates 3d spatial neighbourhood features for each lysine.
#
# for each lysine, finds the N nearest residues in 3d space
# that are sequence-distant (more than sequence_separation
# positions away in the chain). these are residues that are
# far apart in sequence but folded into physical proximity -
# the genuinely structural contacts that sequence features miss.
#
# each neighbour is described by:
#   - residue identity
#   - distance from lysine ca in angstroms
#   - spherical coordinates (phi, theta) relative to lysine
#   - relative accessible surface area
#   - whether the closest contact is backbone or sidechain
#
# spherical coordinates are calculated in a local reference
# frame defined by the lysine backbone atoms (n, ca, c).
# this makes features rotation-invariant - the same structural
# motif looks identical regardless of protein orientation.
#
# input : sites dataframe with sequence and rasa features added
#         cif files in data/structures/
# output: same dataframe with spatial feature columns added
# ------------------------------------------------------------

import numpy as np
import pandas as pd
import warnings
import logging
from pathlib import Path

from Bio.PDB import MMCIFParser
from Bio.PDB.SASA import ShrakeRupley

log = logging.getLogger(__name__)

# maximum asa values (tien et al. 2013) - same as rasa.py
MAX_ASA = {
    "ALA": 121.0, "ARG": 265.0, "ASN": 187.0, "ASP": 187.0,
    "CYS": 148.0, "GLU": 214.0, "GLN": 214.0, "GLY":  97.0,
    "HIS": 216.0, "ILE": 195.0, "LEU": 191.0, "LYS": 230.0,
    "MET": 203.0, "PHE": 228.0, "PRO": 154.0, "SER": 143.0,
    "THR": 163.0, "TRP": 264.0, "TYR": 255.0, "VAL": 165.0,
}

# three letter to one letter amino acid conversion
AA_3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}

# number of nearest neighbours to record per lysine
N_NEIGHBOURS = 10

# residues within this many positions in sequence are ignored -
# they are already captured by sequence window features
SEQUENCE_SEPARATION = 5


# ------------------------------------------------------------
# coordinate geometry functions
# ------------------------------------------------------------

def build_local_frame(n_coord, ca_coord, c_coord):
    """
    Build a local coordinate frame from lysine backbone atoms.

    defines three orthogonal axes using the n, ca, and c atoms
    of the lysine. expressing neighbour positions in this frame
    makes features rotation-invariant across proteins.

    params:
        n_coord  : numpy array, coordinates of backbone N atom
        ca_coord : numpy array, coordinates of CA atom
        c_coord  : numpy array, coordinates of C atom

    returns 3x3 rotation matrix (columns are x, y, z axes)
    """
    # x axis - along ca to c bond
    x = c_coord - ca_coord
    x = x / np.linalg.norm(x)

    # y axis - in the plane of n-ca-c, perpendicular to x
    ca_n = n_coord - ca_coord
    y    = ca_n - np.dot(ca_n, x) * x
    y    = y / np.linalg.norm(y)

    # z axis - perpendicular to both (cross product)
    z = np.cross(x, y)

    return np.column_stack([x, y, z])


def cartesian_to_spherical(vector, rotation_matrix):
    """
    Convert a cartesian displacement vector to spherical coordinates
    in a local reference frame.

    params:
        vector          : numpy array, displacement from lysine ca
                          to neighbour ca
        rotation_matrix : local frame from build_local_frame

    returns (distance, phi, theta) where:
        distance - magnitude in angstroms
        phi      - polar angle in degrees (0-180)
        theta    - azimuthal angle in degrees (0-360)
    """
    distance = np.linalg.norm(vector)

    if distance < 1e-6:
        return 0.0, 0.0, 0.0

    # express vector in local frame
    local = rotation_matrix.T @ vector
    local_unit = local / distance

    # convert to spherical
    phi   = np.degrees(np.arccos(np.clip(local_unit[2], -1.0, 1.0)))
    theta = np.degrees(np.arctan2(local_unit[1], local_unit[0])) % 360

    return round(distance, 3), round(phi, 2), round(theta, 2)


# ------------------------------------------------------------
# step 1 | calculate spatial features for one lysine
# ------------------------------------------------------------

def calc_spatial_for_site(
    protein_id,
    position,
    cif_dir,
    n_neighbours=N_NEIGHBOURS,
    sequence_separation=SEQUENCE_SEPARATION
):
    """
    Calculate spatial neighbourhood features for one lysine site.

    params:
        protein_id          : uniprot accession e.g. "P04637"
        position            : 1-based lysine position
        cif_dir             : directory containing cif files
        n_neighbours        : number of nearest neighbours to record
        sequence_separation : minimum sequence distance for a residue
                              to be considered a spatial neighbour
    
    returns dict of spatial features, or None if calculation fails
    """
    cif_path = Path(cif_dir) / f"{protein_id}.cif"

    if not cif_path.exists():
        log.warning(f"  {protein_id} no cif file found")
        return None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parser    = MMCIFParser(QUIET=True)
            structure = parser.get_structure(protein_id, str(cif_path))

        # calculate sasa for rasa values
        sr = ShrakeRupley()
        sr.compute(structure, level="R")

        # locate the target lysine and its backbone atoms
        target_residue = None
        lys_ca         = None
        lys_n          = None
        lys_c          = None

        for model in structure:
            for chain in model:
                for residue in chain:
                    if (residue.get_id()[1] == position and
                            residue.get_resname() == "LYS"):
                        target_residue = residue
                        if "CA" in residue:
                            lys_ca = residue["CA"].get_coord()
                        if "N" in residue:
                            lys_n = residue["N"].get_coord()
                        if "C" in residue:
                            lys_c = residue["C"].get_coord()
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

        # need all three backbone atoms to build local frame
        if any(coord is None for coord in [lys_ca, lys_n, lys_c]):
            log.warning(
                f"  {protein_id} position {position} - "
                f"lysine missing backbone atoms for local frame"
            )
            return None

        # build local coordinate frame from lysine backbone
        local_frame = build_local_frame(lys_n, lys_ca, lys_c)

        # collect all sequence-distant residues with ca atoms
        candidates = []

        for model in structure:
            for chain in model:
                for residue in chain:

                    resname  = residue.get_resname()
                    res_pos  = residue.get_id()[1]

                    # skip non-standard residues
                    if resname not in MAX_ASA:
                        continue

                    # skip the lysine itself
                    if residue == target_residue:
                        continue

                    # skip sequence neighbours - already in sequence features
                    if abs(res_pos - position) <= sequence_separation:
                        continue

                    # skip residues with no ca atom
                    if "CA" not in residue:
                        continue

                    ca_coord = residue["CA"].get_coord()
                    distance = np.linalg.norm(lys_ca - ca_coord)

                    # calculate rasa for this residue
                    rasa = min(residue.sasa / MAX_ASA[resname], 1.0)

                    # determine if nearest contact is backbone or sidechain
                    # backbone atoms are N, CA, C, O
                    backbone_atoms  = {"N", "CA", "C", "O"}
                    min_dist        = float("inf")
                    is_backbone     = True

                    for atom in residue:
                        atom_dist = np.linalg.norm(
                            lys_ca - atom.get_coord()
                        )
                        if atom_dist < min_dist:
                            min_dist    = atom_dist
                            is_backbone = atom.get_name() in backbone_atoms

                    candidates.append({
                        "residue"    : residue,
                        "resname"    : resname,
                        "aa_1letter" : AA_3TO1.get(resname, "X"),
                        "distance"   : distance,
                        "ca_coord"   : ca_coord,
                        "rasa"       : round(rasa, 4),
                        "is_backbone": int(is_backbone),
                    })

        # sort by distance - nearest first
        candidates.sort(key=lambda x: x["distance"])

        # take the N nearest sequence-distant neighbours
        neighbours = candidates[:n_neighbours]

        # build feature dict
        features = {
            "protein_id"      : protein_id,
            "lysine_position" : position,
        }

        for i, nb in enumerate(neighbours, 1):
            vector = nb["ca_coord"] - lys_ca
            dist, phi, theta = cartesian_to_spherical(vector, local_frame)

            features[f"nb{i}_aa"]          = nb["aa_1letter"]
            features[f"nb{i}_distance"]    = dist
            features[f"nb{i}_phi"]         = phi
            features[f"nb{i}_theta"]       = theta
            features[f"nb{i}_rasa"]        = nb["rasa"]
            features[f"nb{i}_is_backbone"] = nb["is_backbone"]

        # pad with nulls if fewer than n_neighbours found
        # this can happen for very small proteins
        for i in range(len(neighbours) + 1, n_neighbours + 1):
            features[f"nb{i}_aa"]          = None
            features[f"nb{i}_distance"]    = None
            features[f"nb{i}_phi"]         = None
            features[f"nb{i}_theta"]       = None
            features[f"nb{i}_rasa"]        = None
            features[f"nb{i}_is_backbone"] = None

        return features

    except Exception as e:
        log.warning(
            f"  {protein_id} position {position} "
            f"spatial calculation failed: {e}"
        )
        return None


# ------------------------------------------------------------
# step 2 | run for all sites in the dataframe
# ------------------------------------------------------------

def add_spatial_features(
    sites_df,
    cif_dir,
    n_neighbours=N_NEIGHBOURS,
    sequence_separation=SEQUENCE_SEPARATION
):
    """
    Add spatial neighbourhood features to a dataframe of lysine sites.

    follows the standard feature module interface -
    takes a dataframe in, returns it with new columns added.

    params:
        sites_df            : dataframe with protein_id, lysine_position
        cif_dir             : directory containing alphafold cif files
        n_neighbours        : number of nearest neighbours to record
        sequence_separation : minimum sequence distance for a spatial
                              neighbour - avoids redundancy with
                              sequence window features
    
    returns sites_df with spatial feature columns added
    """
    cif_dir = Path(cif_dir)
    total   = len(sites_df)

    print(f"\n  calculating spatial features for {total} sites")
    print(f"  {n_neighbours} neighbours per site, "
          f"sequence separation >= {sequence_separation}")

    spatial_rows = []
    failed       = 0

    for i, (_, row) in enumerate(sites_df.iterrows(), 1):

        if i % 10 == 0 or i == total:
            print(f"  {i}/{total} sites processed")

        pid    = row["protein_id"]
        pos    = int(row["lysine_position"])
        result = calc_spatial_for_site(
            pid, pos, cif_dir, n_neighbours, sequence_separation
        )

        if result is None:
            failed += 1
            # add null row so merge stays clean
            null_row = {"protein_id": pid, "lysine_position": pos}
            for j in range(1, n_neighbours + 1):
                null_row[f"nb{j}_aa"]          = None
                null_row[f"nb{j}_distance"]    = None
                null_row[f"nb{j}_phi"]         = None
                null_row[f"nb{j}_theta"]       = None
                null_row[f"nb{j}_rasa"]        = None
                null_row[f"nb{j}_is_backbone"] = None
            spatial_rows.append(null_row)
        else:
            spatial_rows.append(result)

    if failed > 0:
        print(f"  {failed} sites failed spatial calculation - set to null")

    spatial_df = pd.DataFrame(spatial_rows)

    result = sites_df.merge(
        spatial_df,
        on=["protein_id", "lysine_position"],
        how="left"
    )

    n_features = len(spatial_df.columns) - 2
    print(f"  {n_features} spatial features added")
    print(f"  feature matrix shape: {result.shape}")

    return result