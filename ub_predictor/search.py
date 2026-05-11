# ------------------------------------------------------------
# search.py - structural similarity search for lysine sites
# ------------------------------------------------------------
#
# given a reference ubiquitination site (e.g. K572 on Miro1),
# scans lysines across target proteins and ranks them by
# structural/chemical similarity using gower distance.
#
# the idea: if K572 is ubiquitinated and has a specific 3D
# neighbourhood (P553, Q555, D568 nearby in space), then other
# lysines with similar neighbourhoods might also be targets.
#
# uses spatial neighbour features (10 nearest neighbours -
# identity, distance, phi, theta, rasa, is_backbone) plus
# lysine rasa features for the comparison. gower distance
# handles the mix of categorical and numeric features.
#
# input : reference site (protein_id, lysine_position)
#         targets csv (list of protein_ids to scan)
# output: ranked csv of candidate sites with gower distances
# ------------------------------------------------------------

import pandas as pd
import numpy as np
import logging
import time
from pathlib import Path

from ub_predictor.features.sequence import fetch_sequences
from ub_predictor.features.rasa import add_rasa_features
from ub_predictor.features.spatial import add_spatial_features

log = logging.getLogger(__name__)


# ------------------------------------------------------------
# feature columns used for similarity comparison
# ------------------------------------------------------------

# spatial neighbour features - 10 neighbours x 6 features
SPATIAL_COLS = []
for i in range(1, 11):
    SPATIAL_COLS.extend([
        f"nb{i}_aa",
        f"nb{i}_distance",
        f"nb{i}_phi",
        f"nb{i}_theta",
        f"nb{i}_rasa",
        f"nb{i}_is_backbone",
    ])

# rasa features for the lysine itself
RASA_COLS = [
    "rasa_lysine",
    "rasa_sphere_mean",
    "rasa_sphere_std",
]

# all columns used in the gower distance calculation
SEARCH_COLS = SPATIAL_COLS + RASA_COLS

# which of these are categorical (for gower distance)
CATEGORICAL_COLS = set()
for i in range(1, 11):
    CATEGORICAL_COLS.add(f"nb{i}_aa")
    CATEGORICAL_COLS.add(f"nb{i}_is_backbone")


# ------------------------------------------------------------
# find all lysine positions in a protein sequence
# ------------------------------------------------------------

def find_lysines(sequence):
    """
    Return 1-indexed positions of all lysine residues in a sequence.

    params:
        sequence : amino acid sequence string

    returns list of integer positions
    """
    return [
        i + 1
        for i, aa in enumerate(sequence)
        if aa == "K"
    ]


# ------------------------------------------------------------
# build sites dataframe for all lysines in target proteins
# ------------------------------------------------------------

def build_target_sites(target_ids):
    """
    Fetch sequences for target proteins and build a sites
    dataframe with one row per lysine.

    params:
        target_ids : list of uniprot protein ids

    returns dataframe with protein_id and lysine_position columns
    """
    sequences = fetch_sequences(target_ids)

    rows = []
    for protein_id in target_ids:
        seq = sequences.get(protein_id)
        if seq is None:
            log.warning(f"  no sequence for {protein_id} - skipping")
            continue

        positions = find_lysines(seq)
        if not positions:
            log.info(f"  {protein_id} has no lysines - skipping")
            continue

        for pos in positions:
            rows.append({
                "protein_id": protein_id,
                "lysine_position": pos,
            })

    df = pd.DataFrame(rows)
    print(f"  {len(df)} lysines found across {df['protein_id'].nunique()} proteins")

    return df


# ------------------------------------------------------------
# generate search features (spatial + rasa only)
# ------------------------------------------------------------

def generate_search_features(sites_df, cif_dir):
    """
    Run rasa and spatial feature generation on a sites dataframe.
    skips sequence and structure features - not needed for search.

    params:
        sites_df : dataframe with protein_id, lysine_position
        cif_dir  : path to directory containing cif files

    returns dataframe with search feature columns added
    """
    print(f"\n  -- rasa features --")
    df = add_rasa_features(sites_df, cif_dir=cif_dir)

    print(f"\n  -- spatial features --")
    df = add_spatial_features(df, cif_dir=cif_dir)

    return df


# ------------------------------------------------------------
# gower distance
# ------------------------------------------------------------

def gower_distance(ref_row, candidates_df, feature_cols,
                   categorical_cols):
    """
    Calculate gower distance between a reference site and all
    candidate sites.

    gower distance handles mixed data types - for numeric features
    it uses normalised absolute difference, for categorical features
    it uses simple match/mismatch (0 or 1).

    the final distance is the mean across all features, ranging
    from 0 (identical) to 1 (completely different).

    params:
        ref_row          : series with feature values for reference
        candidates_df    : dataframe of candidate sites
        feature_cols     : list of column names to compare
        categorical_cols : set of column names that are categorical

    returns numpy array of distances, one per candidate
    """
    n_candidates = len(candidates_df)
    n_features = len(feature_cols)

    # accumulate partial distances and valid counts per candidate
    distances = np.zeros(n_candidates)
    valid_counts = np.zeros(n_candidates)

    for col in feature_cols:
        ref_val = ref_row[col]

        # skip if reference value is missing
        if pd.isna(ref_val):
            continue

        cand_vals = candidates_df[col].values

        # mask for candidates that also have this feature
        valid = ~pd.isna(cand_vals)

        if col in categorical_cols:
            # categorical - 0 if match, 1 if mismatch
            partial = np.where(cand_vals == ref_val, 0.0, 1.0)
        else:
            # numeric - normalised absolute difference
            # range across all values (reference + candidates)
            all_vals = np.concatenate([[ref_val], cand_vals[valid]])
            val_range = np.ptp(all_vals)

            if val_range == 0:
                partial = np.zeros(n_candidates)
            else:
                partial = np.abs(cand_vals - ref_val) / val_range

        # only count features where both ref and candidate are valid
        distances += np.where(valid, partial, 0.0)
        valid_counts += valid.astype(float)

    # mean across valid features
    with np.errstate(divide="ignore", invalid="ignore"):
        gower = np.where(
            valid_counts > 0,
            distances / valid_counts,
            np.nan,
        )

    return gower


# ------------------------------------------------------------
# format top neighbours for output
# ------------------------------------------------------------

def format_top_neighbours(row, n_top=3):
    """
    Build a human-readable string of the top spatial neighbours
    for a site, e.g. "P at 9.5a, Q at 9.3a, D at 8.1a".

    params:
        row   : series with spatial feature columns
        n_top : number of neighbours to include

    returns formatted string
    """
    parts = []
    for i in range(1, n_top + 1):
        aa = row.get(f"nb{i}_aa")
        dist = row.get(f"nb{i}_distance")

        if pd.isna(aa) or pd.isna(dist):
            continue

        parts.append(f"{aa} at {dist:.1f}a")

    return ", ".join(parts) if parts else "none"


# ------------------------------------------------------------
# run search
# ------------------------------------------------------------

def run_search(
    ref_protein_id,
    ref_position,
    targets_path,
    structures_dir,
    output_dir,
    n_results=None,
):
    """
    Run structural similarity search.

    generates features for a reference ubiquitination site and
    all lysines in the target proteins, then ranks candidates
    by gower distance to the reference.

    params:
        ref_protein_id : uniprot id of the reference protein
        ref_position   : lysine position in the reference protein
        targets_path   : path to csv with protein_id column
        structures_dir : directory containing alphafold cif files
        output_dir     : directory to write results
        n_results      : max results to return (None = all)

    returns results dataframe
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  -- structural similarity search --")
    print(f"  reference: {ref_protein_id} K{ref_position}")
    
    t_start = time.time()


    # --------------------------------------------------------
    # load target protein list
    # --------------------------------------------------------

    targets_df = pd.read_csv(targets_path)
    target_ids = targets_df["protein_id"].unique().tolist()

    # include the reference protein if not already in targets
    if ref_protein_id not in target_ids:
        target_ids.append(ref_protein_id)

    print(f"  {len(target_ids)} target proteins")

    # --------------------------------------------------------
    # build sites for all lysines in targets + reference
    # --------------------------------------------------------

    print(f"\n  -- finding lysines in target proteins --")
    all_sites = build_target_sites(target_ids)

    # add the reference site if not already present
    ref_mask = (
        (all_sites["protein_id"] == ref_protein_id) &
        (all_sites["lysine_position"] == ref_position)
    )
    if not ref_mask.any():
        ref_row = pd.DataFrame([{
            "protein_id": ref_protein_id,
            "lysine_position": ref_position,
        }])
        all_sites = pd.concat([ref_row, all_sites], ignore_index=True)

    # --------------------------------------------------------
    # generate features for all sites
    # --------------------------------------------------------

    print(f"\n  -- generating features for {len(all_sites)} lysines --")
    featured = generate_search_features(all_sites, cif_dir=structures_dir)

    # --------------------------------------------------------
    # extract reference and candidates
    # --------------------------------------------------------

    ref_mask = (
        (featured["protein_id"] == ref_protein_id) &
        (featured["lysine_position"] == ref_position)
    )

    if not ref_mask.any():
        raise ValueError(
            f"reference site {ref_protein_id} K{ref_position} "
            f"not found in feature matrix - check protein id "
            f"and position are correct"
        )

    ref_features = featured.loc[ref_mask].iloc[0]
    candidates = featured.loc[~ref_mask].copy()

    if candidates.empty:
        print("  no candidate sites to compare")
        return pd.DataFrame()

    # --------------------------------------------------------
    # calculate gower distance
    # --------------------------------------------------------

    print(f"\n  -- calculating gower distances --")

    candidates["gower_distance"] = gower_distance(
        ref_row=ref_features,
        candidates_df=candidates,
        feature_cols=SEARCH_COLS,
        categorical_cols=CATEGORICAL_COLS,
    )

    # --------------------------------------------------------
    # build results table
    # --------------------------------------------------------

    candidates["top_neighbours"] = candidates.apply(
        format_top_neighbours, axis=1
    )

    results = candidates[[
        "protein_id",
        "lysine_position",
        "gower_distance",
        "top_neighbours",
    ]].copy()

    results = results.sort_values("gower_distance").reset_index(drop=True)

    # drop rows where distance could not be calculated
    results = results.dropna(subset=["gower_distance"])

    if n_results is not None:
        results = results.head(n_results)

    # --------------------------------------------------------
    # save results
    # --------------------------------------------------------

    out_path = output_dir / (
        f"search_{ref_protein_id}_K{ref_position}.csv"
    )
    results.to_csv(out_path, index=False)

    # save full feature matrix with readable column names
    # allows inspect the structural values for each candidate
    from ub_predictor.rename_features import rename_columns
    features_path = output_dir / (
        f"search_{ref_protein_id}_K{ref_position}_features.csv"
    )
    rename_columns(featured).to_csv(features_path, index=False)

    print(f"\n  -- search complete --")
    print(f"  {len(results)} candidates ranked")
    print(f"  results saved to {out_path}")

    if len(results) > 0:
        print(f"\n  top 10 hits:")
        print(f"  {'protein':<12} {'lysine':<8} {'distance':<10} {'neighbours'}")
        print(f"  {'-'*60}")
        for _, row in results.head(10).iterrows():
            print(
                f"  {row['protein_id']:<12} "
                f"K{row['lysine_position']:<7} "
                f"{row['gower_distance']:<10.4f} "
                f"{row['top_neighbours']}"
            )

    elapsed = time.time() - t_start
    print(f"\n  total time: {elapsed:.1f}s")

    return results