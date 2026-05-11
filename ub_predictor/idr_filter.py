# ------------------------------------------------------------
# idr_filter.py
#
# classifies each lysine site as structured, disordered, or
# lacking structural data entirely.
#
# lysines in intrinsically disordered regions (idrs) are
# excluded from feature calculation. these are flexible regions
# that do not adopt a stable 3d fold - any structural features
# calculated there would not reflect real biology.
#
# disorder is predicted per-residue using metapredict, which
# fetches the protein sequence from uniprot and returns a
# score between 0 (ordered) and 1 (disordered) for each
# residue. scores >= 0.3 are classified as disordered.
#
# input : sites csv     (protein_id, lysine_position)
#         structures dir containing downloaded cif files
# output: sites_structured.csv   - goes forward to feature calculation
#         sites_disordered.csv   - idr sites, reported to user
#         sites_no_structure.csv - no alphafold entry available
# ------------------------------------------------------------

import pandas as pd
import logging
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
from pathlib import Path

import metapredict as meta


log = logging.getLogger(__name__)


# ------------------------------------------------------------
# step 1 | load the user's lysine sites
# ------------------------------------------------------------

def load_sites(sites_path):
    """
    Load lysine sites from user input csv.

    params:
        sites_path : path to csv with protein_id, lysine_position columns
    """
    sites_path = Path(sites_path)

    if not sites_path.exists():
        raise FileNotFoundError(
            f"input file not found: {sites_path}"
        )

    df = pd.read_csv(sites_path)

    required = {"protein_id", "lysine_position"}
    missing  = required - set(df.columns)

    if missing:
        raise ValueError(
            f"input file missing required columns: {missing}\n"
            f"expected columns: protein_id, lysine_position"
        )

    df["lysine_position"] = pd.to_numeric(
        df["lysine_position"], errors="coerce"
    ).astype("Int64")

    invalid = df["lysine_position"].isna().sum()
    if invalid > 0:
        log.warning(f"  {invalid} rows had non-numeric lysine_position - removed")
        df = df.dropna(subset=["lysine_position"])

    print(f"  {len(df)} lysine sites loaded from {sites_path.name}")
    return df


# ------------------------------------------------------------
# step 2 | predict disorder scores for each protein
# ------------------------------------------------------------

def build_idr_table(protein_ids, structures_dir, disorder_threshold=0.3):
    """
    Predict per-residue disorder scores using metapredict.

    fetches protein sequences from uniprot via protein id and
    returns a disorder score per residue. residues scoring
    >= 0.3 are classified as disordered (IDR = 1).
    
    Added cutoff as a parameter - users can change.

    only processes proteins that have a cif file - no point
    classifying disorder for proteins we have no structure for.

    params:
        protein_ids    : list of uniprot ids
        structures_dir : directory containing .cif files
    """
    structures_dir = Path(structures_dir)
    records        = []

    print(f"  predicting disorder for {len(protein_ids)} proteins")

    for protein_id in protein_ids:

        if not (structures_dir / f"{protein_id}.cif").exists():
            log.warning(
                f"  {protein_id} has no cif file - skipping disorder prediction"
            )
            continue

        try:
            scores = meta.predict_disorder_uniprot(protein_id)

            for position, score in enumerate(scores, start=1):
                records.append({
                    "protein_id"     : protein_id,
                    "position"       : position,
                    "disorder_score" : round(float(score), 4),
                    "IDR" : 1 if score >= disorder_threshold else 0
                })

        except Exception as e:
            log.warning(f"  {protein_id} disorder prediction failed: {e}")
            continue

    if not records:
        log.warning("  no disorder scores generated")
        return pd.DataFrame()

    idr_df = pd.DataFrame(records)

    n_disordered = idr_df["IDR"].sum()
    n_total      = len(idr_df)
    print(f"  {n_disordered} of {n_total} residues predicted as disordered")

    return idr_df


# ------------------------------------------------------------
# step 3 | classify each lysine site
# ------------------------------------------------------------

def classify_sites(sites_df, idr_table, structures_dir):
    """
    Look up each lysine site in the idr table and classify it.

    three outcomes per site:
        structured   - ordered region, goes forward to feature calculation
        disordered   - idr region, excluded from feature calculation
        no_structure - no alphafold cif available for this protein

    params:
        sites_df       : dataframe of lysine sites
        idr_table      : per-residue disorder scores from build_idr_table
        structures_dir : used to check which proteins have cif files
    """
    structures_dir = Path(structures_dir)

    if idr_table.empty:
        sites_df         = sites_df.copy()
        sites_df["region_type"] = "no_structure"
        return sites_df

    # build lookup dict keyed by (protein_id, position) for fast access
    # searching a dict is much faster than filtering a dataframe in a loop
    idr_lookup = {
        (row["protein_id"], int(row["position"])): int(row["IDR"])
        for _, row in idr_table.iterrows()
    }

    proteins_with_structure = set(idr_table["protein_id"].unique())
    region_types            = []

    for _, row in sites_df.iterrows():
        pid = row["protein_id"]
        pos = int(row["lysine_position"])

        if pid not in proteins_with_structure:
            region_types.append("no_structure")

        elif (pid, pos) not in idr_lookup:
            # protein exists but position not in table -
            # likely out of range for that protein sequence
            log.warning(
                f"  {pid} position {pos} not found in disorder table - "
                f"marking as no_structure"
            )
            region_types.append("no_structure")

        elif idr_lookup[(pid, pos)] == 1:
            region_types.append("disordered")

        else:
            region_types.append("structured")

    sites_df                = sites_df.copy()
    sites_df["region_type"] = region_types

    return sites_df


# ------------------------------------------------------------
# step 4 | split into three files and save
# ------------------------------------------------------------

def split_and_save(classified_df, out_dir):
    """
    Split classified sites into three output files.

    params:
        classified_df : sites dataframe with region_type column
        out_dir       : directory to write output files into
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    structured   = classified_df[classified_df["region_type"] == "structured"]
    disordered   = classified_df[classified_df["region_type"] == "disordered"]
    no_structure = classified_df[classified_df["region_type"] == "no_structure"]

    structured.to_csv(out_dir / "sites_structured.csv",   index=False)
    disordered.to_csv(out_dir / "sites_disordered.csv",   index=False)
    no_structure.to_csv(out_dir / "sites_no_structure.csv", index=False)

    total = len(classified_df)
    print(f"\n  {len(structured)} structured   -> sites_structured.csv")
    print(f"  {len(disordered)} disordered   -> sites_disordered.csv")
    print(f"  {len(no_structure)} no structure -> sites_no_structure.csv")
    print(
        f"\n  {len(structured)/total*100:.1f}% of sites moving forward "
        f"to feature calculation"
    )

    if len(disordered) > 0:
        print(
            f"\n  {len(disordered)} sites excluded - these lysines fall in "
            f"intrinsically disordered regions where alphafold coordinates "
            f"are unreliable. see sites_disordered.csv for details."
        )

    if len(no_structure) > 0:
        print(
            f"\n  {len(no_structure)} sites had no alphafold structure. "
            f"see sites_no_structure.csv for details."
        )

    return structured, disordered, no_structure


# ------------------------------------------------------------
# main entry point
# ------------------------------------------------------------

def run(sites_path, structures_dir, out_dir):
    """
    Run the full idr filter pipeline.

    params:
        sites_path     : path to user input csv
        structures_dir : directory containing cif files
        out_dir        : directory to write output files into
    """
    print("\n  -- idr filter --\n")

    sites_df    = load_sites(sites_path)
    protein_ids = sites_df["protein_id"].unique().tolist()

    print(f"  {len(protein_ids)} unique proteins to process\n")

    idr_table     = build_idr_table(protein_ids, structures_dir)
    classified_df = classify_sites(sites_df, idr_table, structures_dir)

    structured, disordered, no_structure = split_and_save(
        classified_df, out_dir
    )

    return structured, disordered, no_structure