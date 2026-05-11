# ------------------------------------------------------------
# features/sequence.py
#
# calculates sequence-based features for each lysine site.
#
# for each lysine, a window of +-10 residues is extracted
# from the full protein sequence. features capture the local
# chemical environment around the lysine - the identity of
# neighbouring residues, their chemical properties, and the
# position of the lysine within the full protein chain.
#
# e3 ligases recognise ubiquitination substrates partly through
# local sequence context, so these features are biologically
# motivated even before any structural information is added.
#
# input : sites_structured.csv (protein_id, lysine_position)
# output: same dataframe with sequence feature columns added
# ------------------------------------------------------------

import pandas as pd
import numpy as np
import requests
import time
import logging
from pathlib import Path
from collections import Counter

log = logging.getLogger(__name__)

# window of residues either side of the lysine to consider
WINDOW_SIZE = 10

# standard amino acids in alphabetical order -
# used for consistent one-hot encoding of positional identity
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

# chemical group classifications -
# each residue belongs to one or more groups
CHEMICAL_GROUPS = {
    "acidic"      : set("DE"),
    "basic"       : set("KRH"),
    "aromatic"    : set("FWY"),
    "hydrophobic" : set("AILMVFWY"),
    "polar"       : set("STNQ"),
    "small"       : set("GASC"),
    "proline"     : set("P"),
}


# ------------------------------------------------------------
# step 1 | fetch protein sequences from uniprot
# ------------------------------------------------------------

def fetch_sequences(protein_ids, pause=0.1):
    """
    Fetch amino acid sequences from uniprot for a list of proteins.

    params:
        protein_ids : list of uniprot accession strings
        pause       : seconds between requests - be polite to uniprot
    
    returns dict of {protein_id: sequence_string}
    """
    sequences = {}
    total     = len(protein_ids)

    print(f"  fetching sequences for {total} proteins from uniprot")

    for i, protein_id in enumerate(protein_ids, 1):

        if i % 10 == 0 or i == total:
            print(f"  {i}/{total} sequences fetched")

        try:
            url      = f"https://rest.uniprot.org/uniprotkb/{protein_id}.fasta"
            response = requests.get(url, timeout=15)

            if response.status_code == 200:
                # fasta format - first line is header, rest is sequence
                lines    = response.text.strip().split("\n")
                sequence = "".join(lines[1:])
                sequences[protein_id] = sequence

            elif response.status_code == 404:
                log.warning(f"  {protein_id} not found in uniprot")

            else:
                log.warning(
                    f"  {protein_id} uniprot returned "
                    f"status {response.status_code}"
                )

        except requests.exceptions.Timeout:
            log.warning(f"  {protein_id} uniprot request timed out")

        except requests.exceptions.RequestException as e:
            log.warning(f"  {protein_id} uniprot request failed: {e}")

        time.sleep(pause)

    n_missing = total - len(sequences)
    if n_missing > 0:
        log.warning(f"  {n_missing} proteins had no sequence available")

    print(f"  {len(sequences)} sequences retrieved")
    return sequences


# ------------------------------------------------------------
# step 2 | extract window and calculate features
# ------------------------------------------------------------

def extract_window(sequence, position, window_size=WINDOW_SIZE):
    """
    Extract the local sequence window around a lysine.

    position is 1-based (as in uniprot) - converted to 0-based
    internally. window is padded with "X" at sequence edges.

    params:
        sequence    : full protein amino acid sequence string
        position    : 1-based position of the lysine
        window_size : number of residues either side to extract
    
    returns window string of length 2*window_size + 1
    """
    idx    = position - 1  # convert to 0-based index
    start  = idx - window_size
    end    = idx + window_size + 1

    # pad with X if window extends beyond sequence boundaries
    left_pad  = max(0, -start) * "X"
    right_pad = max(0, end - len(sequence)) * "X"

    clipped = sequence[max(0, start): min(len(sequence), end)]
    window  = left_pad + clipped + right_pad

    return window


def calc_shannon_entropy(sequence):
    """
    Calculate shannon entropy of a sequence window.

    low entropy = repetitive / low complexity sequence,
    which is associated with disordered regions.
    high entropy = diverse amino acid composition.

    params:
        sequence : amino acid string (the local window)
    """
    sequence = sequence.replace("X", "")
    if len(sequence) == 0:
        return 0.0

    counts = Counter(sequence)
    probs  = [c / len(sequence) for c in counts.values()]
    return round(-sum(p * np.log2(p) for p in probs), 4)


def calc_sequence_features(protein_id, position, sequence, window_size=WINDOW_SIZE):
    """
    Calculate all sequence features for one lysine site.

    params:
        protein_id  : uniprot accession
        position    : 1-based lysine position
        sequence    : full protein sequence string
        window_size : residues either side of lysine to consider
    
    returns dict of feature name -> value
    """
    window = extract_window(sequence, position, window_size)
    centre = window_size  # index of the lysine in the window

    features = {
        "protein_id"      : protein_id,
        "lysine_position" : position,
    }

    # -- positional features --
    # where in the protein is this lysine?
    # expressed as a fraction so it is comparable across proteins
    # of different lengths
    features["position_fraction"]    = round(position / len(sequence), 4)
    features["distance_n_terminus"]  = position
    features["distance_c_terminus"]  = len(sequence) - position
    features["sequence_length"]      = len(sequence)

    # -- window complexity --
    features["window_entropy"] = calc_shannon_entropy(window)

    # -- per-position amino acid identity --
    # what residue is at each position relative to the lysine?
    # encoded as the amino acid letter - will be one-hot encoded
    # later during model training
    for offset in range(-window_size, window_size + 1):
        if offset == 0:
            continue  # skip the lysine itself
        idx = centre + offset
        aa  = window[idx] if 0 <= idx < len(window) else "X"
        features[f"aa_k{offset:+d}"] = aa

    # -- chemical group counts in window --
    # how many of each chemical type surround this lysine?
    window_no_pad = window.replace("X", "")
    for group, residues in CHEMICAL_GROUPS.items():
        features[f"{group}_count"] = sum(
            1 for aa in window_no_pad if aa in residues
        )

    # -- chemical group ratios --
    # normalise by window length so edge lysines are comparable
    # to central lysines with full windows
    n_valid = len(window_no_pad)
    for group in CHEMICAL_GROUPS:
        count = features[f"{group}_count"]
        features[f"{group}_ratio"] = (
            round(count / n_valid, 4) if n_valid > 0 else 0.0
        )

    # -- nearest residue distances --
    # how many positions to the nearest residue of each type
    # measured as absolute sequence distance from the lysine
    # captures whether key residue types are close or far in sequence
    special_types = {
        "nearest_basic_distance"  : set("KRH"),
        "nearest_acidic_distance" : set("DE"),
        "nearest_proline_distance": set("P"),
        "nearest_glycine_distance": set("G"),
    }

    for feature_name, residue_set in special_types.items():
        nearest = None
        for offset in range(-window_size, window_size + 1):
            if offset == 0:
                continue
            idx = centre + offset
            if 0 <= idx < len(window):
                aa = window[idx]
                if aa in residue_set:
                    dist = abs(offset)
                    if nearest is None or dist < nearest:
                        nearest = dist
        # if not found in window, set to window_size + 1
        # indicates the nearest is beyond the window boundary
        features[feature_name] = nearest if nearest is not None else window_size + 1

    # -- charge-based features --
    # net_charge_local: sum of charges in window
    # positive residues (K, R, H) count as +1
    # negative residues (D, E) count as -1
    positive_charge = sum(
        1 for aa in window_no_pad if aa in set("KRH")
    )
    negative_charge = sum(
        1 for aa in window_no_pad if aa in set("DE")
    )

    features["net_charge_local"] = positive_charge - negative_charge

    # charge_asymmetry: difference in charge between
    # n-terminal half and c-terminal half of the window
    # captures whether charge is asymmetrically distributed
    # around the lysine
    left_half  = window[:centre].replace("X", "")
    right_half = window[centre + 1:].replace("X", "")

    left_charge = sum(
        1 if aa in set("KRH") else -1 if aa in set("DE") else 0
        for aa in left_half
    )
    right_charge = sum(
        1 if aa in set("KRH") else -1 if aa in set("DE") else 0
        for aa in right_half
    )

    features["charge_asymmetry"] = left_charge - right_charge

    # -- chemical ratio features --
    # hydrophobic_hydrophilic_ratio: hydrophobic / polar residues
    # aromatic_aliphatic_ratio: aromatic / aliphatic residues
    # acidic_basic_ratio: acidic / basic residues
    # small number added to denominator to avoid division by zero
    hydrophobic = sum(1 for aa in window_no_pad if aa in set("AILMVFWY"))
    hydrophilic  = sum(1 for aa in window_no_pad if aa in set("STNQDE"))
    aromatic     = sum(1 for aa in window_no_pad if aa in set("FWYH"))
    aliphatic    = sum(1 for aa in window_no_pad if aa in set("AILV"))
    acidic       = sum(1 for aa in window_no_pad if aa in set("DE"))
    basic        = sum(1 for aa in window_no_pad if aa in set("KRH"))

    features["hydrophobic_hydrophilic_ratio"] = round(
        hydrophobic / (hydrophilic + 0.001), 4
    )
    features["aromatic_aliphatic_ratio"] = round(
        aromatic / (aliphatic + 0.001), 4
    )
    features["acidic_basic_ratio"] = round(
        acidic / (basic + 0.001), 4
    )


    return features


# ------------------------------------------------------------
# step 3 | run for all sites in the dataframe
# ------------------------------------------------------------

def add_sequence_features(sites_df, sequences=None):
    """
    Add sequence features to a dataframe of lysine sites.

    follows the standard feature module interface -
    takes a dataframe in, returns it with new columns added.

    params:
        sites_df  : dataframe with protein_id, lysine_position columns
        sequences : optional dict of pre-fetched sequences -
                    if not provided, fetches from uniprot automatically
    
    returns sites_df with sequence feature columns added
    """
    if sequences is None:
        protein_ids = sites_df["protein_id"].unique().tolist()
        sequences   = fetch_sequences(protein_ids)

    print(f"\n  calculating sequence features for {len(sites_df)} sites")

    feature_rows = []
    skipped      = 0

    for _, row in sites_df.iterrows():
        pid = row["protein_id"]
        pos = int(row["lysine_position"])

        if pid not in sequences:
            log.warning(
                f"  {pid} has no sequence - skipping position {pos}"
            )
            skipped += 1
            continue

        seq = sequences[pid]

        # sanity check - is this actually a lysine at this position?
        idx = pos - 1
        if 0 <= idx < len(seq) and seq[idx] != "K":
            log.warning(
                f"  {pid} position {pos} is {seq[idx]}, not K - "
                f"check your input file"
            )

        features = calc_sequence_features(pid, pos, seq)
        feature_rows.append(features)

    if skipped > 0:
        print(f"  {skipped} sites skipped - no sequence available")

    features_df = pd.DataFrame(feature_rows)

    # merge back onto original dataframe to preserve all columns
    result = sites_df.merge(
        features_df,
        on=["protein_id", "lysine_position"],
        how="left"
    )

    n_features = len(features_df.columns) - 2  # subtract id columns
    print(f"  {n_features} sequence features added")
    print(f"  feature matrix shape: {result.shape}")

    return result