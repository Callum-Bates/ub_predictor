# ------------------------------------------------------------
# rename_features.py
#
# converts internal feature column names to human-readable
# biological labels before saving output files.
#
# called automatically when saving predictions.csv and
# features_complete.csv - readable names here
# without any extra steps.
# ------------------------------------------------------------

import re

# ------------------------------------------------------------
# amino acid and secondary structure lookup tables
# ------------------------------------------------------------

AA_ONE_TO_THREE = {
    "A": "Ala", "C": "Cys", "D": "Asp", "E": "Glu",
    "F": "Phe", "G": "Gly", "H": "His", "I": "Ile",
    "K": "Lys", "L": "Leu", "M": "Met", "N": "Asn",
    "P": "Pro", "Q": "Gln", "R": "Arg", "S": "Ser",
    "T": "Thr", "V": "Val", "W": "Trp", "Y": "Tyr",
    "X": "unknown",
}

SS_CODES = {
    "H": "helix",
    "E": "sheet",
    "T": "turn",
    "C": "coil",
}

# ------------------------------------------------------------
# static renames - fixed column names
# ------------------------------------------------------------

STATIC_RENAMES = {

    # identity
    "protein_id"               : "uniprot_id",
    "lysine_position"          : "lysine_position",
    "region_type"              : "region_type",
    "ub"                       : "ubiquitinated",

    # prediction output
    "ub_probability"           : "ubiquitination_probability",
    "predicted_ub"             : "predicted_ubiquitinated",
    "prediction"               : "prediction",

    # position
    "position_fraction"        : "position_in_protein_fraction",
    "distance_n_terminus"      : "distance_from_n_terminus_residues",
    "distance_c_terminus"      : "distance_from_c_terminus_residues",
    "sequence_length"          : "protein_length_residues",
    "window_entropy"           : "sequence_complexity_entropy",

    # chemical group counts
    "acidic_count"             : "acidic_residues_in_window",
    "basic_count"              : "basic_residues_in_window",
    "aromatic_count"           : "aromatic_residues_in_window",
    "hydrophobic_count"        : "hydrophobic_residues_in_window",
    "polar_count"              : "polar_residues_in_window",
    "small_count"              : "small_residues_in_window",
    "proline_count"            : "proline_residues_in_window",

    # chemical group ratios
    "acidic_ratio"             : "acidic_fraction_in_window",
    "basic_ratio"              : "basic_fraction_in_window",
    "aromatic_ratio"           : "aromatic_fraction_in_window",
    "hydrophobic_ratio"        : "hydrophobic_fraction_in_window",
    "polar_ratio"              : "polar_fraction_in_window",
    "small_ratio"              : "small_fraction_in_window",
    "proline_ratio"            : "proline_fraction_in_window",

    # rasa
    "rasa_lysine"              : "lysine_surface_exposure",
    "rasa_sphere_mean"         : "mean_surface_exposure_8A_neighbourhood",
    "rasa_sphere_std"          : "variation_surface_exposure_8A_neighbourhood",
    "n_sphere_residues"        : "residue_count_8A_neighbourhood",
    "sphere_radius_a"          : "neighbourhood_sphere_radius_angstroms",

    # secondary structure fractions
    "ss_window_helix_fraction" : "helix_fraction_in_window",
    "ss_window_sheet_fraction" : "sheet_fraction_in_window",
    "ss_window_turn_fraction"  : "turn_fraction_in_window",
    "ss_window_coil_fraction"  : "coil_fraction_in_window",
    "ss_lysine"                : "lysine_secondary_structure",

    # shap output
    "shap_feature_1"           : "top_feature_1",
    "shap_feature_2"           : "top_feature_2",
    "shap_feature_3"           : "top_feature_3",
    "shap_feature_4"           : "top_feature_4",
    "shap_feature_5"           : "top_feature_5",
    "shap_value_1"             : "top_feature_1_contribution",
    "shap_value_2"             : "top_feature_2_contribution",
    "shap_value_3"             : "top_feature_3_contribution",
    "shap_value_4"             : "top_feature_4_contribution",
    "shap_value_5"             : "top_feature_5_contribution",
    
    
    # new rasa features
    "rasa_sphere_median"       : "median_surface_exposure_8A_neighbourhood",
    "rasa_sphere_max"          : "max_surface_exposure_8A_neighbourhood",
    "mean_n_terminal_rasa"     : "mean_surface_exposure_n_terminal_half",
    "mean_c_terminal_rasa"     : "mean_surface_exposure_c_terminal_half",
    "plddt_lysine"             : "alphafold_confidence_score_lysine",

    # new sequence features
    "nearest_basic_distance"   : "distance_to_nearest_basic_residue_KRH",
    "nearest_acidic_distance"  : "distance_to_nearest_acidic_residue_DE",
    "nearest_proline_distance" : "distance_to_nearest_proline",
    "nearest_glycine_distance" : "distance_to_nearest_glycine",
    "net_charge_local"         : "net_charge_in_window",
    "charge_asymmetry"         : "charge_asymmetry_left_vs_right_of_lysine",
    "hydrophobic_hydrophilic_ratio": "hydrophobic_to_hydrophilic_ratio",
    "aromatic_aliphatic_ratio" : "aromatic_to_aliphatic_ratio",
    "acidic_basic_ratio"       : "acidic_to_basic_ratio",
}


# ------------------------------------------------------------
# dynamic renaming for patterned column names
# ------------------------------------------------------------

def _rename_col(col):
    """
    Convert one column name to a readable label.

    tries static lookup first, then pattern matching
    for sequence window, secondary structure, and
    spatial neighbour columns.

    params:
        col : raw column name string

    returns readable label string
    """
    # static lookup first
    if col in STATIC_RENAMES:
        return STATIC_RENAMES[col]

    # sequence window amino acid identity: aa_k-3
    m = re.match(r"^aa_k([+-]\d+)$", col)
    if m:
        return f"sequence_k{m.group(1)}_amino_acid"

    # sequence window one-hot encoded: aa_k-3_E
    m = re.match(r"^aa_k([+-]\d+)_([A-Z])$", col)
    if m:
        offset = m.group(1)
        aa     = AA_ONE_TO_THREE.get(m.group(2), m.group(2))
        return f"sequence_k{offset}_is_{aa}"

    # secondary structure per position: ss_k-2
    m = re.match(r"^ss_k([+-]\d+)$", col)
    if m:
        return f"secondary_structure_k{m.group(1)}"

    # secondary structure one-hot: ss_k-2_H
    m = re.match(r"^ss_k([+-]\d+)_([HETC])$", col)
    if m:
        offset  = m.group(1)
        ss_name = SS_CODES.get(m.group(2), m.group(2))
        return f"secondary_structure_k{offset}_is_{ss_name}"

    # lysine secondary structure one-hot: ss_lysine_H
    m = re.match(r"^ss_lysine_([HETC])$", col)
    if m:
        ss_name = SS_CODES.get(m.group(1), m.group(1))
        return f"lysine_secondary_structure_is_{ss_name}"

    # spatial neighbour amino acid: nb3_aa
    m = re.match(r"^nb(\d+)_aa$", col)
    if m:
        return f"spatial_neighbour_{m.group(1)}_amino_acid"

    # spatial neighbour amino acid one-hot: nb3_aa_E
    m = re.match(r"^nb(\d+)_aa_([A-Z])$", col)
    if m:
        n  = m.group(1)
        aa = AA_ONE_TO_THREE.get(m.group(2), m.group(2))
        return f"spatial_neighbour_{n}_is_{aa}"

    # spatial neighbour distance: nb3_distance
    m = re.match(r"^nb(\d+)_distance$", col)
    if m:
        return f"spatial_neighbour_{m.group(1)}_distance_angstroms"

    # spatial neighbour polar angle: nb3_phi
    m = re.match(r"^nb(\d+)_phi$", col)
    if m:
        return f"spatial_neighbour_{m.group(1)}_polar_angle_degrees"

    # spatial neighbour azimuthal angle: nb3_theta
    m = re.match(r"^nb(\d+)_theta$", col)
    if m:
        return f"spatial_neighbour_{m.group(1)}_azimuthal_angle_degrees"

    # spatial neighbour rasa: nb3_rasa
    m = re.match(r"^nb(\d+)_rasa$", col)
    if m:
        return f"spatial_neighbour_{m.group(1)}_surface_exposure"

    # spatial neighbour backbone flag: nb3_is_backbone
    m = re.match(r"^nb(\d+)_is_backbone$", col)
    if m:
        return f"spatial_neighbour_{m.group(1)}_contact_type"

    # spatial neighbour backbone one-hot: nb3_is_backbone_0 / _1
    m = re.match(r"^nb(\d+)_is_backbone_([01])$", col)
    if m:
        n       = m.group(1)
        contact = "backbone" if m.group(2) == "1" else "sidechain"
        return f"spatial_neighbour_{n}_is_{contact}_contact"

    # return original if no pattern matched
    return col


# ------------------------------------------------------------
# main public function
# ------------------------------------------------------------

def rename_columns(df):
    """
    Rename all columns in a dataframe to readable biological labels.

    safe to call on any dataframe - unrecognised columns are
    left unchanged.

    params:
        df : pandas dataframe with internal feature column names

    returns dataframe with renamed columns (copy - original unchanged)
    """
    rename_map = {col: _rename_col(col) for col in df.columns}
    return df.rename(columns=rename_map)