import pandas as pd
import re
raw_path = "/home/callum/projects/ub_predictor/data/raw"

output_path = "/home/callum/projects/ub_predictor/data/processed"

# Read the Excel file from the U2OS sheet
input_df = pd.read_excel(f"{raw_path}/34740826.xlsx", sheet_name='U2OS')  

print(f"Input file has {len(input_df)} rows")
print("Sample of data:")
print(input_df[['Peptide_sequence', 'Modifications_(peptide)', 'Protein_ACCs', 'Peptide_beginning_indices']].head())

def handle_proteins_and_isoforms_for_peptides(df):
    """
    Handle protein isoforms and multi-protein mappings for peptide data
    Returns: (processed_dataframe, bad_rows_indices)
    """
    bad_rows = []
    
    for index, row in df.iterrows():
        if index % 50000 == 0:
            print(f"Processing protein filtering for row {index}/{len(df)}")
            
        protein_accs = str(row['Protein_ACCs'])
        peptide_starts = str(row['Peptide_beginning_indices'])
        
        # Skip if data is missing
        if pd.isna(protein_accs) or pd.isna(peptide_starts) or protein_accs == 'nan' or peptide_starts == 'nan':
            bad_rows.append(index)
            continue
        
        # Handle multiple proteins
        protein_acc_list = protein_accs.split(';')
        peptide_start_list = peptide_starts.split(';')
        
        if len(protein_acc_list) > 1:
            # Extract base protein names (remove isoform suffixes like -1, -2, etc.)
            base_proteins = []
            for protein in protein_acc_list:
                base_protein = protein.split('-')[0].strip()
                base_proteins.append(base_protein)
            
            # Check if all proteins are isoforms of the same base protein
            unique_base_proteins = set(base_proteins)
            
            if len(unique_base_proteins) == 1:
                # All are isoforms of the same protein - use canonical (base) protein
                canonical_protein = list(unique_base_proteins)[0]
                df.at[index, 'Protein_ACCs'] = canonical_protein
                # Use the first start position for the canonical protein
                df.at[index, 'Peptide_beginning_indices'] = peptide_start_list[0]
            else:
                # Multiple different proteins - discard this row
                bad_rows.append(index)
    
    return df, bad_rows

# Apply protein filtering
print("Filtering proteins and handling isoforms...")
processed_df, bad_rows = handle_proteins_and_isoforms_for_peptides(input_df)
clean_df = processed_df.drop(bad_rows)

print(f"Rows after protein/isoform handling: {len(clean_df)}")
print(f"Discarded {len(bad_rows)} rows due to multi-protein mapping")

def extract_k_sites_from_peptide_data(df):
    """
    Extract K sites from peptide data, handling GlyGly modifications
    """
    all_k_sites = []
    
    for index, row in df.iterrows():
        if index % 10000 == 0:
            print(f"Processing K sites for row {index}/{len(df)}")
        
        peptide_seq = str(row['Peptide_sequence'])
        modifications = str(row['Modifications_(peptide)'])
        protein_acc = str(row['Protein_ACCs'])
        peptide_start_str = str(row['Peptide_beginning_indices'])
        
        # Skip if essential data is missing
        if pd.isna(peptide_start_str) or pd.isna(protein_acc) or peptide_start_str == 'nan' or protein_acc == 'nan':
            continue
        
        try:
            peptide_start = int(peptide_start_str)
        except ValueError:
            continue  # Skip if start position is not a valid integer
        
        # Find all K positions in the peptide sequence
        k_positions_in_peptide = []
        for i, aa in enumerate(peptide_seq):
            if aa == 'K':
                k_positions_in_peptide.append(i + 1)  # 1-based indexing
        
        if not k_positions_in_peptide:
            continue  # No K residues in this peptide
        
        # Check for GlyGly modifications
        glyGly_modifications = {}
        if modifications != '-' and modifications != 'nan' and pd.notna(modifications):
            # Parse modifications like "GG(K8)" or "GG(K7)"
            glyGly_pattern = r'GG\(K(\d+)\)'
            matches = re.findall(glyGly_pattern, modifications)
            for match in matches:
                k_pos_in_peptide = int(match)
                glyGly_modifications[k_pos_in_peptide] = True
        
        # Process each K position
        for k_pos_in_peptide in k_positions_in_peptide:
            # Calculate protein position
            protein_position = peptide_start + k_pos_in_peptide - 1
            
            # Determine if this K is ubiquitinated
            is_ubiquitinated = k_pos_in_peptide in glyGly_modifications
            
            all_k_sites.append({
                'protein_id': protein_acc,
                'AA': 'K',
                'position': protein_position,
                'ub': 1 if is_ubiquitinated else 0,
                'peptide_sequence': peptide_seq,
                'modifications': modifications,
                'k_pos_in_peptide': k_pos_in_peptide,
                'peptide_start': peptide_start
            })
    
    return pd.DataFrame(all_k_sites)

# Extract all K sites
print("Extracting K sites...")
k_sites_df = extract_k_sites_from_peptide_data(clean_df)

print(f"\nExtracted {len(k_sites_df)} K sites before deduplication")
print(f"Ubiquitinated sites: {len(k_sites_df[k_sites_df['ub'] == 1])}")
print(f"Non-ubiquitinated sites: {len(k_sites_df[k_sites_df['ub'] == 0])}")

# Handle duplicates - prioritize ub=1 over ub=0 for same protein-position
def deduplicate_with_ub_priority(df):
    """
    Remove duplicates, prioritizing ub=1 over ub=0 for the same protein-position
    """
    # Sort by protein_id, position, and ub (descending so ub=1 comes first)
    df_sorted = df.sort_values(['protein_id', 'position', 'ub'], ascending=[True, True, False])
    
    # Keep first occurrence of each protein-position pair
    df_dedup = df_sorted.drop_duplicates(subset=['protein_id', 'position'], keep='first')
    
    return df_dedup

# Deduplicate
print("Deduplicating...")
k_sites_clean = deduplicate_with_ub_priority(k_sites_df)

print(f"\nAfter deduplication: {len(k_sites_clean)} unique K sites")
print(f"Ubiquitinated sites: {len(k_sites_clean[k_sites_clean['ub'] == 1])}")
print(f"Non-ubiquitinated sites: {len(k_sites_clean[k_sites_clean['ub'] == 0])}")

# Create final output in standard format
output_df = pd.DataFrame({
    'Unnamed: 0': range(len(k_sites_clean)),
    'protein_id': k_sites_clean['protein_id'],
    'AA': k_sites_clean['AA'],
    'position': k_sites_clean['position'].astype(float),
    'ac': 0,
    'ac_reg': 0,
    'ga': 0,
    'gl': 0,
    'gl_reg': 0,
    'm': 0,
    'm_reg': 0,
    'p': 0,
    'p_reg': 0,
    'sm': 0,
    'sm_reg': 0,
    'ub': k_sites_clean['ub'],
    'ub_reg': 0,
    'dataset': '34740826'
})

# Save the output
output_df.to_csv(f"{output_path}/full_test", sep='\t', index=False)

print(f"\nFinal output saved as 'for_annotation_34740826.tsv'")
print(f"Total sites: {len(output_df)}")
print(f"Unique proteins: {output_df['protein_id'].nunique()}")
print(f"Position range: {output_df['position'].min()} - {output_df['position'].max()}")

# Show some examples
print(f"\nFirst few rows:")
print(output_df.head())

# Show some statistics
print(f"\nDataset statistics:")
print(f"Ubiquitinated sites (ub=1): {len(output_df[output_df['ub'] == 1])}")
print(f"Non-ubiquitinated sites (ub=0): {len(output_df[output_df['ub'] == 0])}")

# Save detailed version for inspection
detailed_output = k_sites_clean[['protein_id', 'position', 'ub', 'peptide_sequence', 'modifications', 'k_pos_in_peptide']].copy()
detailed_output.to_csv('detailed_34740826.tsv', sep='\t', index=False)
print(f"Detailed output saved as 'detailed_34740826.tsv'")

# Show some examples of GlyGly modifications found
if len(k_sites_clean[k_sites_clean['ub'] == 1]) > 0:
    glyGly_examples = k_sites_clean[k_sites_clean['ub'] == 1][['protein_id', 'position', 'peptide_sequence', 'modifications']].head(10)
    print(f"\nExamples of GlyGly modifications found:")
    print(glyGly_examples)
else:
    print(f"\nNo GlyGly modifications found in the data")