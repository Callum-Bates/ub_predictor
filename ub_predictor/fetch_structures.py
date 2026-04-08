# ------------------------------------------------------------
# fetch_structures.py
#
# downloads alphafold cif structure files and pae (predicted
# aligned error) files for a list of uniprot protein ids,
# using the alphafold api to get the correct download urls.
#
# cif files contain the 3d atomic coordinates - used for
# rasa and spatial neighbour calculations downstream.
#
# pae files contain alphafold's per-residue confidence map -
# used by structuremap to identify disordered regions (idrs).
# lysine sites in idrs are excluded from feature calculation
# because predicted coordinates there are unreliable.
#
# input : list of uniprot ids (strings)
# output: cif files  -> data/structures/
#         pae files  -> data/structures/
#         returns dict summarising what downloaded, what failed
# ------------------------------------------------------------

import requests
import time
import logging
from pathlib import Path
import json
import numpy as np
import h5py

log = logging.getLogger(__name__)

ALPHAFOLD_API = "https://alphafold.ebi.ac.uk/api/prediction/{protein_id}"


# ------------------------------------------------------------
# step 1 | query the api for download urls
# ------------------------------------------------------------

def get_urls(protein_id, timeout=15):
    """
    Query the alphafold api for cif and pae download urls.

    params:
        protein_id : uniprot accession e.g. 'P04637'
        timeout    : seconds before request gives up

    returns dict with keys 'cif' and 'pae', or None if not found
    """
    url = ALPHAFOLD_API.format(protein_id=protein_id)

    try:
        response = requests.get(url, timeout=timeout)

        if response.status_code == 404:
            log.debug(f"  {protein_id} not found in alphafold")
            return None

        if response.status_code != 200:
            log.warning(
                f"  {protein_id} api returned status {response.status_code}"
            )
            return None

        entries = response.json()

        if not entries:
            log.debug(f"  {protein_id} api returned empty list")
            return None

        # api returns a list - first entry is always the canonical isoform
        # isoforms like P04637-2, P04637-3 etc come after and are ignored here
        canonical = entries[0]

        cif_url = canonical.get("cifUrl")
        pae_url = canonical.get("paeDocUrl")

        if not cif_url:
            log.warning(f"  {protein_id} api entry had no cifUrl field")
            return None

        if not pae_url:
            log.warning(f"  {protein_id} api entry had no paeDocUrl field")
            return None

        return {"cif": cif_url, "pae": pae_url}

    except requests.exceptions.Timeout:
        log.warning(f"  {protein_id} api request timed out")
        return None

    except requests.exceptions.RequestException as e:
        log.warning(f"  {protein_id} api request failed: {e}")
        return None





# ------------------------------------------------------------
# step 2 | convert pae json to hdf5 format for structuremap
# ------------------------------------------------------------

def convert_pae_to_hdf(json_path, hdf_path):
    """
    Convert alphafold pae json file to hdf5 format.

    structuremap expects pae data as a flat 1d array in an
    hdf5 file named pae_{protein_id}.hdf. alphafold serves
    it as a nested list (n x n matrix) in json format.
    this function flattens and converts it.

    params:
        json_path : path to the downloaded _pae.json file
        hdf_path  : path to write the output .hdf file to
    """
    with open(json_path) as f:
        data = json.load(f)

    # extract the pae matrix and flatten to 1d
    # alphafold v3+ stores it under "predicted_aligned_error"
    pae_matrix = data[0]["predicted_aligned_error"]
    pae_flat   = [val for row in pae_matrix for val in row]

    with h5py.File(hdf_path, "w") as hdf:
        hdf.create_dataset(
            name="dist",
            data=pae_flat,
            compression="lzf",
            shuffle=True
        )

# ------------------------------------------------------------
# step 3 | download and convert files for one protein
# ------------------------------------------------------------

def fetch_files(protein_id, out_dir, timeout=30):
    """
    Download cif and pae files for one protein.
    pae json is converted to hdf5 after download - structuremap
    requires hdf5 format with a specific naming convention.

    params:
        protein_id : uniprot accession e.g. "P04637"
        out_dir    : Path to save files into
        timeout    : seconds before download gives up

    returns one of: "downloaded", "exists", "failed"
    """
    out_dir  = Path(out_dir)
    cif_path = out_dir / f"{protein_id}.cif"
    hdf_path = out_dir / f"pae_{protein_id}.hdf"

    # both already on disk - nothing to do
    if cif_path.exists() and hdf_path.exists():
        log.debug(f"  {protein_id} both files already on disk, skipping")
        return "exists"

    # get download urls from api
    urls = get_urls(protein_id)

    if urls is None:
        return "failed"

    success = True

    # download cif
    if not cif_path.exists():
        try:
            response = requests.get(urls["cif"], timeout=timeout)
            if response.status_code == 200:
                cif_path.write_bytes(response.content)
                log.debug(f"  {protein_id} cif downloaded")
            else:
                log.warning(
                    f"  {protein_id} cif download returned "
                    f"status {response.status_code}"
                )
                success = False
        except requests.exceptions.Timeout:
            log.warning(f"  {protein_id} cif download timed out")
            success = False
        except requests.exceptions.RequestException as e:
            log.warning(f"  {protein_id} cif download failed: {e}")
            success = False

    # download pae json and convert to hdf5
    if not hdf_path.exists():
        json_path = out_dir / f"{protein_id}_pae.json"
        try:
            response = requests.get(urls["pae"], timeout=timeout)
            if response.status_code == 200:
                json_path.write_bytes(response.content)
                log.debug(f"  {protein_id} pae json downloaded")

                # convert to hdf5 for structuremap
                convert_pae_to_hdf(json_path, hdf_path)
                log.debug(f"  {protein_id} pae converted to hdf5")

                # remove the json - we only need the hdf5
                json_path.unlink()
                
            else:
                log.warning(
                    f"  {protein_id} pae download returned "
                    f"status {response.status_code}"
                )
                success = False
        except requests.exceptions.Timeout:
            log.warning(f"  {protein_id} pae download timed out")
            success = False
        except requests.exceptions.RequestException as e:
            log.warning(f"  {protein_id} pae download failed: {e}")
            success = False

    return "downloaded" if success else "failed"



# ------------------------------------------------------------
# step 3 | run for a full list of proteins
# ------------------------------------------------------------

def fetch_all(protein_ids, out_dir, pause=0.2):
    """
    Download cif and pae files for a list of proteins.

    params:
        protein_ids : list of uniprot accession strings
        out_dir     : directory to save files into
        pause       : seconds to wait between requests -
                      alphafold api asks for polite usage
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {"downloaded": [], "exists": [], "failed": []}
    total   = len(protein_ids)

    print(f"\n  fetching structures and pae files for {total} proteins")
    print(f"  saving to {out_dir}\n")

    for i, protein_id in enumerate(protein_ids, 1):

        if i % 10 == 0 or i == total:
            print(f"  {i}/{total} processed")

        status = fetch_files(protein_id, out_dir)
        results[status].append(protein_id)

        time.sleep(pause)

    # summary
    print(f"\n  {len(results['downloaded'])} downloaded")
    print(f"  {len(results['exists'])} already on disk")
    print(f"  {len(results['failed'])} not found in alphafold")

    if results["failed"]:
        print(f"\n  proteins with no alphafold entry:")
        for p in results["failed"]:
            print(f"    {p}")

    return results