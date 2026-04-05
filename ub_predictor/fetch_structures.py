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
# step 2 | download both files for one protein
# ------------------------------------------------------------

def fetch_files(protein_id, out_dir, timeout=30):
    """
    Download cif and pae files for one protein.

    params:
        protein_id : uniprot accession e.g. 'P04637'
        out_dir    : Path to save files into
        timeout    : seconds before download gives up

    returns one of: 'downloaded', 'exists', 'failed'
    """
    out_dir  = Path(out_dir)
    cif_path = out_dir / f"{protein_id}.cif"
    pae_path = out_dir / f"{protein_id}_pae.json"

    # both already on disk - nothing to do
    if cif_path.exists() and pae_path.exists():
        log.debug(f"  {protein_id} both files already on disk, skipping")
        return "exists"

    # get download urls from api
    urls = get_urls(protein_id)

    if urls is None:
        return "failed"

    # download each file - track if either fails
    success = True

    for file_path, url, label in [
        (cif_path, urls["cif"], "cif"),
        (pae_path, urls["pae"], "pae"),
    ]:
        # skip if this particular file already exists
        if file_path.exists():
            log.debug(f"  {protein_id} {label} already on disk, skipping")
            continue

        try:
            response = requests.get(url, timeout=timeout)

            if response.status_code == 200:
                file_path.write_bytes(response.content)
                log.debug(f"  {protein_id} {label} downloaded")

            else:
                log.warning(
                    f"  {protein_id} {label} download returned "
                    f"status {response.status_code}"
                )
                success = False

        except requests.exceptions.Timeout:
            log.warning(f"  {protein_id} {label} download timed out")
            success = False

        except requests.exceptions.RequestException as e:
            log.warning(f"  {protein_id} {label} download failed: {e}")
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