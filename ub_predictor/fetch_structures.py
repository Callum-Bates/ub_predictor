# ------------------------------------------------------------
# fetch_structures.py
#
# downloads alphafold cif structure files for a list of
# uniprot protein ids using the alphafold api.
#

# the api is used rather than direct url construction because
# alphafold db now versioned per-entry. the api always returns
# the correct current url regardless of version good!
# we will always use canonical isoform - important to note this when processing 
# mass spec data 
# input : list of uniprot ids (strings)
# output: cif files saved to data/structures/
#         returns dict summarising what downloaded and  what failed
# ------------------------------------------------------------

import requests
import time
import logging
from pathlib import Path

log = logging.getLogger(__name__)

ALPHAFOLD_API  = "https://alphafold.ebi.ac.uk/api/prediction/{protein_id}"


# ------------------------------------------------------------
# step 1 | query the api for the cif download url
# ------------------------------------------------------------

def get_cif_url(protein_id, timeout=15):
    """
    Ask the alphafold api for the cif download url for one protein.

    params:
        protein_id : uniprot accession e.g. 'P04637'
        timeout    : seconds before request gives up

    returns cifUrl string, or None if not found
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

        # api returns list of entries - first is always the canonical isoform
        # isoforms like P04637-2, P04637-3 etc come after
        # we will always use canonical isoform - important to note this when processing 
        # mass spec data 
        canonical = entries[0]
        cif_url   = canonical.get("cifUrl")

        if not cif_url:
            log.warning(f"  {protein_id} api entry had no cifUrl field")
            return None

        return cif_url

    except requests.exceptions.Timeout:
        log.warning(f"  {protein_id} api request timed out")
        return None

    except requests.exceptions.RequestException as e:
        log.warning(f"  {protein_id} api request failed: {e}")
        return None


# ------------------------------------------------------------
# step 2 | download the cif file from the url
# ------------------------------------------------------------

def fetch_cif(protein_id, out_dir, timeout=30):
    """
    Download the cif file for one protein.

    params:
        protein_id : uniprot accession e.g. 'P04637'
        out_dir    : Path to save the cif file into
        timeout    : seconds before download gives up

    returns one of: 'downloaded', 'exists', 'failed'
    """
    out_path = Path(out_dir) / f"{protein_id}.cif"

    # already on disk — nothing to do
    if out_path.exists():
        log.debug(f"  {protein_id} already on disk, skipping")
        return "exists"

    # get the download url from the api
    cif_url = get_cif_url(protein_id)

    if cif_url is None:
        return "failed"

    # download the file
    try:
        response = requests.get(cif_url, timeout=timeout)

        if response.status_code == 200:
            out_path.write_bytes(response.content)
            log.debug(f"  {protein_id} downloaded from {cif_url}")
            return "downloaded"
        else:
            log.warning(
                f"  {protein_id} cif download returned "
                f"status {response.status_code}"
            )
            return "failed"

    except requests.exceptions.Timeout:
        log.warning(f"  {protein_id} cif download timed out")
        return "failed"

    except requests.exceptions.RequestException as e:
        log.warning(f"  {protein_id} cif download failed: {e}")
        return "failed"


# ------------------------------------------------------------
# step 3 | run for a full list of proteins
# ------------------------------------------------------------

def fetch_all(protein_ids, out_dir, pause=0.2):
    """
    Download cif files for a list of proteins.

    params:
        protein_ids : list of uniprot accession strings
        out_dir     : directory to save cif files
        pause       : seconds to wait between requests
                      alphafold api asks for polite usage
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {"downloaded": [], "exists": [], "failed": []}
    total   = len(protein_ids)

    print(f"\n  fetching structures for {total} proteins")
    print(f"  saving to {out_dir}\n")

    for i, protein_id in enumerate(protein_ids, 1):

        if i % 10 == 0 or i == total:
            print(f"  {i}/{total} processed")

        status = fetch_cif(protein_id, out_dir)
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