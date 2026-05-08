## about
This Ub predictor tool aims to allow wet lab researchers to explore physiochemical environments surrounding ubiquitination sites. 
Currently, this tool offers:  
**train mode** - where users can provide a set of lysine sites with known ubiquitination outcomes.  A gradient boosted machine learning model (XGBoost) is trained on this data.  Models are evaluated using 5-fold cross validation (with future plans to add validation on independent datasets), and highlight significant biological features.

<img width="1229" height="536" alt="image" src="https://github.com/user-attachments/assets/634ce930-eb5a-4bd9-8cb8-2f319c060c69" />

**search mode** - where users can query a specific lysine site in a specific protein.  This site is characterised using the same feature generation process used in *train mode*, and sites are queried against a user provided list of proteins.  Similarity between sites is measured by [Gower's distance](https://rajithkalinda.medium.com/understanding-gower-distance-for-mixed-data-types-in-machine-learning-e90ad42d5684), a similarity measure of 2 data points, capable of handling both numerical and categorical data.  Similarity scores per residue are ranked and reported in an output file with contributing features.

<img width="1231" height="287" alt="image" src="https://github.com/user-attachments/assets/c7a91cae-495f-4bc0-8834-fd95048dc4e7" />





## installation (MARS - first time only)

```bash
git clone git@github.com:Callum-Bates/ub_predictor.git
cd ub_predictor
module load apps/python3
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

edit `run_download.sh` and `run_pipeline.sh` to add your account:
```bash
#SBATCH --account=your_account_name
```

---

## updating (after first install)

```bash
cd ~/projects/ub_predictor
git pull
```

---

## input format

a csv file with these exact column names:

**train mode:**

protein_id,lysine_position,ub
Q8IXI2,572,1
Q8IXI2,194,0
- `protein_id` - uniprot accession
- `lysine_position` - position of the lysine in the sequence
- `ub` - 1 = ubiquitinated, 0 = not ubiquitinated

**search mode:**

two files needed.

reference site:  
protein_id,lysine_position  
Q8IXI2,572

target proteins to scan:  
protein_id  
P21796  
O95140  
Q16539

---

## running on MARS

place your input file in `data/raw/` then submit:

```bash
# train mode
bash submit_job.sh data/raw/my_sites.csv train

# monitor
squeue -u $USER
tail -f outputs/logs/pipeline_<jobid>.log
```

search mode runs directly (no slurm needed for small target lists):
```bash
source .venv/bin/activate

python predict.py \
    --mode search \
    --ref-protein Q8IXI2 \
    --ref-position 572 \
    --targets data/raw/target_proteins.csv \
    --structures data/structures \
    --verbose
```

---

## outputs

all outputs go to a timestamped folder in `outputs/`:

**train mode:**

outputs/20260508_141950_my_sites/
training_report.txt        model performance summary
roc_curve.png              roc curve
feature_importance.png     top features driving predictions
data/processed/20260508_141950_my_sites/
features_complete_readable.csv    full feature table, readable column names
sites_disordered.csv              sites excluded - disordered regions
sites_no_structure.csv            sites excluded - no alphafold entry

**search mode:**
outputs/my_output_folder/
search_Q8IXI2_K572.csv             ranked candidate sites
search_Q8IXI2_K572_features.csv    full feature table for candidates

### search results columns
- `protein_id` - candidate protein
- `lysine_position` - candidate lysine position
- `gower_distance` - similarity to reference site (0 = identical, 1 = completely different)
- `top_neighbours` - nearest 3d neighbours e.g. "P at 9.5a, Q at 9.3a"

---

## notes

- sites in disordered regions are automatically excluded and reported separately
- proteins not in the alphafold database are excluded and reported separately
