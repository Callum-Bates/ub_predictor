
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
