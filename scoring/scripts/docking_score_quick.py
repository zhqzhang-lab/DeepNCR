```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
import pandas as pd
import torch
import joblib

from scoring_function_new import ScoringFunction
from parse_ligand import Ligand
from parse_receptor import Receptor
from model import DeepRMSD


START = int(sys.argv[1])
END   = int(sys.argv[2])


# ============================================
# 1) Load model once (no gradient)
# ============================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "../../retrain/Model_ROOT/multivariable2_rmsd_ratio_best.pth"
FEAT_SCALER_PATH = "../../retrain/feat_scaler/feat_scaler.pkl"
LABEL_SCALER_PATH = "../../retrain/label_scaler/label_scalers.pkl"

torch.serialization.add_safe_globals([DeepRMSD])

print("Loading DeepNCR model...")
GLOBAL_MODEL = torch.load(MODEL_PATH, map_location=device, weights_only=False)
GLOBAL_MODEL.eval()

GLOBAL_FEAT_SCALER = joblib.load(FEAT_SCALER_PATH)
GLOBAL_LABEL_SCALER = joblib.load(LABEL_SCALER_PATH)


# ============================================
# 2) Paths (modifiable)
# ============================================
POSEBUSTERS_ROOT = "./sample"

SAVE_DIR = "./sample/result"
os.makedirs(SAVE_DIR, exist_ok=True)


# ============================================
# 3) Score one PoseBusters complex
# ============================================
def score_complex(complex_id):

    complex_dir = os.path.join(POSEBUSTERS_ROOT, complex_id)

    protein_path = os.path.join(
        complex_dir, f"{complex_id}_protein.pdbqt"
    )
    ligand_path = os.path.join(
        complex_dir, f"{complex_id}_decoys.pdbqt"
    )

    out_path = os.path.join(
        SAVE_DIR, f"{complex_id}_scores.csv"
    )

    # skip if already processed
    if os.path.exists(out_path):
        print(f"Skip {complex_id}, already scored.")
        return

    if not (os.path.exists(protein_path) and os.path.exists(ligand_path)):
        print(f"Missing files for {complex_id}")
        return

    print("\n=======================================")
    print(f"Processing PoseBusters complex: {complex_id}")
    print("=======================================\n")

    # receptor
    receptor = Receptor(receptor_fpath=protein_path)
    try:
        receptor.parse_receptor()
    except Exception as e:
        print(f"Skip {complex_id}, receptor parsing failed: {e}")
        return

    # ligand (decoys)
    ligand = Ligand(poses_file=ligand_path)
    try:
        ligand.parse_ligand()
    except Exception as e:
        print(f"Cannot parse ligand poses for {complex_id}: {e}")
        return

    scoring = ScoringFunction(
        receptor=receptor,
        ligand=ligand,
        model_cached=GLOBAL_MODEL,
        feat_scaler_cached=GLOBAL_FEAT_SCALER,
        label_scaler_cached=GLOBAL_LABEL_SCALER
    )

    # inference (no gradient)
    with torch.no_grad():
        scoring.generate_pldist_mtrx()
        scoring.cal_RMSD()
        scoring.cal_vina()

        ratio_6_int = scoring.pred_rmsd[:, 0].reshape(-1, 1)
        rmsd        = scoring.pred_rmsd[:, 1].reshape(-1, 1)
        inter_vina  = scoring.vina_inter_energy.cpu().numpy().reshape(-1, 1)
        rmsd_vina   = 0.5 * rmsd + 0.5 * inter_vina
        ratio_vina  = 0.5 * inter_vina - 3.5 * ratio_6_int

    # collect results
    rows = []
    N = rmsd.shape[0]
    for i in range(N):
        pose_id = f"{complex_id}_{(i+1)}"
        rows.append([
            pose_id,
            float(rmsd[i][0]),
            float(inter_vina[i][0]),
            float(rmsd_vina[i][0]),
            float(ratio_6_int[i][0]),
            float(ratio_vina[i][0])
        ])

    df = pd.DataFrame(
        rows,
        columns=[
            "pose",
            "pred_rmsd",
            "inter_vina",
            "rmsd_vina",
            "ratio_6_int",
            "ratio_vina"
        ]
    )
    df = df.sort_values("ratio_vina", ascending=True)

    df.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")


# ============================================
# 4) Main (supports slicing)
# ============================================
def main():
    complexes = sorted(os.listdir(POSEBUSTERS_ROOT))
    complexes = complexes[START:END]

    total = len(complexes)
    print(f"Will process complexes[{START}:{END}], total = {total}")

    for idx, complex_id in enumerate(complexes, start=1):
        complex_dir = os.path.join(POSEBUSTERS_ROOT, complex_id)
        if not os.path.isdir(complex_dir):
            continue

        print(f"\n===== [{idx}/{total}] {complex_id} =====")
        score_complex(complex_id)


if __name__ == "__main__":
    main()
```
