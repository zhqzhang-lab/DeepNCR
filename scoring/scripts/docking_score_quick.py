#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import torch
import joblib

from scoring_function_new import ScoringFunction
from parse_ligand import Ligand
from parse_receptor import Receptor
from model import DeepRMSD


# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ============================================
# 1) Model & Scaler Paths
# ============================================
MODEL_PATH = "../../retrain/Model_ROOT/multivariable2_rmsd_ratio_buchong_2.pth"
FEAT_SCALER_PATH = "../../retrain/feat_scaler/feat_scaler_multivariable2_buchong_2.pkl"
LABEL_SCALER_PATH = "../../retrain/label_scaler/label_scalers_multivariable2_rmsd_ratio_buchong_2.pkl"

# Fix for torch serialization safety issues in newer versions
torch.serialization.add_safe_globals([DeepRMSD])

print("⚡ Loading DeepRMSD model (Global)...")
GLOBAL_MODEL = torch.load(MODEL_PATH, map_location=device, weights_only=False)
GLOBAL_MODEL.eval()

print("⚡ Loading scalers (Global)...")
GLOBAL_FEAT_SCALER = joblib.load(FEAT_SCALER_PATH)
GLOBAL_LABEL_SCALER = joblib.load(LABEL_SCALER_PATH)


# ============================================
# 2) Dataset Configuration
# ============================================
# TODO: Update these paths to your actual data directories before running
protein_folder = "/path/to/CASF-2016/coreset_pdbqt/protein_py_sx"
decoy_folder   = "/path/to/CASF-2016/decoys_docking_pdbqt_py_2"

output_folder  = "../../scoring/csv_results_other/1204_rmsd_ratio_buchong_2"

os.makedirs(output_folder, exist_ok=True)

protein_files = os.listdir(protein_folder)
decoy_files   = os.listdir(decoy_folder)


# ============================================
# 3) Scoring Function
# ============================================
def score_one_target(protein_pdbqt, decoy_pdbqt, out_csv):

    print(f"\n==============================")
    print(f"▶ Processing {os.path.basename(protein_pdbqt)}")
    print("==============================")

    # Parse ligand
    ligand = Ligand(poses_file=decoy_pdbqt)
    ligand.parse_ligand()

    # Parse receptor
    receptor = Receptor(receptor_fpath=protein_pdbqt)
    receptor.parse_receptor()

    # Initialize ScoringFunction
    scoring = ScoringFunction(
        receptor=receptor,
        ligand=ligand,
        model_cached=GLOBAL_MODEL,
        feat_scaler_cached=GLOBAL_FEAT_SCALER,
        label_scaler_cached=GLOBAL_LABEL_SCALER
    )

    # Generate features and calculate scores
    scoring.generate_pldist_mtrx()
    scoring.cal_RMSD()
    scoring.cal_vina()

    # Extract multi-variable outputs
    ratio_6_int = scoring.pred_rmsd[:, 0].reshape(-1, 1)
    rmsd        = scoring.pred_rmsd[:, 1].reshape(-1, 1)

    inter_vina = scoring.vina_inter_energy.cpu().numpy()
    
    # Combined scoring strategy
    rmsd_vina  = 0.5 * rmsd + 0.5 * inter_vina

    # Prepare data for saving
    value = np.c_[rmsd, inter_vina, rmsd_vina, ratio_6_int]
    value = np.round(value, 5)

    df = pd.DataFrame(
        value, 
        index=ligand.poses_file_names,
        columns=["pred_rmsd", "inter_vina", "rmsd_vina", "ratio_6_int"]
    )

    # Sort by the combined score
    df = df.sort_values("rmsd_vina", ascending=True)
    df.to_csv(out_csv)

    print(f"✔ Saved result to -> {out_csv}")


# ============================================
# 4) Main Loop: Iterate over all targets
# ============================================
if __name__ == "__main__":
    for protein_file in protein_files:

        protein_id = protein_file.split("_")[0]
        decoy_file = f"{protein_id}_decoys.pdbqt"

        if decoy_file not in decoy_files:
            print(f"⚠ Warning: No decoys found for {protein_id}, skipping...")
            continue

        rec_path  = os.path.join(protein_folder, protein_file)
        dec_path  = os.path.join(decoy_folder,  decoy_file)
        out_csv   = os.path.join(output_folder, f"{protein_id}_docking_score.csv")

        score_one_target(rec_path, dec_path, out_csv)