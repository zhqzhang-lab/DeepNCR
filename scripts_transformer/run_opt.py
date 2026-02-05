#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import joblib
import numpy as np
import argparse

# Local modules
from model import DeepRMSD
from conformation_to_xyz import LigandConformation
from parse_receptor import Receptor
from scoring_function import ScoringFunction
from utils import save_data, save_results, save_final_lig_cnfr, output_ligand_traj, local_set_lr

# ======================================================
# 1) Global Configuration & Resources
# ======================================================
# TODO: Update these paths to your actual model/scaler locations
MODEL_PATH = "retrain/Model_ROOT/multivariable2_rmsd_ratio_buchong_2.pth"
FEAT_SCALER_PATH = "retrain/feat_scaler/feat_scaler_multivariable2_buchong_2.pkl"
LABEL_SCALER_PATH = "retrain/label_scaler/label_scalers_multivariable2_rmsd_ratio_buchong_2.pkl"

# Global cache to prevent reloading large files
GLOBAL_MODEL = None
GLOBAL_FEAT_SCALER = None
GLOBAL_LABEL_SCALER = None


def load_global_resources(device):
    """
    Loads the DeepRMSD model and scalers into global memory only once.
    """
    global GLOBAL_MODEL, GLOBAL_FEAT_SCALER, GLOBAL_LABEL_SCALER

    if GLOBAL_MODEL is None:
        print("🔥 Loading global DeepRMSD model...")
        # Load to CPU first, then move to specified device
        GLOBAL_MODEL = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        GLOBAL_MODEL = GLOBAL_MODEL.to(device)
        GLOBAL_MODEL.eval()

    if GLOBAL_FEAT_SCALER is None:
        print("🔥 Loading feature scaler...")
        GLOBAL_FEAT_SCALER = joblib.load(FEAT_SCALER_PATH)

    if GLOBAL_LABEL_SCALER is None:
        print("🔥 Loading label scaler...")
        GLOBAL_LABEL_SCALER = joblib.load(LABEL_SCALER_PATH)


# ======================================================
# 2) Optimization Engine
# ======================================================
class Optimize:
    """
    Handles the gradient-based optimization of ligand conformations.
    """
    def __init__(self, receptor, ligand, output_path, epochs=150, delta=0.001, torsion_param=2.0, device="cuda:0"):
        self.receptor = receptor
        self.ligand = ligand
        self.output_path = output_path
        self.epochs = epochs
        self.delta = delta
        self.torsion_param = torsion_param
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Ensure output directory exists
        os.makedirs(output_path, exist_ok=True)

        # Initialize Global Resources
        load_global_resources(self.device)
        print("✔ Global model and scalers are ready.")

        # Initialize Conformation Vector (Requires Gradient)
        self.cnfr = ligand.init_cnfr
        self.scores_data = []
        self.optimal_scores = None  # Stores the best RMSD_Vina scores after step 70

    def cal_energy(self):
        """
        Wraps the ScoringFunction process to compute gradients.
        """
        score = ScoringFunction(
            receptor=self.receptor,
            ligand=self.ligand,
            model_cached=GLOBAL_MODEL,
            feat_scaler_cached=GLOBAL_FEAT_SCALER,
            label_scaler_cached=GLOBAL_LABEL_SCALER
        )
        
        # Returns: pred_rmsd, vina, rmsd_vina, combined_score, ratio_vina, ratio
        return score.process()

    def run(self):
        """
        Executes the optimization loop.
        """
        print(f"🚀 Starting optimization for {self.epochs} epochs...")
        
        for step in range(self.epochs):
            if step % 10 == 0:
                print(f"====== [Step {step}/{self.epochs}] ======")

            # 1. Update Coordinates from Conformation Vector
            self.ligand.cnfr2xyz(self.cnfr)

            # 2. Calculate Energies
            # Note: combined_score is the target for backpropagation
            pred_rmsd, vina_score, rmsd_vina, combined_score, ratio_vina, ratio = self.cal_energy()

            # 3. Save Trajectory (Optional: can add condition to save less frequently)
            output_ligand_traj(self.output_path, self.ligand)

            # 4. Log Data
            # Detach tensors to save numpy values
            _rows = np.c_[
                vina_score.detach().cpu().numpy(),
                pred_rmsd.detach().cpu().numpy(),
                rmsd_vina.detach().cpu().numpy(),
                ratio_vina.detach().cpu().numpy(),
                ratio.detach().cpu().numpy()
            ]
            self.scores_data.append(_rows)
            save_data(self.scores_data, self.output_path, self.ligand)

            # 5. Track Optimal Structure (Convergence Check)
            # Strategy: Start tracking best structures after step 70
            rmsd_vina_flat = rmsd_vina.reshape(-1).detach().cpu()
            
            if step == 70:
                self.optimal_scores = rmsd_vina_flat.clone()
                cond = torch.ones_like(self.optimal_scores)
                
                # Save initial baseline
                save_results(cond, self.output_path, self.ligand, self.scores_data)
                save_final_lig_cnfr(cond, self.output_path, self.ligand)

            elif step > 70:
                # Update if new score is better by margin delta
                cond = (rmsd_vina_flat <= self.optimal_scores - self.delta).float()
                
                # Update optimal scores where condition is met
                self.optimal_scores = cond * rmsd_vina_flat + (1 - cond) * self.optimal_scores
                
                # Save updated best structures
                save_results(cond, self.output_path, self.ligand, self.scores_data)
                save_final_lig_cnfr(cond, self.output_path, self.ligand)

            # ==========================================================
            # 6. Gradient Descent Step
            # ==========================================================
            # Backward pass on the differentiable combined score
            combined_score.backward(torch.ones_like(combined_score), retain_graph=True)

            # Get gradients
            grad = self.cnfr.grad
            
            # Gradient Clipping / Normalization (Sigmoid squash)
            # This prevents exploding gradients during molecular dynamics
            grad = 2 * torch.sigmoid(grad) - 1 

            # Dynamic Learning Rate
            lr = local_set_lr(step, self.torsion_param, self.ligand.number_of_frames)

            # Update Conformation Vector
            # Note: We manually update to avoid optimizer overhead for this specific physics task
            self.cnfr.data = self.cnfr.data - lr * grad

            # Zero gradients for next step
            self.cnfr.grad.zero_()


# ======================================================
# 3) Entry Point
# ======================================================
def run_optimize(receptor_pdb, poses_dir, output_dir):
    print(f"📌 Loading receptor: {receptor_pdb}")
    receptor = Receptor(receptor_pdb)
    receptor.parse_receptor()

    print(f"📌 Loading ligand poses: {poses_dir}")
    # Assumes LigandConformation handles the parsing logic
    ligand = LigandConformation(poses_dir)

    print(f"📌 Output directory: {output_dir}")
    optimizer = Optimize(receptor, ligand, output_dir)
    optimizer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepRMSD Conformation Optimization")

    parser.add_argument("--receptor", type=str, required=True, help="Path to receptor PDB/PDBQT file")
    parser.add_argument("--poses", type=str, required=True, help="Directory or file containing initial ligand poses")
    parser.add_argument("--output", type=str, required=True, help="Directory to save optimization results")

    args = parser.parse_args()

    run_optimize(args.receptor, args.poses, args.output)