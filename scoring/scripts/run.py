import os
import argparse
import numpy as np
import pandas as pd
from argparse import RawDescriptionHelpFormatter

# Local modules
from argparse_utils import add_args
from parse_receptor import Receptor
from parse_ligand import Ligand
from scoring_function_new import ScoringFunction

def perform_scoring(ligand: Ligand, 
                    receptor: Receptor, 
                    mean_std_file: str, 
                    model_fpath: str,
                    out_fpath: str):
    """
    Executes the scoring workflow and saves the results to CSV.
    """
    
    # Initialize Scoring Function
    # Note: Ensure ScoringFunction handles model loading internally using model_fpath
    scoring = ScoringFunction(
        receptor=receptor, 
        ligand=ligand, 
        mean_std_file=mean_std_file, 
        model_fpath=model_fpath
    )
    
    # 1. Feature Generation & Calculation
    scoring.generate_pldist_mtrx()
    scoring.cal_RMSD()
    scoring.cal_vina()
    
    # 2. Extract Predictions (Vectorized)
    # Assumes scoring.pred_rmsd is a numpy array of shape (N, 2)
    # Column 0: ratio_6_int, Column 1: RMSD
    ratio_6_int = scoring.pred_rmsd[:, 0].reshape(-1, 1)
    rmsd_score  = scoring.pred_rmsd[:, 1].reshape(-1, 1)

    # 3. Extract Vina Score
    # Handle PyTorch tensor conversion safely (CPU/GPU)
    if hasattr(scoring.vina_inter_energy, 'detach'):
        inter_vina = scoring.vina_inter_energy.cpu().detach().numpy()
    else:
        inter_vina = scoring.vina_inter_energy
    
    # 4. Compute Combined Score
    # Hybrid scoring: 50% DeepRMSD + 50% Vina Inter Energy
    rmsd_vina = 0.5 * rmsd_score + 0.5 * inter_vina

    # 5. Format Data for Output
    # Concatenate all metrics horizontally
    data_block = np.hstack([rmsd_score, inter_vina, rmsd_vina, ratio_6_int])
    data_block = np.round(data_block, 5)
    
    # Create DataFrame
    df = pd.DataFrame(
        data_block, 
        index=ligand.poses_file_names, 
        columns=["pred_rmsd", "inter_vina", "rmsd_vina", "ratio_6_int"]
    )
    
    # Sort by the combined score (Ascending: lower is better)
    df = df.sort_values(by="rmsd_vina", ascending=True)
    
    # 6. Save to CSV
    # Ensure output directory exists
    output_dir = os.path.dirname(out_fpath)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    df.to_csv(out_fpath)
    print(f"Successfully saved scoring results to: {out_fpath}")


if __name__ == "__main__":
    description = """
    Score protein-ligand binding poses using DeepRMSD + Vina.
    """
    parser = argparse.ArgumentParser(description=description, formatter_class=RawDescriptionHelpFormatter)
    
    # Load arguments from utils
    add_args(parser)
    args = parser.parse_args()

    # Parse Molecule Files
    print(f"Processing Target: {args.target}")
    
    ligand = Ligand(poses_file=args.pose_fpath)
    ligand.parse_ligand()

    receptor = Receptor(receptor_fpath=args.rec_fpath)
    receptor.parse_receptor()

    # Execute Scoring
    perform_scoring(
        ligand=ligand,
        receptor=receptor,
        mean_std_file=args.mean_std_file,
        model_fpath=args.model,
        out_fpath=args.out_fpath
    )