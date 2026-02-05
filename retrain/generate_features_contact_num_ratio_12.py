import os
import math
import argparse
import itertools
import numpy as np
import pandas as pd
import torch as th
from scipy.spatial.distance import cdist
from argparse import RawDescriptionHelpFormatter

# Assuming 'utils' is a local module you have. 
# Ensure utils.py is in the same directory.
from utils import (
    all_defined_residues, 
    all_rec_defined_ele, 
    ad4_to_ele_dict, 
    all_lig_ele, 
    get_elementtype
)

# ==========================================
# File Parsing Classes
# ==========================================

class ReceptorFile:
    """Parses receptor PDB/PDBQT files."""
    def __init__(self, rec_fpath: str):
        self.rec_fpath = rec_fpath
        self.rec_ha_types = []
        self.rec_ha_xyz = None
        self.rec_ha_num = []
        self._load_rec()

    def _load_rec(self):
        xyz_list = []
        with open(self.rec_fpath) as f:
            lines = [x for x in f.readlines() if x.startswith("ATOM")]

        for line in lines:
            ele = line.split()[-1]
            if ele in ["H", "HD"]:
                continue
            
            res = line[17:20].strip()
            res = res if res in all_defined_residues else "OTH"
            
            ele_type = ele if ele in all_rec_defined_ele else "DU"
            
            atom_num = line[6:11].strip()
            coords = [float(line[30:38]), float(line[38:46]), float(line[46:54])]

            self.rec_ha_num.append(atom_num)
            xyz_list.append(coords)
            self.rec_ha_types.append(f"{res}-{ele_type}")

        self.rec_ha_xyz = np.array(xyz_list, dtype=np.float32)


class LigandFile:
    """Parses native ligand files."""
    def __init__(self, lig_fpath: str):
        self.lig_fpath = lig_fpath
        self.lig_ha_num = []
        self.lig_ha_xyz = None
        self._load_lig()

    def _load_lig(self):
        xyz_list = []
        with open(self.lig_fpath, 'r') as f:
            for line in f:
                if line.startswith('ATOM') and line[77] != 'H':
                    self.lig_ha_num.append(line[6:11].strip())
                    coords = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                    xyz_list.append(coords)
        self.lig_ha_xyz = np.array(xyz_list, dtype=np.float32)


class DecoyFile:
    """Parses docking decoy (multi-model) files."""
    def __init__(self, lig_fpath: str):
        self.lig_fpath = lig_fpath
        self.lig_ha_ele = []
        self.all_pose_xyz = None  # Shape: [N_pose, N_ha, 3]
        self.all_pose_num = []
        self._parse_lig()

    def _parse_lig(self):
        with open(self.lig_fpath) as f:
            lines = f.readlines()

        all_xyz, current_xyz, current_ele, current_num = [], [], [], []
        is_first_model = True

        for line in lines:
            if line.startswith("ENDMDL"):
                all_xyz.append(np.array(current_xyz))
                self.all_pose_num.append(current_num)
                if is_first_model:
                    self.lig_ha_ele = list(current_ele)
                    is_first_model = False
                current_xyz, current_num = [], [] # Reset for next model
                continue

            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue

            ad4_type = line.split()[-1]
            if ad4_type in ["H", "HD"]:
                continue

            # Coordinate extraction
            coords = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            current_xyz.append(coords)
            current_num.append(line[6:11].strip())

            # Element extraction (only needed once really, but kept for logic flow)
            if is_first_model:
                ele = ad4_to_ele_dict.get(ad4_type, "DU")
                current_ele.append(get_elementtype(ele))

        self.all_pose_xyz = np.array(all_xyz, dtype=np.float32)


class PocketFile(ReceptorFile):
    """Parses receptor residues within a cutoff of a specific point."""
    def __init__(self, file_path, pocket_xyz, cutoff=35):
        self.pocket_xyz = pocket_xyz
        self.cutoff = cutoff
        self.rec_fpath = file_path # Inherited requirement
        self.rec_ha_types = []
        self.rec_ha_num = []
        self.rec_ha_xyz = None
        
        # Parse logic
        valid_lines = self._filter_residues(file_path)
        self._parse_lines(valid_lines)

    def _filter_residues(self, file_path):
        pocket_lines = []
        with open(file_path, 'r') as f:
            current_res_id = None
            buffer_lines = []
            keep_residue = False

            for line in f:
                if line.startswith(('ATOM', 'HETATM')):
                    res_id = line[22:26].strip()
                    xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                    
                    if res_id != current_res_id:
                        if keep_residue: pocket_lines.extend(buffer_lines)
                        current_res_id = res_id
                        buffer_lines = []
                        keep_residue = False
                    
                    buffer_lines.append(line)
                    # Check distance using CA or any atom
                    if not keep_residue and np.linalg.norm(xyz - self.pocket_xyz) < self.cutoff:
                        keep_residue = True
                
                elif line.startswith('TER') and keep_residue:
                    pocket_lines.extend(buffer_lines)
                    pocket_lines.append(line)
                    buffer_lines = []
            
            if keep_residue: pocket_lines.extend(buffer_lines)
        return pocket_lines

    def _parse_lines(self, lines):
        # Re-use similar logic to ReceptorFile but for specific lines
        xyz_list = []
        for line in lines:
            if not line.startswith("ATOM"): continue
            ele = line.split()[-1]
            if ele in ["H", "HD"]: continue
            
            res = line[17:20].strip()
            res = res if res in all_defined_residues else "OTH"
            ele = ele if ele in all_rec_defined_ele else "DU"
            
            self.rec_ha_num.append(line[6:11].strip())
            xyz_list.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            self.rec_ha_types.append(f"{res}-{ele}")
        
        self.rec_ha_xyz = np.array(xyz_list, dtype=np.float32)


# ==========================================
# Feature Generation
# ==========================================

class FeatureGenerator:
    def __init__(self, receptor: ReceptorFile, ligand: DecoyFile, pre_cut=0.3, cutoff=2.0):
        self.rec = receptor
        self.lig = ligand
        self.pre_cut = pre_cut
        self.cutoff = cutoff
        self.keys = self._get_defined_pairs()
        self.output_dim = 1470 # Fixed based on defined pairs

    def _get_defined_pairs(self):
        defined_res_pairs = [f"{r}-{a}" for r, a in itertools.product(all_defined_residues, all_rec_defined_ele)]
        keys = []
        for pair, lig_ele in itertools.product(defined_res_pairs, all_lig_ele):
            keys.extend([f"r6_{pair}_{lig_ele}", f"r1_{pair}_{lig_ele}"])
        return keys

    def compute(self):
        # Shape: [N_poses, N_lig_atoms, 3] vs [1, N_rec_atoms, 3]
        rec_xyz = self.rec.rec_ha_xyz[None, :, :] 
        lig_xyz = self.lig.all_pose_xyz
        
        # Calculate distance matrix using Torch for speed (GPU ready if moved)
        # Result: [N_poses, N_rec, N_lig] or similar. Let's strictly follow orig logic dim
        # Original: cdist(rec, lig) -> [N_rec, N_lig]. 
        # But here we have multiple poses. 
        # Using CDIST for batch:
        t_rec = th.from_numpy(self.rec.rec_ha_xyz).unsqueeze(0).repeat(lig_xyz.shape[0], 1, 1) # [Batch, N_rec, 3]
        t_lig = th.from_numpy(lig_xyz) # [Batch, N_lig, 3]
        
        dist = th.cdist(t_rec, t_lig).numpy() / 10.0 # [Batch, N_rec, N_lig] (in nm)
        
        # Vectorized Feature Calculation
        mask_1 = dist <= self.pre_cut
        dist_1 = np.where(mask_1, self.pre_cut, 0.0)
        
        mask_2 = (dist > self.pre_cut) & (dist < self.cutoff)
        dist_2 = dist * mask_2

        # Helper for power calculation to avoid div by zero
        def safe_pow(arr, p):
            return np.power(arr + (arr == 0.), p) - (arr == 0.)

        feat_matrix_1_r6 = safe_pow(dist_1, -6)
        feat_matrix_2_r6 = safe_pow(dist_2, -6)
        features_r6 = (feat_matrix_1_r6 + feat_matrix_2_r6).reshape(lig_xyz.shape[0], -1, 1)

        feat_matrix_1_r1 = safe_pow(dist_1, -1)
        feat_matrix_2_r1 = safe_pow(dist_2, -1)
        features_r1 = (feat_matrix_1_r1 + feat_matrix_2_r1).reshape(lig_xyz.shape[0], -1, 1)
        
        # Concatenate: [Batch, N_interactions, 2] -> flatten last 2 dims -> [Batch, 1, Features]
        features = np.concatenate([features_r6, features_r1], axis=2).reshape(lig_xyz.shape[0], 1, -1)
        
        # Mapping to fixed size vector
        current_pairs = [f"{r}_{l}" for r, l in itertools.product(self.rec.rec_ha_types, self.lig.lig_ha_ele)]
        init_matrix = np.zeros((len(current_pairs) * 2, self.output_dim), dtype=np.float32)
        
        # Optimized mapping construction
        # Note: This part is still loop-heavy but runs once per complex.
        # Can be optimized further with indexing if needed.
        for i, pair_base in enumerate(current_pairs):
            # pair_base ex: "ALA-N_C"
            k1 = f"r6_{pair_base}"
            k2 = f"r1_{pair_base}"
            if k1 in self.keys: init_matrix[2*i, self.keys.index(k1)] = 1
            if k2 in self.keys: init_matrix[2*i+1, self.keys.index(k2)] = 1
            
        init_matrix = np.repeat(init_matrix[None, ...], lig_xyz.shape[0], axis=0)
        
        # Matmul: [Batch, 1, Interactions] @ [Batch, Interactions, 1470] -> [Batch, 1, 1470]
        final_features = np.matmul(features, init_matrix).reshape(-1, self.output_dim)
        return final_features


class ContactCalculator:
    """Calculates native contact recovery rates efficiently."""
    def __init__(self, receptor: PocketFile, native_ligand: LigandFile, decoy: DecoyFile):
        self.rec_xyz = receptor.rec_ha_xyz
        self.nat_xyz = native_ligand.lig_ha_xyz
        self.dec_xyz = decoy.all_pose_xyz
        self.results = {}
        self._calculate()

    def _calculate(self):
        # 1. Identify Native Contacts
        # Shape: [N_lig_nat, N_rec]
        nat_dists = cdist(self.nat_xyz, self.rec_xyz)
        
        # Find indices (i, j) for native contacts < 10A
        # We process everything relative to the 10A set to avoid re-looping
        native_mask_10 = nat_dists < 10.0
        native_indices = np.argwhere(native_mask_10) # Array of [lig_idx, rec_idx]
        
        # Pre-compute subsets for tighter thresholds
        native_vals_10 = nat_dists[native_mask_10]
        mask_8_in_10 = native_vals_10 < 8.0
        mask_6_in_10 = native_vals_10 < 6.0
        mask_4_in_10 = native_vals_10 < 4.0
        
        # Counts
        cnt_nat = {
            'int_4': np.sum(mask_4_in_10),
            'int_6': np.sum(mask_6_in_10),
            'int_8': np.sum(mask_8_in_10),
            'int_10': len(native_vals_10),
            'exp_10': np.sum(np.exp(-native_vals_10 / 4)),
            'recip_10': np.sum(1 / (native_vals_10 + 0.1))
        }

        # 2. Process Decoys
        # We only care about the specific pairs identified in native_indices
        # dec_xyz shape: [N_poses, N_lig, 3]
        # We need distances for specific atom pairs across all poses.
        # Gather relevant ligand atoms and receptor atoms
        lig_idxs = native_indices[:, 0]
        rec_idxs = native_indices[:, 1]
        
        relevant_rec_atoms = self.rec_xyz[rec_idxs] # [K_pairs, 3]
        relevant_lig_atoms = self.dec_xyz[:, lig_idxs, :] # [N_poses, K_pairs, 3]
        
        # Calculate distances only for relevant pairs
        # Vectorized dist: sqrt(sum((x-y)^2))
        dists = np.linalg.norm(relevant_lig_atoms - relevant_rec_atoms, axis=2) # [N_poses, K_pairs]
        
        # 3. Compute Metrics per Pose
        # int_count
        self.results['decoy_num_int_4'] = (dists < 4.0).sum(axis=1) * mask_4_in_10 # Only count if native was <4
        self.results['decoy_num_int_6'] = (dists < 6.0).sum(axis=1) * mask_6_in_10
        self.results['decoy_num_int_8'] = (dists < 8.0).sum(axis=1) * mask_8_in_10
        self.results['decoy_num_int_10'] = (dists < 10.0).sum(axis=1)
        
        # continuous metrics
        self.results['decoy_num_exp_10'] = np.sum(np.exp(-dists / 4), axis=1)
        self.results['decoy_num_recip_10'] = np.sum(1 / (dists + 0.1), axis=1)
        
        # Ratios (safe division)
        for k in ['int_4', 'int_6', 'int_8', 'int_10', 'exp_10', 'recip_10']:
            num_key = f'decoy_num_{k}'
            ratio_key = f'decoy_ratio_{k}'
            denom = cnt_nat[k]
            if denom == 0:
                self.results[ratio_key] = np.zeros_like(self.results[num_key])
            else:
                self.results[ratio_key] = self.results[num_key] / denom

    def get_metrics_array(self):
        # Order matters to match original output
        keys = [
            'decoy_num_int_4', 'decoy_ratio_int_4',
            'decoy_num_int_6', 'decoy_ratio_int_6',
            'decoy_num_int_8', 'decoy_ratio_int_8',
            'decoy_num_int_10', 'decoy_ratio_int_10',
            'decoy_num_exp_10', 'decoy_ratio_exp_10',
            'decoy_num_recip_10', 'decoy_ratio_recip_10'
        ]
        # Stack columns: [N_poses, N_metrics]
        return np.stack([self.results[k] for k in keys], axis=1)


# ==========================================
# Main Execution
# ==========================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate Protein-Ligand Interaction Features",
        formatter_class=RawDescriptionHelpFormatter
    )
    parser.add_argument("-inp", type=str, default="inputs.dat", help="Input file list (target rec pose ref_lig)")
    parser.add_argument("-out", type=str, default="features_label.pkl", help="Output pickle path")
    parser.add_argument("-pre_cut", type=float, default=0.3, help="Min dist threshold")
    parser.add_argument("-cutoff", type=float, default=2.0, help="Max dist threshold")
    args = parser.parse_args()

    with open(args.inp) as f:
        inputs = [x.strip() for x in f.readlines() if not x.startswith("#") and x.strip()]

    all_indices = []
    all_values = []
    feature_names = []

    print(f"Processing {len(inputs)} complexes...")

    for i, line in enumerate(inputs):
        try:
            target, rec_path, pose_path, ref_lig_path = line.split()
            print(f"[{i+1}/{len(inputs)}] Processing {target} ...")

            # 1. Load Files
            rec = ReceptorFile(rec_path)
            decoy = DecoyFile(pose_path)
            ref_lig = LigandFile(ref_lig_path)

            # 2. Generate Interaction Features
            feat_gen = FeatureGenerator(rec, decoy, args.pre_cut, args.cutoff)
            interaction_features = feat_gen.compute()

            # 3. Generate Contact Labels
            pocket_center = np.mean(ref_lig.lig_ha_xyz, axis=0)
            pocket = PocketFile(rec_path, pocket_center, cutoff=35)
            
            contact_calc = ContactCalculator(pocket, ref_lig, decoy)
            contact_metrics = contact_calc.get_metrics_array()

            # 4. Merge
            full_features = np.concatenate([interaction_features, contact_metrics], axis=1)
            
            # 5. Store
            pose_basename = os.path.basename(pose_path).split(".")[0]
            indices = [f"{pose_basename}-{k+1}" for k in range(full_features.shape[0])]
            
            all_indices.extend(indices)
            all_values.append(full_features)

            if i == 0:
                # Define column names only once
                metric_names = [
                    'decoy_num_int_4', 'decoy_ratio_int_4',
                    'decoy_num_int_6', 'decoy_ratio_int_6',
                    'decoy_num_int_8', 'decoy_ratio_int_8',
                    'decoy_num_int_10', 'decoy_ratio_int_10',
                    'decoy_num_exp_10', 'decoy_ratio_exp_10',
                    'decoy_num_recip_10', 'decoy_ratio_recip_10'
                ]
                feature_names = feat_gen.keys + metric_names

        except Exception as e:
            print(f"Error processing {line}: {e}")
            continue

    if all_values:
        final_data = np.concatenate(all_values, axis=0)
        df = pd.DataFrame(final_data, index=all_indices, columns=feature_names)
        df.to_pickle(args.out)
        print(f"Saved features to {args.out} with shape {df.shape}")
    else:
        print("No features generated.")

if __name__ == "__main__":
    main()