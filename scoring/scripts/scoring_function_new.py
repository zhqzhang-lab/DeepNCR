import os
import json
import itertools
import torch
import numpy as np
import joblib
from parse_ligand import Ligand
from parse_receptor import Receptor

# ==========================================
# Global Constants & Configuration
# ==========================================
_current_dpath = os.path.dirname(os.path.abspath(__file__))

ALL_RESIDUES = ['GLY', 'ALA', 'VAL', 'LEU', 'ILE', 'PRO', 'PHE', 'TYR', 'TRP', 'SER',
                'THR', 'CYS', 'MET', 'ASN', 'GLN', 'ASP', 'GLU', 'LYS', 'ARG', 'HIS', 'OTH']
REC_ELEMENTS = ['C', 'O', 'N', 'S', 'DU']
LIG_ELEMENTS = ['C', 'O', 'N', 'P', 'S', 'Hal', 'DU']
HALOGENS = ['F', 'Cl', 'Br', 'I']

def get_residue(r_atom):
    r, a = r_atom.split('-')
    r = r if r in ALL_RESIDUES else 'OTH'
    a = a if a in REC_ELEMENTS else 'DU'
    return f"{r}-{a}"

def get_elementtype(e):
    if e in LIG_ELEMENTS:
        return e
    elif e in HALOGENS:
        return 'Hal'
    else:
        return 'DU'

# Pre-compute defined keys for feature mapping
_residues_atoms_pairs = ["-".join(x) for x in itertools.product(ALL_RESIDUES, REC_ELEMENTS)]
DEFINED_KEYS = []
for r, a in itertools.product(_residues_atoms_pairs, LIG_ELEMENTS):
    DEFINED_KEYS.append(f'r6_{r}_{a}')
    DEFINED_KEYS.append(f'r1_{r}_{a}')


class ScoringFunction(object):
    """
    Main class for calculating DeepRMSD and Vina scores.
    """
    def __init__(self,
                 receptor: Receptor = None,
                 ligand: Ligand = None,
                 mean_std_file: str = None,
                 model_fpath: str = None,
                 model_cached=None,
                 feat_scaler_cached=None,
                 label_scaler_cached=None,
                 pre_cut: float = 0.3,
                 cutoff: float = 2.0,
                 n_features: int = 868):

        # Parameters for DeepRMSD
        self.pre_cut = pre_cut
        self.cutoff = cutoff
        self.n_features = n_features
        self.mean_std_file = mean_std_file
        self.model_fpath = model_fpath

        # Ligand Data
        self.ligand = ligand
        self.pose_heavy_atoms_coords = self.ligand.init_lig_heavy_atoms_xyz
        self.lig_heavy_atoms_element = self.ligand.lig_heavy_atoms_element
        self.updated_lig_heavy_atoms_xs_types = self.ligand.updated_lig_heavy_atoms_xs_types
        self.number_of_poses = len(self.pose_heavy_atoms_coords)

        # Receptor Data
        self.receptor = receptor
        self.rec_heavy_atoms_xyz = self.receptor.rec_heavy_atoms_xyz
        self.rec_heavy_atoms_xs_types = self.receptor.rec_heavy_atoms_xs_types
        self.residues_heavy_atoms_pairs = self.receptor.residues_heavy_atoms_pairs
        self.heavy_atoms_residues_indices = self.receptor.heavy_atoms_residues_indices
        self.rec_index_to_series_dict = self.receptor.rec_index_to_series_dict
        
        # Interaction Variables
        self.dist = torch.tensor([])
        self.vina_inter_energy = 0.0
        self.origin_energy = torch.tensor([])
        self.pred_rmsd = torch.tensor([])

        # Device & Cached Models
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_cached = model_cached
        self.feat_scaler_cached = feat_scaler_cached
        self.label_scaler_cached = label_scaler_cached
        
        # Ignored columns during inference (if applicable)
        self.cols_to_delete = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 672, 673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768, 769, 826, 827, 828, 829, 830, 831, 832, 833, 834, 835, 836, 837, 838, 839, 896, 897, 898, 899, 900, 901, 902, 903, 904, 905, 906, 907, 908, 909, 952, 953, 954, 955, 956, 957, 958, 959, 960, 961, 962, 963, 964, 965, 966, 967, 968, 969, 970, 971, 972, 973, 974, 975, 976, 977, 978, 979, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035, 1036, 1037, 1038, 1039, 1040, 1041, 1042, 1043, 1044, 1045, 1046, 1047, 1048, 1049, 1092, 1093, 1094, 1095, 1096, 1097, 1098, 1099, 1100, 1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1109, 1110, 1111, 1112, 1113, 1114, 1115, 1116, 1117, 1118, 1119, 1162, 1163, 1164, 1165, 1166, 1167, 1168, 1169, 1170, 1171, 1172, 1173, 1174, 1175, 1176, 1177, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186, 1187, 1188, 1189, 1232, 1233, 1234, 1235, 1236, 1237, 1238, 1239, 1240, 1241, 1242, 1243, 1244, 1245, 1246, 1247, 1248, 1249, 1250, 1251, 1252, 1253, 1254, 1255, 1256, 1257, 1258, 1259, 1302, 1303, 1304, 1305, 1306, 1307, 1308, 1309, 1310, 1311, 1312, 1313, 1314, 1315, 1316, 1317, 1318, 1319, 1320, 1321, 1322, 1323, 1324, 1325, 1326, 1327, 1328, 1329, 1372, 1373, 1374, 1375, 1376, 1377, 1378, 1379, 1380, 1381, 1382, 1383, 1384, 1385, 1386, 1387, 1388, 1389, 1390, 1391, 1392, 1393, 1394, 1395, 1396, 1397, 1398, 1399, 1400, 1401, 1402, 1403, 1404, 1405, 1406, 1407, 1408, 1409, 1410, 1411, 1412, 1413, 1414, 1415, 1416, 1417, 1418, 1419, 1420, 1421, 1422, 1423, 1424, 1425, 1426, 1427, 1428, 1429, 1430, 1431, 1432, 1433, 1434, 1435, 1436, 1437, 1438, 1439, 1440, 1441, 1442, 1443, 1444, 1445, 1446, 1447, 1448, 1449, 1450, 1451, 1452, 1453, 1454, 1455, 1456, 1457, 1458, 1459, 1460, 1461, 1462, 1463, 1464, 1465, 1466, 1467, 1468, 1469]


        # Load parameter dictionaries
        with open(os.path.join(_current_dpath, "atomtype_mapping.json")) as f:
            self.atomtype_mapping = json.load(f)
        with open(os.path.join(_current_dpath, "covalent_radii_dict.json")) as f:
            self.covalent_radii_dict = json.load(f)
        with open(os.path.join(_current_dpath, "vdw_radii_dict.json")) as f:
            self.vdw_radii_dict = json.load(f)

    def generate_pldist_mtrx(self):
        """
        Generates the pairwise distance matrix between receptor and ligand heavy atoms.
        """
        # Expand receptor coordinates to match the number of poses
        # Shape: (N_poses, N_rec_atoms, 3)
        rec_xyz_expanded = self.rec_heavy_atoms_xyz.expand(self.number_of_poses, -1, 3)

        # Calculate Euclidean distance using matrix operations
        # (a-b)^2 = a^2 + b^2 - 2ab
        n, N, C = rec_xyz_expanded.size()
        n, M, _ = self.pose_heavy_atoms_coords.size()
        
        dist = -2 * torch.matmul(rec_xyz_expanded, self.pose_heavy_atoms_coords.permute(0, 2, 1))
        dist += torch.sum(rec_xyz_expanded ** 2, -1).view(-1, N, 1)
        dist += torch.sum(self.pose_heavy_atoms_coords ** 2, -1).view(-1, 1, M)

        # Clamp negative values due to float precision errors and take sqrt
        dist = (dist >= 0) * dist
        self.dist = torch.sqrt(dist)

        return self

    def cal_RMSD(self):
        """
        Calculates the DeepRMSD score using the pre-trained Transformer model.
        """
        device = self.device

        # 1. Feature Generation (Vectorized on GPU/Device)
        dist_nm = (self.dist / 10).to(device)
        
        # Define interaction zones
        mask_close = (dist_nm <= self.pre_cut)
        mask_inter = (dist_nm > self.pre_cut) & (dist_nm < self.cutoff)
        
        dist_nm_1 = mask_close * self.pre_cut
        dist_nm_2 = dist_nm * mask_inter

        # Avoid division by zero
        eps_mask_1 = (dist_nm_1 == 0.)
        eps_mask_2 = (dist_nm_2 == 0.)

        # Compute Lennard-Jones-like terms (r^-6 and r^-1)
        r6_term_1 = torch.pow(dist_nm_1 + eps_mask_1 * 1.0, -6) - eps_mask_1 * 1.0
        r6_term_2 = torch.pow(dist_nm_2 + eps_mask_2 * 1.0, -6) - eps_mask_2 * 1.0
        features_r6 = (r6_term_1 + r6_term_2).reshape(-1, 1)

        r1_term_1 = torch.pow(dist_nm_1 + eps_mask_1 * 1.0, -1) - eps_mask_1 * 1.0
        r1_term_2 = torch.pow(dist_nm_2 + eps_mask_2 * 1.0, -1) - eps_mask_2 * 1.0
        features_r1 = (r1_term_1 + r1_term_2).reshape(-1, 1)

        # Concatenate features: Shape (N_poses, 1, N_pairs * 2)
        features = torch.cat((features_r6, features_r1), dim=1)
        features = features.reshape(self.number_of_poses, 1, -1)

        # 2. Build Interaction Mapping Matrix (Target Specific)
        # Note: This runs on CPU as string operations are not GPU compatible
        curr_rec_res_atoms = [get_residue(x) for x in self.residues_heavy_atoms_pairs]
        curr_lig_ele = [get_elementtype(x) for x in self.lig_heavy_atoms_element]

        rec_lig_pairs = ["_".join(x) for x in itertools.product(curr_rec_res_atoms, curr_lig_ele)]
        
        rec_lig_feature_keys = []
        for pair in rec_lig_pairs:
            rec_lig_feature_keys.append("r6_" + pair)
            rec_lig_feature_keys.append("r1_" + pair)

        # Create One-Hot Mapping Matrix
        # Shape: (Current_Pairs_Count, Total_Feature_Dim)
        init_matrix = torch.zeros(len(rec_lig_feature_keys), 1470)
        
        # Map current complex keys to global feature vector indices
        for i, key in enumerate(rec_lig_feature_keys):
            if key in DEFINED_KEYS:
                idx = DEFINED_KEYS.index(key)
                init_matrix[i][idx] = 1

        # 3. Aggregation via Matrix Multiplication (CPU)
        # We perform this on CPU to avoid large memory spikes on GPU for the sparse mapping
        features_cpu = features.cpu()
        init_matrix_cpu = init_matrix.cpu()

        # [Batch, 1, Pairs] @ [Pairs, 1470] -> [Batch, 1, 1470]
        raw_feature_vector = torch.matmul(features_cpu, init_matrix_cpu)
        self.origin_energy = raw_feature_vector.reshape(-1, 1470)

        # 4. Feature Masking (Remove specific columns if needed)
        # Caching the mask to avoid re-creation
        global _MASK_CACHE
        if "_MASK_CACHE" not in globals():
            m = torch.ones(1470, dtype=torch.bool)
            m[torch.tensor(self.cols_to_delete)] = False
            _MASK_CACHE = m
        
        data = self.origin_energy[:, _MASK_CACHE].numpy()

        # 5. Model Inference (GPU)
        data = self.feat_scaler_cached.transform(data)
        data_tensor = torch.from_numpy(data).float().to(device)

        with torch.no_grad():
            pred_out = self.model_cached(data_tensor).cpu().numpy()

        # 6. Inverse Transform Labels
        # Assuming output dim is handled by list of scalers
        for i in range(pred_out.shape[1]):
            scaler = self.label_scaler_cached[i]
            pred_out[:, i] = scaler.inverse_transform(pred_out[:, i].reshape(-1, 1)).flatten()

        self.pred_rmsd = pred_out
        return self

    # ==========================
    # Vina Helper Functions
    # ==========================
    
    def get_vdw_radii(self, xs):
        return self.vdw_radii_dict.get(xs, 1.5)

    def is_hydrophobic(self, index, is_lig):
        atom_xs = self.updated_lig_heavy_atoms_xs_types[index] if is_lig else self.rec_heavy_atoms_xs_types[index]
        return atom_xs in ["C_H", "F_H", "Cl_H", "Br_H", "I_H"]

    def is_hbdonor(self, index, is_lig):
        atom_xs = self.updated_lig_heavy_atoms_xs_types[index] if is_lig else self.rec_heavy_atoms_xs_types[index]
        return atom_xs in ["N_D", "N_DA", "O_DA", "Met_D"]

    def is_hbacceptor(self, index, is_lig):
        atom_xs = self.updated_lig_heavy_atoms_xs_types[index] if is_lig else self.rec_heavy_atoms_xs_types[index]
        return atom_xs in ["N_A", "N_DA", "O_A", "O_DA"]

    def cal_vina(self):
        """
        Calculates Vina intermolecular energy terms.
        Optimized to process each pose individually to handle dynamic interaction pair sizes.
        """
        vina_inter_list = []

        for each_dist in self.dist:
            # Filter atoms within 8 Angstroms
            # Note: indices correspond to receptor (dim 0) and ligand (dim 1)
            rec_indices, lig_indices = torch.where(each_dist <= 8)
            
            # Convert to list for efficient indexing in Python loops
            rec_idx_list = rec_indices.tolist()
            lig_idx_list = lig_indices.tolist()

            if not rec_idx_list:
                vina_inter_list.append(torch.tensor([0.0]))
                continue

            # Ensure indices are within bounds (Safety check)
            rec_idx_list = [i for i in rec_idx_list if i < len(self.rec_heavy_atoms_xs_types)]
            lig_idx_list = [i for i in lig_idx_list if i < len(self.updated_lig_heavy_atoms_xs_types)]
            
            if not rec_idx_list or not lig_idx_list:
                vina_inter_list.append(torch.tensor([0.0]))
                continue

            # Update receptor atom types for the active pocket
            # (Vina requires specific context-dependent atom typing)
            for i in rec_idx_list:
                self.receptor.update_rec_xs(
                    self.rec_heavy_atoms_xs_types[i], 
                    i,
                    self.rec_index_to_series_dict[i],
                    self.heavy_atoms_residues_indices[i]
                )

            # Pre-calculate property vectors
            r_hydro = torch.tensor([self.is_hydrophobic(i, False) for i in rec_idx_list], dtype=torch.float)
            l_hydro = torch.tensor([self.is_hydrophobic(i, True) for i in lig_idx_list], dtype=torch.float)
            rec_lig_is_hydrophobic = r_hydro * l_hydro

            r_hbd = torch.tensor([self.is_hbdonor(i, False) for i in rec_idx_list], dtype=torch.float)
            l_hbd = torch.tensor([self.is_hbdonor(i, True) for i in lig_idx_list], dtype=torch.float)
            r_hba = torch.tensor([self.is_hbacceptor(i, False) for i in rec_idx_list], dtype=torch.float)
            l_hba = torch.tensor([self.is_hbacceptor(i, True) for i in lig_idx_list], dtype=torch.float)
            
            # Hydrogen Bond: (Donor_Rec * Acceptor_Lig) + (Acceptor_Rec * Donor_Lig)
            rec_lig_is_hbond = ((r_hbd * l_hba + r_hba * l_hbd) > 0).float()

            r_vdw = torch.tensor([self.vdw_radii_dict.get(self.rec_heavy_atoms_xs_types[i], 1.5) for i in rec_idx_list])
            l_vdw = torch.tensor([self.vdw_radii_dict.get(self.updated_lig_heavy_atoms_xs_types[i], 1.5) for i in lig_idx_list])
            rec_lig_atom_vdw_sum = r_vdw + l_vdw

            # Extract distances
            # Clone to ensure we don't modify the original matrix logic
            d_val = each_dist.clone()
            d_val[d_val > 8] = 0 # Safety filter
            vina_dist = d_val[rec_idx_list, lig_idx_list]

            # Compute Vina Score Core
            vina_core = VinaScoreCore(
                vina_dist.reshape(1, -1),
                rec_lig_is_hydrophobic.reshape(1, -1),
                rec_lig_is_hbond.reshape(1, -1),
                rec_lig_atom_vdw_sum.reshape(1, -1)
            )
            score = vina_core.process()

            # Normalize by torsion count
            torsion_factor = 1 + 0.05846 * (self.ligand.active_torsion + 0.5 * self.ligand.inactive_torsion)
            score_normalized = score / torsion_factor
            
            vina_inter_list.append(score_normalized.reshape(-1))

        # Stack results: [N_poses, 1]
        self.vina_inter_energy = torch.stack(vina_inter_list).reshape(-1, 1)
        return self


class VinaScoreCore(object):
    """
    Vectorized implementation of the AutoDock Vina scoring function terms.
    """
    def __init__(self, dist_matrix, rec_lig_is_hydrophobic, rec_lig_is_hbond, rec_lig_atom_vdw_sum):
        """
        Args:
            dist_matrix (torch.Tensor): Distances between interacting atoms.
            rec_lig_is_hydrophobic (torch.Tensor): Mask for hydrophobic interactions.
            rec_lig_is_hbond (torch.Tensor): Mask for hydrogen bond interactions.
            rec_lig_atom_vdw_sum (torch.Tensor): Sum of Van der Waals radii.
        """
        self.dist_matrix = dist_matrix
        self.rec_lig_is_hydro = rec_lig_is_hydrophobic
        self.rec_lig_is_hb = rec_lig_is_hbond
        self.rec_lig_atom_vdw_sum = rec_lig_atom_vdw_sum

    def score_function(self):
        # Surface distance: d_ij = r_ij - (R_i + R_j)
        d_ij = self.dist_matrix - self.rec_lig_atom_vdw_sum

        # Gaussian Terms (Steric)
        # Gauss 1: exp(-(d/0.5)^2)
        gauss_1 = torch.sum(torch.exp(- torch.pow(d_ij / 0.5, 2)), axis=1)
        # Fix: Remove zero-padded entries if any exist (though input should be filtered)
        gauss_1 -= torch.sum((self.dist_matrix == 0).float(), axis=1)

        # Gauss 2: exp(-((d-3)/2)^2)
        gauss_2_raw = torch.exp(- torch.pow((d_ij - 3) / 2, 2))
        gauss_2 = torch.sum(gauss_2_raw, axis=1)
        # Correction for zero-padding in d_ij
        gauss_2 -= torch.sum((self.dist_matrix == 0).float() * torch.exp(torch.tensor(-2.25)), axis=1)

        # Repulsion Term (d < 0)
        repulsion = torch.sum(torch.pow(((d_ij < 0) * d_ij), 2), axis=1)

        # Hydrophobic Term
        # Condition 1: d <= 0.5 -> 1.0
        hydro_1 = self.rec_lig_is_hydro * (d_ij <= 0.5).float()
        # Condition 2: 0.5 < d < 1.5 -> Linear ramp
        hydro_cond_2 = self.rec_lig_is_hydro * (d_ij > 0.5) * (d_ij < 1.5)
        hydro_2 = hydro_cond_2.float() * (1.5 - d_ij)
        
        hydrophobic = torch.sum(hydro_1 + hydro_2, axis=1)

        # Hydrogen Bonding Term
        # Condition 1: d <= -0.7 -> 1.0
        hbond_1 = self.rec_lig_is_hb * (d_ij <= -0.7).float()
        # Condition 2: -0.7 < d < 0 -> Linear ramp
        hbond_cond_2 = self.rec_lig_is_hb * (d_ij < 0) * (d_ij > -0.7)
        hbond_2 = hbond_cond_2.float() * (d_ij / -0.7)
        
        hbonding = torch.sum(hbond_1 + hbond_2, axis=1)

        # Weights from AutoDock Vina
        # Weights: Gauss1 (-0.035579), Gauss2 (-0.005156), Repulsion (0.840245), Hydro (-0.035069), HB (-0.587439)
        inter_energy = (-0.035579 * gauss_1 
                        - 0.005156 * gauss_2 
                        + 0.840245 * repulsion 
                        - 0.035069 * hydrophobic 
                        - 0.587439 * hbonding)

        return inter_energy

    def process(self):
        return self.score_function()