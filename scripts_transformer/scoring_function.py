from utils import *
import numpy as np
import pandas as pd
import torch
import json
import itertools
from parse_ligand import Ligand
from parse_receptor import Receptor
from model import DeepRMSD
import os, sys
import time
import joblib

_current_dpath = os.path.dirname(os.path.abspath(__file__))

all_residues = ['GLY', 'ALA', 'VAL', 'LEU', 'ILE', 'PRO', 'PHE', 'TYR', 'TRP', 'SER',
                'THR', 'CYS', 'MET', 'ASN', 'GLN', 'ASP', 'GLU', 'LYS', 'ARG', 'HIS', 'OTH']
rec_elements = ['C', 'O', 'N', 'S', 'DU']
lig_elements = ['C', 'O', 'N', 'P', 'S', 'Hal', 'DU']
cols_to_delete = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 672, 673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768, 769, 826, 827, 828, 829, 830, 831, 832, 833, 834, 835, 836, 837, 838, 839, 896, 897, 898, 899, 900, 901, 902, 903, 904, 905, 906, 907, 908, 909, 952, 953, 954, 955, 956, 957, 958, 959, 960, 961, 962, 963, 964, 965, 966, 967, 968, 969, 970, 971, 972, 973, 974, 975, 976, 977, 978, 979, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035, 1036, 1037, 1038, 1039, 1040, 1041, 1042, 1043, 1044, 1045, 1046, 1047, 1048, 1049, 1092, 1093, 1094, 1095, 1096, 1097, 1098, 1099, 1100, 1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1109, 1110, 1111, 1112, 1113, 1114, 1115, 1116, 1117, 1118, 1119, 1162, 1163, 1164, 1165, 1166, 1167, 1168, 1169, 1170, 1171, 1172, 1173, 1174, 1175, 1176, 1177, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186, 1187, 1188, 1189, 1232, 1233, 1234, 1235, 1236, 1237, 1238, 1239, 1240, 1241, 1242, 1243, 1244, 1245, 1246, 1247, 1248, 1249, 1250, 1251, 1252, 1253, 1254, 1255, 1256, 1257, 1258, 1259, 1302, 1303, 1304, 1305, 1306, 1307, 1308, 1309, 1310, 1311, 1312, 1313, 1314, 1315, 1316, 1317, 1318, 1319, 1320, 1321, 1322, 1323, 1324, 1325, 1326, 1327, 1328, 1329, 1372, 1373, 1374, 1375, 1376, 1377, 1378, 1379, 1380, 1381, 1382, 1383, 1384, 1385, 1386, 1387, 1388, 1389, 1390, 1391, 1392, 1393, 1394, 1395, 1396, 1397, 1398, 1399, 1400, 1401, 1402, 1403, 1404, 1405, 1406, 1407, 1408, 1409, 1410, 1411, 1412, 1413, 1414, 1415, 1416, 1417, 1418, 1419, 1420, 1421, 1422, 1423, 1424, 1425, 1426, 1427, 1428, 1429, 1430, 1431, 1432, 1433, 1434, 1435, 1436, 1437, 1438, 1439, 1440, 1441, 1442, 1443, 1444, 1445, 1446, 1447, 1448, 1449, 1450, 1451, 1452, 1453, 1454, 1455, 1456, 1457, 1458, 1459, 1460, 1461, 1462, 1463, 1464, 1465, 1466, 1467, 1468, 1469]


Hal = ['F', 'Cl', 'Br', 'I']


def get_residue(r_atom):
    r, a = r_atom.split('-')
    if not r in all_residues:

        if r == "HID" or r == "HIE" or r == "HIP" or r == "HIZ" or r == "HIY":
            r = "HIS"
        elif r == "CYX" or r == "CYM" or r == "CYT":
            r = "CYS"
        elif r == "MEU":
            r = "MET"
        elif r == "LEV":
            r = "LEU"
        elif r == "ASQ" or r == "ASH" or r == "DID" or r == "DIC":
            r = "ASP"
        elif r == "GLZ":
            r = "GLY"
        elif r == "GLV" or r == "GLH" or r == "GLM":
            r = "GLU"
        elif r == "ASZ" or r == "ASM":
            r = "ASN"
        elif r == "GLO":
            r = "GLN"
        elif r == "SEM":
            r = "SER"
        elif r == "TYM":
            r = "TYR"
        elif r == "ALB":
            r = "ALA"
        else:
            print("OTH:", r)
            r = 'OTH'

    if a in rec_elements:
        a = a
    else:
        a = 'DU'
    return r + '-' + a

def get_elementtype(e):
    if e in lig_elements:
        return e
    elif e in Hal:
        return 'Hal'
    else:
        return 'DU'


residues_atoms_pairs = ["-".join(x) for x in list(itertools.product(all_residues, rec_elements))]
keys = []
for r, a in list(itertools.product(residues_atoms_pairs, lig_elements)):
    keys.append('r6_' + r + '_' + a)
    keys.append('r1_' + r + '_' + a)

class ScoringFunction(object):

    def __init__(self,
                 receptor: Receptor = None,
                 ligand: Ligand = None,
                 model_cached=None,
                 feat_scaler_cached=None,
                 label_scaler_cached=None,
                 pre_cut: float = 0.3,
                 cutoff: float = 2.0,
                 n_features: int = 1470,
                 alpha: float = 0.5,
                 weight_1: float = 0.5,
                 ):

        self.repulsive_ = 6
        self.pre_cut = pre_cut
        self.cutoff = cutoff
        self.n_features = n_features

        # RMSD_Vina
        self.alpha = alpha
        self.weight_1 = weight_1
        self.weight_2 = 1.0 - weight_1

        self.model = model_cached
        self.feat_scaler = feat_scaler_cached
        self.label_scaler = label_scaler_cached

        if self.model is None:
            raise ValueError("❌ ERROR: ScoringFunction requires model_cached, but received None")

        self.ligand = ligand
        self.pose_heavy_atoms_coords = self.ligand.pose_heavy_atoms_coords
        self.lig_heavy_atoms_element = self.ligand.lig_heavy_atoms_element
        self.updated_lig_heavy_atoms_xs_types = self.ligand.updated_lig_heavy_atoms_xs_types
        self.lig_root_atom_index = self.ligand.root_heavy_atom_index
        self.lig_frame_heavy_atoms_index_list = self.ligand.frame_heavy_atoms_index_list
        self.lig_torsion_bond_index = self.ligand.torsion_bond_index
        self.num_of_lig_ha = self.ligand.number_of_heavy_atoms
        self.number_of_poses = len(self.pose_heavy_atoms_coords)
        self.receptor = receptor
        self.rec_heavy_atoms_xyz = self.receptor.init_rec_heavy_atoms_xyz
        self.rec_heavy_atoms_xs_types = self.receptor.rec_heavy_atoms_xs_types
        self.residues_heavy_atoms_pairs = self.receptor.residues_heavy_atoms_pairs
        self.heavy_atoms_residues_indices = self.receptor.heavy_atoms_residues_indices
        self.rec_index_to_series_dict = self.receptor.rec_index_to_series_dict
        self.num_of_rec_ha = len(self.receptor.init_rec_heavy_atoms_xyz)

        # ============================================
        #  interaction
        # ============================================
        self.dist = torch.tensor([])
        self.intra_repulsive_term = torch.tensor(1e-6)
        self.inter_repulsive_term = torch.tensor(1e-6)

        self.vina_inter_energy = 0.0
        self.origin_energy = torch.tensor([])
        self.features_matrix = torch.tensor([])

        self.pred_rmsd = torch.tensor([])
        self.pred_ratio = torch.tensor([])

    
        with open(os.path.join(_current_dpath, "atomtype_mapping.json")) as f:
            self.atomtype_mapping = json.load(f)

        with open(os.path.join(_current_dpath, "covalent_radii_dict.json")) as f:
            self.covalent_radii_dict = json.load(f)

        with open(os.path.join(_current_dpath, "vdw_radii_dict.json")) as f:
            self.vdw_radii_dict = json.load(f)

    def generate_pldist_mtrx(self):

        self.rec_heavy_atoms_xyz = self.rec_heavy_atoms_xyz.expand(len(self.pose_heavy_atoms_coords), -1, 3)

        # Generate the distance matrix of heavy atoms between the protein and the ligand.
        n, N, C = self.rec_heavy_atoms_xyz.size()
        n, M, _ = self.pose_heavy_atoms_coords.size()
        dist = -2 * torch.matmul(self.rec_heavy_atoms_xyz, self.pose_heavy_atoms_coords.permute(0, 2, 1))
        dist += torch.sum(self.rec_heavy_atoms_xyz ** 2, -1).view(-1, N, 1)
        dist += torch.sum(self.pose_heavy_atoms_coords ** 2, -1).view(-1, 1, M)

        dist = (dist >= 0) * dist
        self.dist = torch.sqrt(dist)

        return self
    
    def cal_RMSD(self):
        dist_nm = self.dist / 10

        # ============================================
        # 1) 生成 r6 / r1 特征
        # ============================================
        dist_nm_1 = (dist_nm <= self.pre_cut) * self.pre_cut
        dist_nm_2 = dist_nm * (dist_nm > self.pre_cut) * (dist_nm < self.cutoff)

        features_1 = (torch.pow(dist_nm_1 + (dist_nm_1 == 0.) * 1., -6) -
                    (dist_nm_1 == 0.) * 1. +
                    torch.pow(dist_nm_2 + (dist_nm_2 == 0.) * 1., -6) -
                    (dist_nm_2 == 0.) * 1.).reshape(-1, 1)

        features_2 = (torch.pow(dist_nm_1 + (dist_nm_1 == 0.) * 1., -1) -
                    (dist_nm_1 == 0.) * 1. +
                    torch.pow(dist_nm_2 + (dist_nm_2 == 0.) * 1., -1) -
                    (dist_nm_2 == 0.) * 1.).reshape(-1, 1)

        features = torch.cat((features_1, features_2), axis=1)
        features = features.reshape(self.number_of_poses, 1, -1)

        # ============================================
        # 2) one-hot
        # ============================================
        residues = [get_residue(x) for x in self.residues_heavy_atoms_pairs]
        lig_ele = [get_elementtype(x) for x in self.lig_heavy_atoms_element]

        rec_lig_pairs = ["_".join(x) for x in itertools.product(residues, lig_ele)]

        cols = []
        for p in rec_lig_pairs:
            cols.append("r6_" + p)
            cols.append("r1_" + p)

        global init_matrix
        init_matrix = torch.zeros(len(cols), 1470)

        for idx, c in enumerate(cols):
            key_idx = keys.index(c)
            init_matrix[idx][key_idx] = 1

        init_matrix = init_matrix.expand(self.number_of_poses,
                                        init_matrix.shape[0],
                                        init_matrix.shape[1])

        matrix = torch.matmul(features, init_matrix)
        self.origin_energy = matrix.reshape(-1, 1470)

        mask = torch.ones(self.origin_energy.size(1), dtype=torch.bool)
        mask[cols_to_delete] = False
        data = self.origin_energy[:, mask]     # [poses, N_features]


        feat_scaler = self.feat_scaler   # sklearn MinMax / Std / etc.
        label_scaler = self.label_scaler
        model = self.model

        feat_mean = torch.from_numpy(feat_scaler.mean_).float()
        feat_std = torch.from_numpy(feat_scaler.scale_).float() + 1e-6

        data_norm = (data - feat_mean) / feat_std

        data_norm_gpu = data_norm.to("cuda")

        model_gpu = self.model

        # GPU forward
        pred_norm_gpu = model_gpu(data_norm_gpu)

        pred_norm = pred_norm_gpu.to("cpu")

        pred_outputs = []

        for i in range(pred_norm.shape[1]):
            y_mean = torch.from_numpy(label_scaler[i].mean_).float()
            y_std = torch.from_numpy(label_scaler[i].scale_).float() + 1e-6

            y = pred_norm[:, i] * y_std + y_mean
            pred_outputs.append(y.reshape(-1, 1))

        pred_label = torch.cat(pred_outputs, dim=1)

        self.pred_ratio = pred_label[:, 0].unsqueeze(1)
        self.pred_rmsd = pred_label[:, 1].unsqueeze(1)

        return self

    def cal_inter_repulsion(self, dist, vdw_sum):

        """
             When the distance between two atoms from the protein-ligand complex is less than the sum of the van der Waals radii,
            an intermolecular repulsion term is generated.
        """
        _cond = (dist < vdw_sum) * 1.
        _cond_sum = torch.sum(_cond, axis=1)
        _zero_indices = torch.where(_cond_sum == 0)[0]
        for index in _zero_indices:
            index = int(index)
            _cond[index][0] = torch.pow(dist[index][0], 20)

        self.inter_repulsive_term = torch.sum(torch.pow(_cond * dist + (_cond * dist == 0) * 1., -1 * self.repulsive_), axis=1) - \
                torch.sum((_cond * dist) * 1., axis=1)
        
        self.inter_repulsive_term = self.inter_repulsive_term.reshape(-1, 1)

        return self

    def cal_intra_repulsion(self):

        """

            When the distance between two atoms in adjacent frames are less than the sum of the van der Waals radii
        of the two atoms, an intramolecular repulsion term is generated.

        """
        all_root_frame_heavy_atoms_index_list = [self.lig_root_atom_index] + self.lig_frame_heavy_atoms_index_list
        number_of_all_frames = len(all_root_frame_heavy_atoms_index_list)

        dist_list = []
        vdw_list = []

        for frame_i in range(0, number_of_all_frames - 1):
            for frame_j in range(frame_i + 1, number_of_all_frames):

                for i in all_root_frame_heavy_atoms_index_list[frame_i]:
                    for j in all_root_frame_heavy_atoms_index_list[frame_j]:

                        if [i, j] in self.lig_torsion_bond_index or [j, i] in self.lig_torsion_bond_index:
                            continue

                        # angstrom
                        d = torch.sqrt(
                            torch.sum(
                                torch.square(self.pose_heavy_atoms_coords[:, i] - self.pose_heavy_atoms_coords[:, j]),
                                axis=1))
                        dist_list.append(d.reshape(-1, 1))

                        i_xs = self.updated_lig_heavy_atoms_xs_types[i]
                        j_xs = self.updated_lig_heavy_atoms_xs_types[j]

                        # angstrom
                        vdw_distance = self.vdw_radii_dict[i_xs] + self.vdw_radii_dict[j_xs]
                        vdw_list.append(torch.tensor([vdw_distance]))
    
        dist_tensor = torch.cat(dist_list, axis=1)
        vdw_tensor = torch.cat(vdw_list, axis=0)
    
        self.intra_repulsive_term = torch.sum(torch.pow((dist_tensor < vdw_tensor) * 1. * dist_tensor + \
                                                (dist_tensor >= vdw_tensor) * 1., -1 * self.repulsive_), axis=1) - \
                                                torch.sum((dist_tensor >= vdw_tensor) * 1., axis=1)

        self.intra_repulsive_term = self.intra_repulsive_term.reshape(-1, 1)

        return self

    def get_vdw_radii(self, xs):
        return self.vdw_radii_dict[xs]

    def get_vina_dist(self, r_index, l_index):
        return self.dist[:, r_index, l_index]

    def get_vina_rec_xs(self, index):
        return self.rec_heavy_atoms_xs_types[index]

    def get_vina_lig_xs(self, index):
        return self.updated_lig_heavy_atoms_xs_types[index]

    def is_hydrophobic(self, index, is_lig):

        if is_lig == True:
            atom_xs = self.updated_lig_heavy_atoms_xs_types[index]
        else:
            atom_xs = self.rec_heavy_atoms_xs_types[index]

        return atom_xs in ["C_H", "F_H", "Cl_H", "Br_H", "I_H"]

    def is_hbdonor(self, index, is_lig):

        if is_lig == True:
            atom_xs = self.updated_lig_heavy_atoms_xs_types[index]
        else:
            atom_xs = self.rec_heavy_atoms_xs_types[index]

        return atom_xs in ["N_D", "N_DA", "O_DA", "Met_D"]

    def is_hbacceptor(self, index, is_lig):

        if is_lig == True:
            atom_xs = self.updated_lig_heavy_atoms_xs_types[index]
        else:
            atom_xs = self.rec_heavy_atoms_xs_types[index]

        return atom_xs in ["N_A", "N_DA", "O_A", "O_DA"]

    def is_hbond(self, atom_1, atom_2):
        return (
                (self.is_hbdonor(atom_1) and self.is_hbacceptor(atom_2)) or
                (self.is_hbdonor(atom_2) and self.is_hbacceptor(atom_1))
        )

    def _pad(self, vector, _Max_dim):

        _vec = torch.zeros(_Max_dim - len(vector))
        new_vector = torch.cat((vector, _vec), axis=0)

        return new_vector

    def cal_vina(self):

        rec_atom_indices_list = []  # [[]]
        lig_atom_indices_list = []  # [[]]
        all_selected_rec_atom_indices = []
        all_selected_lig_atom_indices = []

        _Max_dim = 0
        for each_dist in self.dist:

            each_rec_atom_indices, each_lig_atom_indices = torch.where(each_dist <= 8)
            rec_atom_indices_list.append(each_rec_atom_indices.numpy().tolist())
            lig_atom_indices_list.append(each_lig_atom_indices.numpy().tolist())
            all_selected_rec_atom_indices += each_rec_atom_indices.numpy().tolist()
            all_selected_lig_atom_indices += each_lig_atom_indices.numpy().tolist()

            if len(each_rec_atom_indices) > _Max_dim:
                _Max_dim = len(each_rec_atom_indices)

        all_selected_rec_atom_indices = list(set(all_selected_rec_atom_indices))
        all_selected_lig_atom_indices = list(set(all_selected_lig_atom_indices))

        # Update the xs atom type of heavy atoms for receptor.
        for i in all_selected_rec_atom_indices:
            i = int(i)
            self.receptor.update_rec_xs(self.rec_heavy_atoms_xs_types[i], i,
                                        self.rec_index_to_series_dict[i], self.heavy_atoms_residues_indices[i])

        # is_hydrophobic
        rec_atom_is_hydrophobic_dict = dict(zip(all_selected_rec_atom_indices,
                                                np.array(list(map(self.is_hydrophobic, all_selected_rec_atom_indices,
                                                                  [False] * len(all_selected_rec_atom_indices)))) * 1.))
        lig_atom_is_hydrophobic_dict = dict(zip(all_selected_lig_atom_indices,
                                                np.array(list(map(self.is_hydrophobic, all_selected_lig_atom_indices,
                                                                  [True] * len(all_selected_lig_atom_indices)))) * 1.))

        # is_hbdonor
        rec_atom_is_hbdonor_dict = dict(zip(all_selected_rec_atom_indices,
                                            np.array(list(map(self.is_hbdonor, all_selected_rec_atom_indices,
                                                              [False] * len(all_selected_rec_atom_indices)))) * 1.))
        lig_atom_is_hbdonor_dict = dict(zip(all_selected_lig_atom_indices,
                                            np.array(list(map(self.is_hbdonor, all_selected_lig_atom_indices,
                                                              [True] * len(all_selected_lig_atom_indices)))) * 1.))

        # is_hbacceptor
        rec_atom_is_hbacceptor_dict = dict(zip(all_selected_rec_atom_indices,
                                               np.array(list(map(self.is_hbacceptor, all_selected_rec_atom_indices,
                                                                 [False] * len(all_selected_rec_atom_indices)))) * 1.))
        lig_atom_is_hbacceptor_dict = dict(zip(all_selected_lig_atom_indices,
                                               np.array(list(map(self.is_hbacceptor, all_selected_lig_atom_indices,
                                                                 [True] * len(all_selected_lig_atom_indices)))) * 1.))

        rec_lig_is_hydrophobic = []
        rec_lig_is_hbond = []
        rec_lig_atom_vdw_sum = []
        for each_rec_indices, each_lig_indices in zip(rec_atom_indices_list, lig_atom_indices_list):

            r_hydro = []
            l_hydro = []
            r_hbdonor = []
            l_hbdonor = []
            r_hbacceptor = []
            l_hbacceptor = []

            r_vdw = []
            l_vdw = []

            for r_index, l_index in zip(each_rec_indices, each_lig_indices):
                # is hydrophobic
                r_hydro.append(rec_atom_is_hydrophobic_dict[r_index])
                l_hydro.append(lig_atom_is_hydrophobic_dict[l_index])

                # is hbdonor & hbacceptor
                r_hbdonor.append(rec_atom_is_hbdonor_dict[r_index])
                l_hbdonor.append(lig_atom_is_hbdonor_dict[l_index])

                r_hbacceptor.append(rec_atom_is_hbacceptor_dict[r_index])
                l_hbacceptor.append(lig_atom_is_hbacceptor_dict[l_index])

                # vdw 
                r_vdw.append(self.vdw_radii_dict[self.rec_heavy_atoms_xs_types[r_index]])
                l_vdw.append(self.vdw_radii_dict[self.updated_lig_heavy_atoms_xs_types[l_index]])

            # rec-atom hydro
            r_hydro = self._pad(torch.from_numpy(np.array(r_hydro)), _Max_dim)
            l_hydro = self._pad(torch.from_numpy(np.array(l_hydro)), _Max_dim)
            rec_lig_is_hydrophobic.append(r_hydro * l_hydro.reshape(1, -1))

            # hbond
            r_hbdonor = self._pad(torch.from_numpy(np.array(r_hbdonor)), _Max_dim)
            l_hbdonor = self._pad(torch.from_numpy(np.array(l_hbdonor)), _Max_dim)

            r_hbacceptor = self._pad(torch.from_numpy(np.array(r_hbacceptor)), _Max_dim)
            l_hbacceptor = self._pad(torch.from_numpy(np.array(l_hbacceptor)), _Max_dim)
            _is_hbond = ((r_hbdonor * l_hbacceptor + r_hbacceptor * l_hbdonor) > 0) * 1.
            rec_lig_is_hbond.append(_is_hbond.reshape(1, -1))

            # rec-lig vdw 
            rec_lig_atom_vdw_sum.append(
                self._pad(torch.from_numpy(np.array(r_vdw) + np.array(l_vdw)), _Max_dim).reshape(1, -1))

        rec_lig_is_hydrophobic = torch.cat(rec_lig_is_hydrophobic, axis=0)
        rec_lig_is_hbond = torch.cat(rec_lig_is_hbond, axis=0)

        rec_lig_atom_vdw_sum = torch.cat(rec_lig_atom_vdw_sum, axis=0)

        # vina dist 
        vina_dist_list = []

        for _num, dist in enumerate(self.dist):
            dist = dist * ((dist <= 8) * 1.)
            l = len(dist[dist != 0])
            vina_dist_list.append(self._pad(dist[dist != 0], _Max_dim).reshape(1, -1))

        vina_dist = torch.cat(vina_dist_list, axis=0)

        vina = VinaScoreCore(vina_dist, rec_lig_is_hydrophobic, rec_lig_is_hbond, rec_lig_atom_vdw_sum)
        vina_inter_term = vina.process()
        self.vina_inter_energy = vina_inter_term / (
                    1 + 0.05846 * (self.ligand.active_torsion + 0.5 * self.ligand.inactive_torsion))

        self.vina_inter_energy = self.vina_inter_energy.reshape(-1, 1)

        # inter clash
        self.cal_inter_repulsion(vina_dist, rec_lig_atom_vdw_sum)

        return self

    def process(self):
        self.generate_pldist_mtrx()

        self.cal_RMSD()
        self.cal_vina()

        if self.ligand.number_of_frames == 0:
            self.intra_repulsive_term = 0
        else:
            self.cal_intra_repulsion()

        rmsd_vina = self.weight_1 * self.pred_rmsd + self.weight_2 * self.vina_inter_energy
        ratio_vina = 0.5 * self.vina_inter_energy - 3.5 * self.pred_ratio

        loss_main = ratio_vina
        loss_rep = torch.log(self.inter_repulsive_term + 1e-6)
        if self.ligand.number_of_frames > 0:
            loss_rep += torch.log(self.intra_repulsive_term + 1e-6)

        loss_geom = self.pred_rmsd

        combined_score = (
            0.7 * loss_main
            + 0.25 * loss_rep
            + 0.25 * loss_geom
        )


        return self.pred_rmsd, self.vina_inter_energy, rmsd_vina, combined_score, ratio_vina, self.pred_ratio

class VinaScoreCore(object):

    def __init__(self, dist_matrix, rec_lig_is_hydrophobic, rec_lig_is_hbond, rec_lig_atom_vdw_sum):
        """
        Args:
            dist_matrix [N, M]: the distance matrix with less than 8 angstroms. N is the number of poses,
        M is the number of rec-lig atom pairs less than 8 Angstroms in each pose.

        Returns:
            final_inter_score [N, 1]

        """

        self.dist_matrix = dist_matrix
        self.rec_lig_is_hydro = rec_lig_is_hydrophobic
        self.rec_lig_is_hb = rec_lig_is_hbond
        self.rec_lig_atom_vdw_sum = rec_lig_atom_vdw_sum

    def score_function(self):
        d_ij = self.dist_matrix - self.rec_lig_atom_vdw_sum

        Gauss_1 = torch.sum(torch.exp(- torch.pow(d_ij / 0.5, 2)), axis=1) - torch.sum((d_ij == 0) * 1., axis=1)
        Gauss_2 = torch.sum(torch.exp(- torch.pow((d_ij - 3) / 2, 2)), axis=1) - \
                  torch.sum((d_ij == 0) * 1. * torch.exp(torch.tensor(-1 * 9 / 4)), axis=1)
        #print("Gauss_1:", Gauss_1)
        #print("Gauss_2:", Gauss_2)

        # Repulsion
        Repulsion = torch.sum(torch.pow(((d_ij < 0) * d_ij), 2), axis=1)
        #print("Repulsion:", Repulsion)

        # Hydrophobic
        Hydro_1 = self.rec_lig_is_hydro * (d_ij <= 0.5) * 1.

        Hydro_2_condition = self.rec_lig_is_hydro * (d_ij > 0.5) * (d_ij < 1.5) * 1.
        Hydro_2 = 1.5 * Hydro_2_condition - Hydro_2_condition * d_ij

        Hydrophobic = torch.sum(Hydro_1 + Hydro_2, axis=1)
        #print("Hydro:", Hydrophobic)

        # HBonding
        hbond_1 = self.rec_lig_is_hb * (d_ij <= -0.7) * 1.
        hbond_2 = self.rec_lig_is_hb * (d_ij < 0) * (d_ij > -0.7) * 1.0 * (- d_ij) / 0.7
        HBonding = torch.sum(hbond_1 + hbond_2, axis=1)
        #print("HB:", HBonding)

        inter_energy = - 0.035579 * Gauss_1 - 0.005156 * Gauss_2 + 0.840245 * Repulsion - 0.035069 * Hydrophobic - 0.587439 * HBonding

        return inter_energy

    def process(self):
        final_inter_score = self.score_function()

        return final_inter_score
