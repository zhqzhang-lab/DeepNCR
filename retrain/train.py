import os
import argparse
import datetime
import joblib
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler

# ==========================================
# Utils & Setup
# ==========================================

def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(directory):
    """Create directory if it does not exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

# ==========================================
# Dataset
# ==========================================

class ComplexDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __getitem__(self, item):
        return self.features[item].clone().detach(), self.labels[item]

    def __len__(self):
        return len(self.features)

class DataProcessor:
    """Handles data loading, cleaning, and preprocessing."""
    
    def __init__(self, ignore_cols):
        self.ignore_cols = ignore_cols
        self.feat_scaler = StandardScaler()
        self.label_scalers = []

    def load_data(self, file_path, mode="train", batch_size=64):
        print(f"Loading {mode} data from {file_path} ...")
        df = pd.read_pickle(file_path)
        
        # Drop specified columns
        if self.ignore_cols:
            df.drop(df.columns[self.ignore_cols], axis=1, inplace=True)
            
        print(f"{mode} data shape: {df.shape}")
        
        values = df.values
        # Assuming last 13 columns contain metadata/labels, and specific indices for targets
        # Adjust indices if your data structure changes
        feats = values[:, :-13]
        labels = values[:, [-10, -1]] 

        # Feature Scaling
        if mode == "train":
            feats = self.feat_scaler.fit_transform(feats)
            ensure_dir("saved_scalers")
            joblib.dump(self.feat_scaler, "saved_scalers/feat_scaler.pkl")
        else:
            # Load scaler if not in memory (optional safety check)
            if not hasattr(self.feat_scaler, 'mean_'):
                self.feat_scaler = joblib.load("saved_scalers/feat_scaler.pkl")
            feats = self.feat_scaler.transform(feats)

        # Label Scaling
        labels = self._process_labels(labels, mode)

        # Convert to Tensor
        feats_tensor = torch.from_numpy(feats).float()
        labels_tensor = torch.from_numpy(labels).float()
        
        dataset = ComplexDataset(feats_tensor, labels_tensor)
        shuffle = (mode == "train")
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def _process_labels(self, labels, mode):
        """Normalizes labels column-wise."""
        if mode == "train":
            self.label_scalers = []
            for i in range(labels.shape[1]):
                scaler = StandardScaler()
                labels[:, i] = scaler.fit_transform(labels[:, i].reshape(-1, 1)).flatten()
                self.label_scalers.append(scaler)
            joblib.dump(self.label_scalers, "saved_scalers/label_scalers.pkl")
        else:
            if not self.label_scalers:
                self.label_scalers = joblib.load("saved_scalers/label_scalers.pkl")
            for i in range(labels.shape[1]):
                labels[:, i] = self.label_scalers[i].transform(labels[:, i].reshape(-1, 1)).flatten()
        return labels

# ==========================================
# Model Architecture
# ==========================================

class DeepRMSD(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, num_layers, dropout_rate):
        super(DeepRMSD, self).__init__()
        
        self.input_embed = nn.Linear(input_dim, 512)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=num_heads, dropout=dropout_rate)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # MLP Block
        self.mlp = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.BatchNorm1d(256),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.BatchNorm1d(128),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.BatchNorm1d(64),
            
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        x = self.input_embed(x)             # (batch_size, 512)
        x = x.unsqueeze(1)                  # (batch_size, 1, 512) -> Fake seq_len
        x = x.permute(1, 0, 2)              # (seq_len=1, batch_size, 512)
        
        x = self.transformer_encoder(x)
        x = x[-1, :, :]                     # Take last output: (batch_size, 512)
        
        x = self.dropout(x)
        out = self.mlp(x)
        return out

# ==========================================
# Metrics & Loss
# ==========================================

def calc_rmse(y_true, y_pred):
    """Root Mean Square Error."""
    return torch.sqrt(torch.mean(torch.pow(y_pred - y_true, 2), axis=-1))

def calc_pcc(y_true, y_pred):
    """Pearson Correlation Coefficient."""
    fsp = y_pred - torch.mean(y_pred)
    fst = y_true - torch.mean(y_true)
    devP = torch.std(y_pred, unbiased=False)
    devT = torch.std(y_true, unbiased=False)
    # Add epsilon to avoid division by zero
    pcc = torch.mean(fsp * fst) / (devP * devT + 1e-8)
    return pcc

def weighted_mse_loss(pred, target, weights):
    """Weighted Mean Squared Error for multi-target regression."""
    # MSE per column
    loss = torch.mean((pred - target) ** 2, dim=0) 
    weighted_loss = loss * weights.to(loss.device)
    return weighted_loss.sum()

# ==========================================
# Training Engine
# ==========================================

class Trainer:
    def __init__(self, model, device, optimizer, patience=20):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.patience = patience
        self.loss_weights = torch.tensor([0.8, 1.2], dtype=torch.float32)
        
        ensure_dir('Log_ROOT')
        ensure_dir('Model_ROOT')

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        batch_loss, batch_pcc, batch_rmse = [], [], []

        for step, (x, y) in enumerate(dataloader):
            b_x, b_y = x.to(self.device), y.to(self.device)
            
            pred = self.model(b_x)
            loss = weighted_mse_loss(pred, b_y, self.loss_weights)
            
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5, norm_type=2)
            self.optimizer.step()
            
            batch_loss.append(loss.item())
            
            # Metrics for each target
            for i in range(2):
                batch_pcc.append(calc_pcc(b_y[:, i], pred[:, i]).item())
                batch_rmse.append(calc_rmse(b_y[:, i], pred[:, i]).mean().item())

        return np.mean(batch_loss), np.mean(batch_pcc), np.mean(batch_rmse)

    def validate(self, dataloader):
        self.model.eval()
        val_loss, val_pcc, val_rmse = 0, np.zeros(2), np.zeros(2)
        steps = 0
        
        with torch.no_grad():
            for x, y in dataloader:
                b_x, b_y = x.to(self.device), y.to(self.device)
                pred = self.model(b_x)
                
                loss = weighted_mse_loss(pred, b_y, self.loss_weights)
                val_loss += loss.item()
                
                for i in range(2):
                    val_pcc[i] += calc_pcc(b_y[:, i], pred[:, i]).item()
                    val_rmse[i] += calc_rmse(b_y[:, i], pred[:, i]).mean().item()
                steps += 1
        
        return val_loss / steps, val_pcc / steps, val_rmse / steps

    def run(self, train_loader, valid_loader, epochs, log_name):
        best_val_loss = float('inf')
        patience_counter = 0
        log_path = os.path.join('Log_ROOT', log_name)
        model_path = os.path.join('Model_ROOT', f"{log_name}.pth")

        # Init Log file
        with open(log_path, 'w') as f:
            f.write("epoch,train_loss,train_PCC,train_RMSE,valid_loss,valid_PCC,valid_RMSE\n")

        print("Start training ...")
        for epoch in range(epochs):
            t_loss, t_pcc, t_rmse = self.train_epoch(train_loader, epoch)
            v_loss, v_pcc_arr, v_rmse_arr = self.validate(valid_loader)
            v_pcc_mean, v_rmse_mean = np.mean(v_pcc_arr), np.mean(v_rmse_arr)

            # Logging
            print(f'Epoch: {epoch:03d} | Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f} | Val PCC: {v_pcc_mean:.4f}')
            
            with open(log_path, 'a') as f:
                f.write(f"{epoch},{t_loss},{t_pcc},{t_rmse},{v_loss},{v_pcc_mean},{v_rmse_mean}\n")

            # Early Stopping & Checkpointing
            if v_loss < best_val_loss - 1e-5:
                print(f"  [Save] Val Loss improved {best_val_loss:.4f} -> {v_loss:.4f}")
                best_val_loss = v_loss
                torch.save(self.model, model_path)
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"  [Info] EarlyStopping counter: {patience_counter}/{self.patience}")
            
            if patience_counter >= self.patience:
                print("EarlyStopping triggered!")
                break

# ==========================================
# Main
# ==========================================

# Indices to be removed (Collapsed for readability)
IGNORED_FEATURE_INDICES = [
    42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 
    112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 
    134, 135, 136, 137, 138, 139, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 
    198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 
    262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 322, 323, 324, 325, 
    326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 
    348, 349, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 
    412, 413, 414, 415, 416, 417, 418, 419, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 
    476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 532, 533, 534, 535, 536, 537, 538, 539, 
    540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 602, 603, 
    604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 
    626, 627, 628, 629, 672, 673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 
    690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751, 752, 753, 
    754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768, 769, 826, 827, 828, 829, 830, 831, 
    832, 833, 834, 835, 836, 837, 838, 839, 896, 897, 898, 899, 900, 901, 902, 903, 904, 905, 906, 907, 908, 909, 
    952, 953, 954, 955, 956, 957, 958, 959, 960, 961, 962, 963, 964, 965, 966, 967, 968, 969, 970, 971, 972, 973, 
    974, 975, 976, 977, 978, 979, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 
    1035, 1036, 1037, 1038, 1039, 1040, 1041, 1042, 1043, 1044, 1045, 1046, 1047, 1048, 1049, 1092, 1093, 1094, 
    1095, 1096, 1097, 1098, 1099, 1100, 1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1109, 1110, 1111, 1112, 
    1113, 1114, 1115, 1116, 1117, 1118, 1119, 1162, 1163, 1164, 1165, 1166, 1167, 1168, 1169, 1170, 1171, 1172, 
    1173, 1174, 1175, 1176, 1177, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186, 1187, 1188, 1189, 1232, 
    1233, 1234, 1235, 1236, 1237, 1238, 1239, 1240, 1241, 1242, 1243, 1244, 1245, 1246, 1247, 1248, 1249, 1250, 
    1251, 1252, 1253, 1254, 1255, 1256, 1257, 1258, 1259, 1302, 1303, 1304, 1305, 1306, 1307, 1308, 1309, 1310, 
    1311, 1312, 1313, 1314, 1315, 1316, 1317, 1318, 1319, 1320, 1321, 1322, 1323, 1324, 1325, 1326, 1327, 1328, 
    1329, 1372, 1373, 1374, 1375, 1376, 1377, 1378, 1379, 1380, 1381, 1382, 1383, 1384, 1385, 1386, 1387, 1388, 
    1389, 1390, 1391, 1392, 1393, 1394, 1395, 1396, 1397, 1398, 1399, 1400, 1401, 1402, 1403, 1404, 1405, 1406, 
    1407, 1408, 1409, 1410, 1411, 1412, 1413, 1414, 1415, 1416, 1417, 1418, 1419, 1420, 1421, 1422, 1423, 1424, 
    1425, 1426, 1427, 1428, 1429, 1430, 1431, 1432, 1433, 1434, 1435, 1436, 1437, 1438, 1439, 1440, 1441, 1442, 
    1443, 1444, 1445, 1446, 1447, 1448, 1449, 1450, 1451, 1452, 1453, 1454, 1455, 1456, 1457, 1458, 1459, 1460, 
    1461, 1462, 1463, 1464, 1465, 1466, 1467, 1468, 1469
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DeepRMSD Training Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--train_file", type=str, default="train_features_label.csv", help="Path to training set pickle/csv")
    parser.add_argument("--valid_file", type=str, default="valid_features_label.csv", help="Path to validation set pickle/csv")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use (cuda or cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log_name", type=str, default="multivariable2_rmsd_ratio", help="Name for log file and model checkpoint")

    args = parser.parse_args()
    
    # 1. Setup
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data
    processor = DataProcessor(ignore_cols=IGNORED_FEATURE_INDICES)
    train_loader = processor.load_data(args.train_file, mode="train", batch_size=args.batch_size)
    valid_loader = processor.load_data(args.valid_file, mode="eval", batch_size=args.batch_size)

    # 3. Model
    # Note: Adjust input_dim calculation if the number of dropped columns changes logic
    # Original logic: 1470 (total) - len(dropped) = 868
    # But code says input_dim=868 manually. 
    # For safety, you might want to dynamically calculate: train_loader.dataset.features.shape[1]
    sample_feat, _ = next(iter(train_loader))
    input_dim = sample_feat.shape[1] 
    print(f"Detected Input Dimension: {input_dim}")

    model = DeepRMSD(
        input_dim=input_dim, 
        output_dim=2, 
        num_heads=8, 
        num_layers=3, 
        dropout_rate=args.dropout
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params}")

    # 4. Training
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    trainer = Trainer(model, device, optimizer, patience=20)
    
    start_time = datetime.datetime.now()
    trainer.run(train_loader, valid_loader, args.epochs, args.log_name)
    end_time = datetime.datetime.now()
    
    with open('time_running.dat', 'w') as f:
        f.write(f'Start Time: {start_time}\n')
        f.write(f'End Time:   {end_time}\n')
        f.write(f'Duration:   {end_time - start_time}\n')