import torch
import torch.nn as nn

class DeepRMSD(nn.Module):
    """
    Transformer-based architecture for RMSD prediction.
    
    Architecture:
    Input -> Linear Embedding -> Transformer Encoder -> MLP Head -> Output
    """
    def __init__(self, input_dim, output_dim, num_heads, num_layers, rate):
        """
        Args:
            input_dim (int): Dimension of input features.
            output_dim (int): Dimension of output targets.
            num_heads (int): Number of attention heads in Transformer.
            num_layers (int): Number of Transformer encoder layers.
            rate (float): Dropout rate.
        """
        super(DeepRMSD, self).__init__()

        # Input Embedding: Maps input features to a high-dimensional space (512)
        self.input_embed = nn.Linear(input_dim, 512)

        # Transformer Encoder Layer
        # Note: We use the default PyTorch TransformerEncoder structure
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=num_heads, dropout=rate)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(p=rate)

        # MLP Prediction Head
        # Block 1
        self.fc3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=rate),
            nn.BatchNorm1d(256),
        )

        # Block 2
        self.fc4 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=rate),
            nn.BatchNorm1d(128),
        )

        # Block 3
        self.fc5 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=rate),
            nn.BatchNorm1d(64),
        )

        # Output Layer
        self.out = nn.Sequential(
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
        
        Returns:
            out (torch.Tensor): Output predictions of shape (batch_size, output_dim)
        """
        # 1. Embedding
        # Shape: (batch_size, input_dim) -> (batch_size, 512)
        x = self.input_embed(x)

        # 2. Sequence Formatting for Transformer
        # Transformer expects (seq_len, batch_size, d_model) by default.
        # Since our input is a single feature vector, we treat it as a sequence of length 1.
        
        # Add sequence dimension: (batch_size, 512) -> (batch_size, 1, 512)
        x = x.unsqueeze(1) 
        
        # Permute to (seq_len, batch_size, d_model): (1, batch_size, 512)
        x = x.permute(1, 0, 2)

        # 3. Transformer Encoder
        x = self.transformer_encoder(x)

        # 4. Extract Representation
        # Take the output of the first (and only) token
        # Shape: (1, batch_size, 512) -> (batch_size, 512)
        x = x[-1, :, :]

        # 5. MLP Head
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        
        out = self.out(x)
        
        return out