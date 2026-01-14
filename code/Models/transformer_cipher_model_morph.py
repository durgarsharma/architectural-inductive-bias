"""
MORPH - CONSTRUCTED LANGUAGE
Transformer Model for Cipher Decryption
Self-attention based architecture with positional encoding
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer"""
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        return x + self.pe[:, :x.size(1), :]


class TransformerCipherModel(nn.Module):
    """
    Transformer architecture for cipher decryption
    - Character-level embeddings
    - Positional encoding
    - Multi-head self-attention layers
    - Character prediction at each position
    """
    
    def __init__(self, vocab_size=27, d_model=256, nhead=8, num_layers=4, 
                 dim_feedforward=1024, dropout=0.1):
        """
        Args:
            vocab_size: Size of vocabulary (26 letters + space = 27)
            d_model: Dimension of model embeddings
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout probability
        """
        super(TransformerCipherModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Character embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Output layer
        self.fc = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len)
        
        Returns:
            logits: Output tensor of shape (batch_size, seq_len, vocab_size)
        """
        # Embed characters and scale
        embedded = self.embedding(x) * math.sqrt(self.d_model)  # (batch_size, seq_len, d_model)
        
        # Add positional encoding
        embedded = self.pos_encoder(embedded)
        embedded = self.dropout(embedded)
        
        # Pass through transformer encoder
        transformer_out = self.transformer_encoder(embedded)  # (batch_size, seq_len, d_model)
        
        # Project to vocabulary
        logits = self.fc(transformer_out)  # (batch_size, seq_len, vocab_size)
        
        return logits
    
    def predict(self, x):
        """
        Predict characters (inference mode)
        
        Args:
            x: Input tensor of shape (batch_size, seq_len)
        
        Returns:
            predictions: Predicted character indices (batch_size, seq_len)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            predictions = torch.argmax(logits, dim=-1)
        return predictions


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    print("Testing Transformer Cipher Model...")
    
    model = TransformerCipherModel(
        vocab_size=27,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=1024,
        dropout=0.1
    )
    
    print(f"\nModel architecture:")
    print(model)
    
    print(f"\nTotal parameters: {count_parameters(model):,}")
    
    # Test forward pass
    batch_size = 4
    seq_len = 100
    
    dummy_input = torch.randint(0, 27, (batch_size, seq_len))
    output = model(dummy_input)
    
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test prediction
    predictions = model.predict(dummy_input)
    print(f"Predictions shape: {predictions.shape}")
    
    print("\n✓ Transformer model test passed!")