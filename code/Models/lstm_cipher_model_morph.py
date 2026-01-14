"""
MORPH - CONSTRUCTED LANGUAGE
LSTM Model for Cipher Decryption
Bidirectional LSTM with character-level processing
"""

import torch
import torch.nn as nn


class LSTMCipherModel(nn.Module):
    """
    LSTM architecture for cipher decryption
    - Character-level embeddings
    - Bidirectional LSTM layers
    - Character prediction at each position
    """
    
    def __init__(self, vocab_size=27, embedding_dim=128, hidden_dim=256, 
                 num_layers=2, dropout=0.3, bidirectional=True):
        """
        Args:
            vocab_size: Size of vocabulary (26 letters + space = 27)
            embedding_dim: Dimension of character embeddings
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            bidirectional: Use bidirectional LSTM
        """
        super(LSTMCipherModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Character embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Output layer
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(lstm_output_dim, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len)
        
        Returns:
            logits: Output tensor of shape (batch_size, seq_len, vocab_size)
        """
        # Embed characters
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        embedded = self.dropout(embedded)
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(embedded)  # (batch_size, seq_len, hidden_dim * 2)
        lstm_out = self.dropout(lstm_out)
        
        # Project to vocabulary
        logits = self.fc(lstm_out)  # (batch_size, seq_len, vocab_size)
        
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
    print("Testing LSTM Cipher Model...")
    
    model = LSTMCipherModel(
        vocab_size=27,
        embedding_dim=128,
        hidden_dim=256,
        num_layers=2,
        dropout=0.3,
        bidirectional=True
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
    
    print("\n✓ LSTM model test passed!")