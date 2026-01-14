"""
MORPH - CONSTRUCTED LANGUAGE
CNN Model for Cipher Decryption
Convolutional Neural Network with multiple kernel sizes
"""

import torch
import torch.nn as nn


class CNNCipherModel(nn.Module):
    """
    CNN architecture for cipher decryption
    - Character-level embeddings
    - Multiple convolutional layers with different kernel sizes
    - Character prediction at each position
    """
    
    def __init__(self, vocab_size=27, embedding_dim=128, num_filters=256, 
                 kernel_sizes=[3, 5, 7], dropout=0.3):
        """
        Args:
            vocab_size: Size of vocabulary (26 letters + space = 27)
            embedding_dim: Dimension of character embeddings
            num_filters: Number of filters per kernel size
            kernel_sizes: List of kernel sizes
            dropout: Dropout probability
        """
        super(CNNCipherModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Character embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Multiple convolutional layers with different kernel sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=num_filters,
                kernel_size=k,
                padding=k//2  # Same padding
            )
            for k in kernel_sizes
        ])
        
        # Combine all conv outputs
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        combined_dim = num_filters * len(kernel_sizes)
        self.fc = nn.Linear(combined_dim, vocab_size)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len)
        
        Returns:
            logits: Output tensor of shape (batch_size, seq_len, vocab_size)
        """
        # Embed characters
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # Transpose for conv1d: (batch_size, embedding_dim, seq_len)
        embedded = embedded.transpose(1, 2)
        
        # Apply multiple convolutions
        conv_outputs = []
        for conv in self.convs:
            conv_out = conv(embedded)  # (batch_size, num_filters, seq_len)
            conv_out = torch.relu(conv_out)
            conv_outputs.append(conv_out)
        
        # Concatenate all conv outputs
        combined = torch.cat(conv_outputs, dim=1)  # (batch_size, combined_dim, seq_len)
        
        # Transpose back: (batch_size, seq_len, combined_dim)
        combined = combined.transpose(1, 2)
        combined = self.dropout(combined)
        
        # Project to vocabulary
        logits = self.fc(combined)  # (batch_size, seq_len, vocab_size)
        
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
    print("Testing CNN Cipher Model...")
    
    model = CNNCipherModel(
        vocab_size=27,
        embedding_dim=128,
        num_filters=256,
        kernel_sizes=[3, 5, 7],
        dropout=0.3
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
    
    print("\n✓ CNN model test passed!")