"""
MORPH - CONSTRUCTED LANGUAGE
MLP Model for Cipher Decryption
Multi-Layer Perceptron with character-level embeddings
"""

import torch
import torch.nn as nn


class MLPCipherModel(nn.Module):
    """
    MLP architecture for cipher decryption
    - Character-level embeddings
    - Multiple fully connected layers
    - Character prediction at each position
    """
    
    def __init__(self, vocab_size=27, embedding_dim=128, hidden_dims=[512, 256, 128]):
        """
        Args:
            vocab_size: Size of vocabulary (26 letters + space = 27)
            embedding_dim: Dimension of character embeddings
            hidden_dims: List of hidden layer dimensions
        """
        super(MLPCipherModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Character embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Build MLP layers
        layers = []
        input_dim = embedding_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, vocab_size))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len)
        
        Returns:
            logits: Output tensor of shape (batch_size, seq_len, vocab_size)
        """
        # Embed characters
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # Apply MLP to each position independently
        batch_size, seq_len, embed_dim = embedded.shape
        
        # Reshape to process all positions together
        embedded_flat = embedded.view(-1, embed_dim)  # (batch_size * seq_len, embedding_dim)
        
        # Pass through MLP
        logits_flat = self.mlp(embedded_flat)  # (batch_size * seq_len, vocab_size)
        
        # Reshape back
        logits = logits_flat.view(batch_size, seq_len, self.vocab_size)
        
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
    print("Testing MLP Cipher Model...")
    
    model = MLPCipherModel(
        vocab_size=27,
        embedding_dim=128,
        hidden_dims=[512, 256, 128]
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
    
    print("\n✓ MLP model test passed!")