"""
MLP Model for Cipher Decryption
Simple feedforward network baseline
"""

import torch
import torch.nn as nn


class MLPDecryptor(nn.Module):
    """
    Multi-Layer Perceptron for cipher decryption
    Processes each character position independently
    """
    
    def __init__(self,
                 vocab_size=38,
                 embedding_dim=128,
                 hidden_dims=[512, 256, 128],
                 dropout=0.3):
        super(MLPDecryptor, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Character embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Build MLP layers
        layers = []
        input_dim = embedding_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, vocab_size))
        
        self.mlp = nn.Sequential(*layers)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight, -0.1, 0.1)
    
    def forward(self, encrypted_input):
        """
        Forward pass - process each position independently
        Args:
            encrypted_input: [batch_size, seq_len] - encrypted character indices
        Returns:
            output: [batch_size, seq_len, vocab_size] - character predictions
        """
        # Embed input
        embedded = self.embedding(encrypted_input)  # [batch, seq_len, embed_dim]
        
        # Process each position independently through MLP
        batch_size, seq_len, embed_dim = embedded.shape
        
        # Reshape to process all positions at once
        flat_embedded = embedded.view(batch_size * seq_len, embed_dim)
        
        # Pass through MLP
        flat_output = self.mlp(flat_embedded)  # [batch*seq_len, vocab_size]
        
        # Reshape back
        output = flat_output.view(batch_size, seq_len, self.vocab_size)
        
        return output
    
    def decrypt(self, encrypted_input):
        """Inference mode"""
        self.eval()
        with torch.no_grad():
            return self.forward(encrypted_input)


class ContextualMLPDecryptor(nn.Module):
    """
    MLP with context window
    Uses neighboring characters as context
    """
    
    def __init__(self,
                 vocab_size=38,
                 embedding_dim=128,
                 context_size=5,  # Use 2 chars on each side
                 hidden_dims=[512, 256, 128],
                 dropout=0.3):
        super(ContextualMLPDecryptor, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.context_size = context_size
        
        # Character embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # MLP input includes context
        input_dim = embedding_dim * context_size
        
        # Build MLP
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, vocab_size))
        
        self.mlp = nn.Sequential(*layers)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight, -0.1, 0.1)
    
    def forward(self, encrypted_input):
        """
        Forward pass with context window
        """
        batch_size, seq_len = encrypted_input.shape
        
        # Embed
        embedded = self.embedding(encrypted_input)  # [batch, seq_len, embed_dim]
        
        # Pad sequence for context
        pad_size = self.context_size // 2
        padded = nn.functional.pad(embedded, (0, 0, pad_size, pad_size), value=0)
        
        # Extract context windows
        outputs = []
        for i in range(seq_len):
            # Get context window
            context = padded[:, i:i+self.context_size, :]  # [batch, context_size, embed_dim]
            context_flat = context.reshape(batch_size, -1)  # [batch, context_size*embed_dim]
            
            # Pass through MLP
            output = self.mlp(context_flat)  # [batch, vocab_size]
            outputs.append(output)
        
        # Stack outputs
        output = torch.stack(outputs, dim=1)  # [batch, seq_len, vocab_size]
        
        return output
    
    def decrypt(self, encrypted_input):
        """Inference mode"""
        self.eval()
        with torch.no_grad():
            return self.forward(encrypted_input)


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Test the models
if __name__ == "__main__":
    print("Testing MLP Models for Cipher Decryption\n")
    
    # Create sample input
    batch_size = 4
    seq_len = 100
    vocab_size = 36
    
    encrypted = torch.randint(1, vocab_size, (batch_size, seq_len))
    
    print("="*60)
    print("1. Simple MLP (Position-Independent)")
    print("="*60)
    model1 = MLPDecryptor(
        vocab_size=vocab_size,
        embedding_dim=128,
        hidden_dims=[512, 256, 128],
        dropout=0.3
    )
    
    output1 = model1(encrypted)
    print(f"Input shape:  {encrypted.shape}")
    print(f"Output shape: {output1.shape}")
    print(f"Parameters:   {count_parameters(model1):,}")
    
    print("\n" + "="*60)
    print("2. Contextual MLP (With Context Window)")
    print("="*60)
    model2 = ContextualMLPDecryptor(
        vocab_size=vocab_size,
        embedding_dim=128,
        context_size=5,
        hidden_dims=[512, 256, 128],
        dropout=0.3
    )
    
    output2 = model2(encrypted)
    print(f"Input shape:  {encrypted.shape}")
    print(f"Output shape: {output2.shape}")
    print(f"Parameters:   {count_parameters(model2):,}")
    
    print("\n✓ Models initialized successfully!")
    print("\nNote:")
    print("- Simple MLP: Fastest, but no context between characters")
    print("- Contextual MLP: Uses neighbor info, better for substitution ciphers")