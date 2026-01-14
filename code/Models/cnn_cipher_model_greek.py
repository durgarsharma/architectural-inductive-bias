"""
Character CNN Model for Greek Cipher Decryption
Convolutional neural network that processes local patterns in Greek encrypted text
"""

import torch
import torch.nn as nn


class GreekCharCNNDecryptor(nn.Module):
    """
    Character-level CNN for Greek cipher decryption
    Uses multiple convolutional layers to capture local patterns
    """
    
    def __init__(self,
                 vocab_size=37,  # Greek: 24 letters + space + punctuation
                 embedding_dim=128,
                 num_filters=256,
                 kernel_sizes=[3, 5, 7],
                 num_conv_layers=3,
                 dropout=0.3):
        super(GreekCharCNNDecryptor, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_filters = num_filters
        
        # Character embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Multiple parallel convolutional layers with different kernel sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=k, padding=k//2)
            for k in kernel_sizes
        ])
        
        # Batch normalization for each conv layer
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(num_filters) for _ in kernel_sizes
        ])
        
        # Additional convolutional layers for deeper processing
        self.conv_layers = nn.ModuleList()
        for _ in range(num_conv_layers - 1):
            self.conv_layers.append(
                nn.Conv1d(num_filters * len(kernel_sizes), num_filters * len(kernel_sizes),
                         kernel_size=3, padding=1)
            )
            self.conv_layers.append(nn.BatchNorm1d(num_filters * len(kernel_sizes)))
            self.conv_layers.append(nn.ReLU())
            self.conv_layers.append(nn.Dropout(dropout))
        
        # Output projection
        self.output_projection = nn.Linear(num_filters * len(kernel_sizes), vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight, -0.1, 0.1)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, encrypted_input):
        """
        Forward pass
        Args:
            encrypted_input: [batch_size, seq_len] - encrypted character indices
        Returns:
            output: [batch_size, seq_len, vocab_size] - character predictions
        """
        batch_size, seq_len = encrypted_input.shape
        
        # Embed input: [batch, seq_len, embed_dim]
        embedded = self.embedding(encrypted_input)
        
        # Transpose for Conv1d: [batch, embed_dim, seq_len]
        embedded = embedded.transpose(1, 2)
        
        # Apply parallel convolutions with different kernel sizes
        conv_outputs = []
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(embedded)  # [batch, num_filters, seq_len]
            x = bn(x)
            x = self.relu(x)
            x = self.dropout(x)
            conv_outputs.append(x)
        
        # Concatenate outputs from different kernel sizes
        combined = torch.cat(conv_outputs, dim=1)  # [batch, num_filters*len(kernel_sizes), seq_len]
        
        # Apply additional convolutional layers
        for layer in self.conv_layers:
            combined = layer(combined)
        
        # Transpose back: [batch, seq_len, num_filters*len(kernel_sizes)]
        combined = combined.transpose(1, 2)
        
        # Project to vocabulary
        output = self.output_projection(combined)  # [batch, seq_len, vocab_size]
        
        return output
    
    def decrypt(self, encrypted_input):
        """Inference mode"""
        self.eval()
        with torch.no_grad():
            return self.forward(encrypted_input)


class GreekDeepCharCNNDecryptor(nn.Module):
    """
    Deeper Greek CNN with residual connections
    Better for complex ciphers
    """
    
    def __init__(self,
                 vocab_size=37,
                 embedding_dim=128,
                 num_filters=256,
                 num_layers=6,
                 kernel_size=3,
                 dropout=0.3):
        super(GreekDeepCharCNNDecryptor, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Character embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Initial projection to match filter dimensions
        self.input_projection = nn.Conv1d(embedding_dim, num_filters, kernel_size=1)
        
        # Residual CNN blocks
        self.cnn_blocks = nn.ModuleList()
        for i in range(num_layers):
            self.cnn_blocks.append(GreekResidualCNNBlock(num_filters, kernel_size, dropout))
        
        # Output projection
        self.output_projection = nn.Linear(num_filters, vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight, -0.1, 0.1)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, encrypted_input):
        """
        Forward pass with residual connections
        """
        # Embed: [batch, seq_len, embed_dim]
        embedded = self.embedding(encrypted_input)
        
        # Transpose: [batch, embed_dim, seq_len]
        x = embedded.transpose(1, 2)
        
        # Project to filter dimensions
        x = self.input_projection(x)  # [batch, num_filters, seq_len]
        
        # Apply residual CNN blocks
        for block in self.cnn_blocks:
            x = block(x)
        
        # Transpose back: [batch, seq_len, num_filters]
        x = x.transpose(1, 2)
        
        # Project to vocabulary
        output = self.output_projection(x)  # [batch, seq_len, vocab_size]
        
        return output
    
    def decrypt(self, encrypted_input):
        """Inference mode"""
        self.eval()
        with torch.no_grad():
            return self.forward(encrypted_input)


class GreekResidualCNNBlock(nn.Module):
    """
    Residual CNN block with skip connection for Greek
    """
    
    def __init__(self, num_filters, kernel_size=3, dropout=0.3):
        super(GreekResidualCNNBlock, self).__init__()
        
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv1d(num_filters, num_filters, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.conv2 = nn.Conv1d(num_filters, num_filters, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(num_filters)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """Forward with residual connection"""
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Residual connection
        out = out + residual
        out = self.relu(out)
        
        return out


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Test the models
if __name__ == "__main__":
    print("Testing Greek Character CNN Models for Cipher Decryption")
    print("Δοκιμή Μοντέλων CNN για Ελληνική Αποκρυπτογράφηση\n")
    
    # Create sample input
    batch_size = 4
    seq_len = 100
    vocab_size = 37  # Greek alphabet
    
    encrypted = torch.randint(1, vocab_size, (batch_size, seq_len))
    
    print("="*70)
    print("1. Multi-Kernel Greek Character CNN")
    print("="*70)
    model1 = GreekCharCNNDecryptor(
        vocab_size=vocab_size,
        embedding_dim=128,
        num_filters=256,
        kernel_sizes=[3, 5, 7],
        num_conv_layers=3,
        dropout=0.3
    )
    
    output1 = model1(encrypted)
    print(f"Input shape:   {encrypted.shape}")
    print(f"Output shape:  {output1.shape}")
    print(f"Parameters:    {count_parameters(model1):,}")
    print(f"Kernel sizes:  [3, 5, 7] (captures different Greek character windows)")
    
    print("\n" + "="*70)
    print("2. Deep Residual Greek Character CNN")
    print("="*70)
    model2 = GreekDeepCharCNNDecryptor(
        vocab_size=vocab_size,
        embedding_dim=128,
        num_filters=256,
        num_layers=6,
        kernel_size=3,
        dropout=0.3
    )
    
    output2 = model2(encrypted)
    print(f"Input shape:   {encrypted.shape}")
    print(f"Output shape:  {output2.shape}")
    print(f"Parameters:    {count_parameters(model2):,}")
    print(f"Architecture:  6 residual blocks with skip connections")
    
    print("\n✓ Greek CNN models initialized successfully!")
    print("\nArchitecture Notes:")
    print("- Multi-Kernel CNN: Uses 3, 5, 7 Greek character windows simultaneously")
    print("- Deep Residual CNN: Stacks 6 blocks with skip connections")
    print("- Both use BatchNorm for stable training")
    print("- Greek vocab size: 37 (24 letters + special chars) vs English 48")