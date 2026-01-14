"""
LSTM Model for Greek Cipher Decryption
Character-level sequence-to-sequence LSTM for Greek alphabet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GreekCharLSTMDecryptor(nn.Module):
    """
    Character-level LSTM for Greek cipher decryption
    Architecture: Encoder-Decoder with attention
    """
    
    def __init__(self, 
                 vocab_size=37,  # Greek: 24 letters + space + punctuation
                 embedding_dim=128,
                 hidden_dim=256,
                 num_layers=2,
                 dropout=0.3):
        super(GreekCharLSTMDecryptor, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Character embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Encoder LSTM (processes encrypted text)
        self.encoder_lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Decoder LSTM (generates plaintext)
        self.decoder_lstm = nn.LSTM(
            embedding_dim,
            hidden_dim * 2,  # Match bidirectional encoder output
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        
        # Output projection layer
        self.output_projection = nn.Linear(hidden_dim * 2, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, encrypted_input, target_input=None, teacher_forcing_ratio=0.5):
        """
        Forward pass
        Args:
            encrypted_input: [batch_size, seq_len] - encrypted character indices
            target_input: [batch_size, seq_len] - target plaintext (for training)
            teacher_forcing_ratio: probability of using ground truth as input
        Returns:
            output: [batch_size, seq_len, vocab_size] - character predictions
        """
        batch_size = encrypted_input.size(0)
        seq_len = encrypted_input.size(1)
        
        # Embed encrypted input
        embedded = self.embedding(encrypted_input)  # [batch, seq_len, embed_dim]
        embedded = self.dropout(embedded)
        
        # Encode
        encoder_outputs, (hidden, cell) = self.encoder_lstm(embedded)
        # encoder_outputs: [batch, seq_len, hidden_dim*2]
        
        # Initialize decoder hidden state
        decoder_hidden = hidden[-self.num_layers:]
        decoder_cell = cell[-self.num_layers:]
        
        # Expand to match decoder's hidden size
        decoder_hidden = decoder_hidden.repeat(1, 1, 2)
        decoder_cell = decoder_cell.repeat(1, 1, 2)
        
        # Decoder
        outputs = []
        decoder_input = embedded[:, 0:1, :]
        
        for t in range(seq_len):
            decoder_output, (decoder_hidden, decoder_cell) = self.decoder_lstm(
                decoder_input,
                (decoder_hidden, decoder_cell)
            )
            
            output = self.output_projection(decoder_output)
            outputs.append(output)
            
            # Teacher forcing
            if target_input is not None and torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = self.embedding(target_input[:, t:t+1])
            else:
                predicted_char = output.argmax(dim=-1)
                decoder_input = self.embedding(predicted_char)
            
            decoder_input = self.dropout(decoder_input)
        
        outputs = torch.cat(outputs, dim=1)
        
        return outputs
    
    def decrypt(self, encrypted_input):
        """Decrypt without teacher forcing (inference mode)"""
        self.eval()
        with torch.no_grad():
            return self.forward(encrypted_input, target_input=None, teacher_forcing_ratio=0.0)


class GreekSimpleLSTMDecryptor(nn.Module):
    """
    Simpler LSTM model for Greek without encoder-decoder architecture
    Direct sequence-to-sequence mapping
    """
    
    def __init__(self,
                 vocab_size=37,
                 embedding_dim=128,
                 hidden_dim=256,
                 num_layers=2,
                 dropout=0.3):
        super(GreekSimpleLSTMDecryptor, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.output_layer = nn.Linear(hidden_dim * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, encrypted_input):
        """Forward pass - direct mapping from encrypted to plaintext"""
        # Embed
        embedded = self.embedding(encrypted_input)
        embedded = self.dropout(embedded)
        
        # LSTM
        lstm_out, _ = self.lstm(embedded)
        lstm_out = self.dropout(lstm_out)
        
        # Project to vocabulary
        output = self.output_layer(lstm_out)
        
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
    print("Testing Greek LSTM Models for Cipher Decryption")
    print("Δοκιμή Μοντέλων LSTM για Ελληνική Αποκρυπτογράφηση\n")
    
    # Create sample input
    batch_size = 4
    seq_len = 100
    vocab_size = 37  # Greek alphabet
    
    encrypted = torch.randint(1, vocab_size, (batch_size, seq_len))
    target = torch.randint(1, vocab_size, (batch_size, seq_len))
    
    print("="*70)
    print("1. Encoder-Decoder LSTM for Greek")
    print("="*70)
    model1 = GreekCharLSTMDecryptor(
        vocab_size=vocab_size,
        embedding_dim=128,
        hidden_dim=256,
        num_layers=2
    )
    
    output1 = model1(encrypted, target)
    print(f"Input shape:  {encrypted.shape}")
    print(f"Output shape: {output1.shape}")
    print(f"Parameters:   {count_parameters(model1):,}")
    
    print("\n" + "="*70)
    print("2. Simple Bidirectional LSTM for Greek")
    print("="*70)
    model2 = GreekSimpleLSTMDecryptor(
        vocab_size=vocab_size,
        embedding_dim=128,
        hidden_dim=256,
        num_layers=2
    )
    
    output2 = model2(encrypted)
    print(f"Input shape:  {encrypted.shape}")
    print(f"Output shape: {output2.shape}")
    print(f"Parameters:   {count_parameters(model2):,}")
    
    print("\n✓ Greek LSTM models initialized successfully!")
    print("\nRecommendation: Start with GreekSimpleLSTMDecryptor (faster, easier to train)")
    print("Greek vocab size: 37 (24 letters + special chars) vs English 48")