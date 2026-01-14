"""
Transformer Model for Greek Cipher Decryption
Character-level Transformer for Greek alphabet
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class GreekTransformerDecryptor(nn.Module):
    """
    Transformer model for Greek cipher decryption
    Uses encoder-decoder architecture
    """
    
    def __init__(self,
                 vocab_size=37,  # Greek: 24 letters + space + punctuation
                 d_model=256,
                 nhead=8,
                 num_encoder_layers=3,
                 num_decoder_layers=3,
                 dim_feedforward=1024,
                 dropout=0.1):
        super(GreekTransformerDecryptor, self).__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Embeddings
        self.src_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.tgt_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        initrange = 0.1
        self.src_embedding.weight.data.uniform_(-initrange, initrange)
        self.tgt_embedding.weight.data.uniform_(-initrange, initrange)
        self.output_projection.bias.data.zero_()
        self.output_projection.weight.data.uniform_(-initrange, initrange)
    
    def generate_square_subsequent_mask(self, sz):
        """Generate mask for decoder (prevents looking ahead)"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, src, tgt):
        """
        Forward pass
        Args:
            src: [batch_size, src_seq_len] - encrypted Greek text
            tgt: [batch_size, tgt_seq_len] - target plaintext (for training)
        Returns:
            output: [batch_size, tgt_seq_len, vocab_size]
        """
        # Create masks
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(src.device)
        src_padding_mask = (src == 0)  # Mask padding tokens
        tgt_padding_mask = (tgt == 0)
        
        # Embed and add positional encoding
        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        
        # Transformer forward pass
        output = self.transformer(
            src_emb,
            tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )
        
        # Project to vocabulary
        output = self.output_projection(output)
        
        return output
    
    def decrypt(self, src, max_len=None):
        """
        Decrypt without ground truth (inference mode)
        Uses greedy decoding
        """
        self.eval()
        
        if max_len is None:
            max_len = src.size(1)
        
        batch_size = src.size(0)
        device = src.device
        
        # Start with padding token
        tgt = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        
        with torch.no_grad():
            for i in range(max_len - 1):
                # Forward pass
                output = self.forward(src, tgt)
                
                # Get next token (greedy)
                next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
                
                # Append to target
                tgt = torch.cat([tgt, next_token], dim=1)
        
        return tgt


class GreekSimpleTransformerDecryptor(nn.Module):
    """
    Simpler transformer model for Greek (encoder-only)
    Direct mapping from encrypted to plaintext
    """
    
    def __init__(self,
                 vocab_size=37,
                 d_model=256,
                 nhead=8,
                 num_layers=4,
                 dim_feedforward=1024,
                 dropout=0.1):
        super(GreekSimpleTransformerDecryptor, self).__init__()
        
        self.d_model = d_model
        
        # Embedding
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer encoder
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
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.output_projection.bias.data.zero_()
        self.output_projection.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, src):
        """
        Forward pass
        Args:
            src: [batch_size, seq_len] - encrypted Greek text
        Returns:
            output: [batch_size, seq_len, vocab_size]
        """
        # Create padding mask
        src_padding_mask = (src == 0)
        
        # Embed and add positional encoding
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)
        
        # Transformer encoding
        output = self.transformer_encoder(
            src_emb,
            src_key_padding_mask=src_padding_mask
        )
        
        # Project to vocabulary
        output = self.output_projection(output)
        
        return output
    
    def decrypt(self, src):
        """Inference mode"""
        self.eval()
        with torch.no_grad():
            return self.forward(src)


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Test the models
if __name__ == "__main__":
    print("Testing Greek Transformer Models for Cipher Decryption")
    print("Δοκιμή Μοντέλων Transformer για Ελληνική Αποκρυπτογράφηση\n")
    
    # Create sample input
    batch_size = 4
    seq_len = 100
    vocab_size = 37  # Greek alphabet
    
    encrypted = torch.randint(1, vocab_size, (batch_size, seq_len))
    target = torch.randint(1, vocab_size, (batch_size, seq_len))
    
    print("="*70)
    print("1. Encoder-Decoder Greek Transformer")
    print("="*70)
    model1 = GreekTransformerDecryptor(
        vocab_size=vocab_size,
        d_model=256,
        nhead=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=1024,
        dropout=0.1
    )
    
    output1 = model1(encrypted, target[:, :-1])  # Teacher forcing
    print(f"Input shape:  {encrypted.shape}")
    print(f"Output shape: {output1.shape}")
    print(f"Parameters:   {count_parameters(model1):,}")
    
    print("\n" + "="*70)
    print("2. Simple Encoder-Only Greek Transformer")
    print("="*70)
    model2 = GreekSimpleTransformerDecryptor(
        vocab_size=vocab_size,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=1024,
        dropout=0.1
    )
    
    output2 = model2(encrypted)
    print(f"Input shape:  {encrypted.shape}")
    print(f"Output shape: {output2.shape}")
    print(f"Parameters:   {count_parameters(model2):,}")
    
    print("\n✓ Greek Transformer models initialized successfully!")
    print("\nRecommendation: Start with GreekSimpleTransformerDecryptor")
    print("(Similar to LSTM but with attention mechanism)")
    print("Greek vocab size: 37 (24 letters + special chars) vs English 48")