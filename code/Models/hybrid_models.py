"""
Neural Network Models for HYBRID Factorial Cipher Decryption
"""

import torch
import torch.nn as nn


class MLPCipherModel(nn.Module):
    """MLP architecture for cipher decryption"""
    
    def __init__(self, vocab_size=27, embedding_dim=128, hidden_dims=[512, 256, 128]):
        super(MLPCipherModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        layers = []
        in_dim = embedding_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            in_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_dims[-1], vocab_size)
    
    def forward(self, x):
        embedded = self.embedding(x)
        hidden = self.mlp(embedded)
        logits = self.output(hidden)
        return logits


class LSTMCipherModel(nn.Module):
    """LSTM architecture for cipher decryption"""
    
    def __init__(self, vocab_size=27, embedding_dim=128, hidden_dim=256, num_layers=2, bidirectional=True):
        super(LSTMCipherModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.3 if num_layers > 1 else 0
        )
        
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.output = nn.Linear(lstm_output_dim, vocab_size)
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        logits = self.output(lstm_out)
        return logits


class CNNCipherModel(nn.Module):
    """CNN architecture for cipher decryption"""
    
    def __init__(self, vocab_size=27, embedding_dim=128, num_filters=256, kernel_sizes=[3, 5, 7]):
        super(CNNCipherModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, k, padding=k//2)
            for k in kernel_sizes
        ])
        
        self.dropout = nn.Dropout(0.3)
        combined_dim = num_filters * len(kernel_sizes)
        self.fc = nn.Linear(combined_dim, vocab_size)
    
    def forward(self, x):
        embedded = self.embedding(x)
        embedded = embedded.transpose(1, 2)
        
        conv_outputs = []
        for conv in self.convs:
            conv_out = torch.relu(conv(embedded))
            conv_outputs.append(conv_out)
        
        combined = torch.cat(conv_outputs, dim=1)
        combined = combined.transpose(1, 2)
        combined = self.dropout(combined)
        logits = self.fc(combined)
        
        return logits


class TransformerCipherModel(nn.Module):
    """Transformer architecture for cipher decryption"""
    
    def __init__(self, vocab_size=27, d_model=256, nhead=8, num_layers=4):
        super(TransformerCipherModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 512, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        seq_len = x.size(1)
        embedded = self.embedding(x)
        embedded = embedded + self.pos_encoder[:, :seq_len, :]
        
        transformer_out = self.transformer(embedded)
        logits = self.output(transformer_out)
        
        return logits