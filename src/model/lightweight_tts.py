import torch
import torch.nn as nn
import torch.nn.functional as F

class LightweightTTS(nn.Module):
    """
    A lightweight Convolutional TTS model suitable for isolated words.
    Predicts Mel Spectrogram from Word Embedding + Speaker Embedding.
    """
    def __init__(self, num_words, num_speakers, embed_dim=128, mel_channels=80, max_seq_len=200):
        super(LightweightTTS, self).__init__()
        self.mel_channels = mel_channels
        self.max_seq_len = max_seq_len
        
        self.word_emb = nn.Embedding(num_words, embed_dim)
        self.spk_emb = nn.Embedding(num_speakers, embed_dim)
        
        # Generator: Transposed Convolutions to upsample to max_seq_len
        self.decoder = nn.Sequential(
            nn.Conv1d(embed_dim * 2, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Conv1d(256, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Conv1d(256, mel_channels, kernel_size=1)
        )
        
    def forward(self, word_idx, spk_idx):
        # word_idx: [batch_size]
        # spk_idx: [batch_size]
        
        w_e = self.word_emb(word_idx) # [B, embed_dim]
        s_e = self.spk_emb(spk_idx) # [B, embed_dim]
        
        x = torch.cat([w_e, s_e], dim=1) # [B, embed_dim * 2]
        
        # Expand to sequence length
        x = x.unsqueeze(-1).expand(-1, -1, self.max_seq_len) # [B, embed_dim * 2, max_seq_len]
        
        mel_pred = self.decoder(x) # [B, mel_channels, max_seq_len]
        return mel_pred
