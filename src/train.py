import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from model.lightweight_tts import LightweightTTS

DATA_DIR = r"D:\dysarthria_project\data\processed"
RESULTS_DIR = r"D:\dysarthria_project\results\phase4"
CHECKPOINT_DIR = r"D:\dysarthria_project\checkpoints"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

class UASpeechDataset(Dataset):
    def __init__(self, data_dir, max_seq_len=200):
        self.data_dir = data_dir
        self.max_seq_len = max_seq_len
        self.files = glob.glob(os.path.join(data_dir, '**', '*.npy'), recursive=True)
        
        self.speakers = []
        self.words = []
        
        for f in self.files:
            basename = os.path.basename(f)
            parts = os.path.splitext(basename)[0].split('_')
            self.speakers.append(parts[0])
            self.words.append(parts[2])
            
        self.unique_speakers = sorted(list(set(self.speakers)))
        self.unique_words = sorted(list(set(self.words)))
        
        self.spk2idx = {s: i for i, s in enumerate(self.unique_speakers)}
        self.word2idx = {w: i for i, w in enumerate(self.unique_words)}
        
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        f = self.files[idx]
        mel = np.load(f) # [80, time]
        
        # Pad or truncate to max_seq_len
        if mel.shape[1] > self.max_seq_len:
            mel = mel[:, :self.max_seq_len]
        elif mel.shape[1] < self.max_seq_len:
            pad = self.max_seq_len - mel.shape[1]
            mel = np.pad(mel, ((0, 0), (0, pad)), mode='constant')
            
        spk_idx = self.spk2idx[self.speakers[idx]]
        word_idx = self.word2idx[self.words[idx]]
        
        return {
            'mel': torch.FloatTensor(mel),
            'spk_idx': torch.tensor(spk_idx, dtype=torch.long),
            'word_idx': torch.tensor(word_idx, dtype=torch.long)
        }

def train():
    dataset = UASpeechDataset(DATA_DIR)
    
    if len(dataset) == 0:
        print("No processed data found. Run Phase 2 first.")
        return
        
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    num_speakers = len(dataset.unique_speakers)
    num_words = len(dataset.unique_words)
    
    model = LightweightTTS(num_words=num_words, num_speakers=num_speakers)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 10
    losses = []
    
    print(f"Training on {device} for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            mel = batch['mel'].to(device)
            spk_idx = batch['spk_idx'].to(device)
            word_idx = batch['word_idx'].to(device)
            
            optimizer.zero_grad()
            
            mel_pred = model(word_idx, spk_idx)
            
            loss = criterion(mel_pred, mel)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")
        
    # Save checkpoint
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "lightweight_tts.pt"))
    
    # Save word/spk mapping for generation
    np.save(os.path.join(CHECKPOINT_DIR, "spk2idx.npy"), dataset.spk2idx)
    np.save(os.path.join(CHECKPOINT_DIR, "word2idx.npy"), dataset.word2idx)
    
    # Generate loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), losses, marker='o', color='green')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid()
    plt.savefig(os.path.join(RESULTS_DIR, 'training_loss.png'), dpi=300)
    plt.close()
    
    print("Training complete!")

if __name__ == "__main__":
    train()
