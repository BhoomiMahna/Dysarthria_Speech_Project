import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cosine
import librosa
from tqdm import tqdm

REAL_DIR = r"D:\dysarthria_project\data\processed"
SYNTH_DIR = r"D:\dysarthria_project\results\phase5_synthetic"
RESULTS_DIR = r"D:\dysarthria_project\results\phase7"

os.makedirs(RESULTS_DIR, exist_ok=True)

def extract_embedding(filepath):
    signal, fs = librosa.load(filepath, sr=16000)
    mfcc = librosa.feature.mfcc(y=signal, sr=fs, n_mfcc=20)
    return np.mean(mfcc, axis=1)

def evaluate_speaker():
    if not os.path.exists(SYNTH_DIR):
        print("No synthetic data found. Run Phase 5 first.")
        return
        
    speakers = ['F02', 'F03', 'F04', 'F05']
    
    # Extract representative embeddings for real speakers
    print("Extracting real speaker profiles using MFCC...")
    real_profiles = {}
    for spk in speakers:
        spk_files = glob.glob(os.path.join(REAL_DIR, spk, '*.wav'))[:20]
        if not spk_files: continue
        
        embs = []
        for f in spk_files:
            embs.append(extract_embedding(f))
            
        real_profiles[spk] = np.mean(embs, axis=0)
        
    print("Evaluating synthetic speaker preservation...")
    
    similarity_matrix = np.zeros((len(speakers), len(speakers)))
    
    for i, true_spk in enumerate(speakers):
        synth_files = glob.glob(os.path.join(SYNTH_DIR, true_spk, '*.wav'))
        if not synth_files: continue
        
        synth_embs = []
        for f in synth_files[:50]:
            synth_embs.append(extract_embedding(f))
            
        if not synth_embs: continue
        avg_synth_emb = np.mean(synth_embs, axis=0)
        
        # Compare against all real profiles
        for j, test_spk in enumerate(speakers):
            if test_spk in real_profiles:
                sim = 1 - cosine(avg_synth_emb, real_profiles[test_spk])
                similarity_matrix[i, j] = sim
                
    # Plot Similarity Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(similarity_matrix, annot=True, xticklabels=speakers, yticklabels=speakers, cmap='Blues')
    plt.title('Speaker Similarity (Synthetic vs Real Profiles)')
    plt.ylabel('Synthetic Speaker')
    plt.xlabel('Real Profile')
    plt.savefig(os.path.join(RESULTS_DIR, 'speaker_similarity_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Phase 7 speaker preservation evaluation complete.")

if __name__ == "__main__":
    evaluate_speaker()
