import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import librosa

REAL_DIR = r"D:\dysarthria_project\data\processed"
SYNTH_DIR = r"D:\dysarthria_project\results\phase5_synthetic"
RESULTS_DIR = r"D:\dysarthria_project\results\phase9"

os.makedirs(RESULTS_DIR, exist_ok=True)
TARGET_SR = 16000

# Words to annotate in the t-SNE plot to match the reference
WORDS_TO_ANNOTATE = ['equilibrium', 'technology', 'lamp', 'he', 'what', 'caterpillar', 'haranguing', 'three']
CODE_TO_WORD = {
    'CW22': 'equilibrium', 'CW96': 'technology', 'CW42': 'lamp',
    'UW24': 'he', 'UW98': 'what', 'CW15': 'caterpillar', 
    'CW39': 'haranguing', 'UW93': 'three'
}

# Color mapping requested
COLOR_MAP = {
    'F02': '#5c73a4', # blue
    'F03': '#cf9053', # orange
    'F04': '#68a169', # green
    'F05': '#b54d52'  # red
}

def extract_embedding(filepath):
    signal, fs = librosa.load(filepath, sr=TARGET_SR)
    mfcc = librosa.feature.mfcc(y=signal, sr=fs, n_mfcc=20)
    return np.mean(mfcc, axis=1)

def evaluate_clustering():
    if not os.path.exists(SYNTH_DIR):
        print("No synthetic data found.")
        return
        
    speakers = ['F02', 'F03', 'F04', 'F05']
    
    embeddings = []
    metadata = []
    
    print("Extracting embeddings for Real Data...")
    for spk in speakers:
        files = glob.glob(os.path.join(REAL_DIR, spk, '*.wav'))[:80]
        for f in files:
            emb = extract_embedding(f)
            basename = os.path.basename(f)
            code = basename.split('_')[2] if len(basename.split('_')) > 2 else ""
            word = CODE_TO_WORD.get(code, "")
            embeddings.append(emb)
            metadata.append({'Speaker': spk, 'Source': 'Original', 'Word': word})
            
    print("Extracting embeddings for Synthetic Data...")
    for spk in speakers:
        files = glob.glob(os.path.join(SYNTH_DIR, spk, '*.wav'))[:80]
        for f in files:
            emb = extract_embedding(f)
            basename = os.path.basename(f)
            code = basename.split('_')[2] if len(basename.split('_')) > 2 else ""
            word = CODE_TO_WORD.get(code, "")
            embeddings.append(emb)
            metadata.append({'Speaker': spk, 'Source': 'Generated', 'Word': word})
            
    if not embeddings: return
    
    embeddings = np.array(embeddings)
    
    print("Computing t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embs_2d = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    
    # Plotting based on reference styling
    for i, data in enumerate(metadata):
        spk = data['Speaker']
        src = data['Source']
        word = data['Word']
        
        c = COLOR_MAP[spk]
        m = 'o' if src == 'Original' else 'x'
        s = 60 if src == 'Original' else 80
        lw = 2 if src == 'Generated' else 1
        
        plt.scatter(embs_2d[i, 0], embs_2d[i, 1], c=c, marker=m, s=s, linewidths=lw, alpha=0.8)
        
        # Annotate specific words near the center like in the reference Fig 10
        if word in WORDS_TO_ANNOTATE and src == 'Original' and np.random.rand() > 0.8:
            # We add a little random chance so we don't annotate every single file if there are duplicates
            plt.annotate(word, (embs_2d[i, 0], embs_2d[i, 1]), fontsize=8, alpha=0.7, xytext=(5, 5), textcoords='offset points')
            WORDS_TO_ANNOTATE.remove(word) # Only annotate once

    plt.title('Multi-Speaker t-SNE: Word Clusters (Circle=Original, Cross=Generated)')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#5c73a4', markersize=8, label='F02(low)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#cf9053', markersize=8, label='F03(very low)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#68a169', markersize=8, label='F04(mid)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#b54d52', markersize=8, label='F05(high)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', markersize=8, label='Original'),
        Line2D([0], [0], marker='x', color='w', markeredgecolor='grey', markersize=8, markeredgewidth=2, label='Generated')
    ]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.02, 1), loc='upper left', title='Speaker / Source')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'tsne_multi_speaker.png'), dpi=300)
    plt.close()
    print("Customized t-SNE plot complete.")

if __name__ == "__main__":
    evaluate_clustering()
