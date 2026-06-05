import os
import glob
import librosa
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

DATA_DIR = r"C:\Users\DELL\My Drive\UASpeech_Project"
OUTPUT_DIR = r"D:\dysarthria_project\data\processed"
RESULTS_DIR = r"D:\dysarthria_project\results\phase2"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

TARGET_SR = 22050
TOP_DB = 30 # For silence trimming

def extract_mel_spectrogram(y, sr):
    # Standard VITS mel-spectrogram parameters
    n_fft = 1024
    hop_length = 256
    win_length = 1024
    n_mels = 80
    fmin = 0
    fmax = 8000
    
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, 
        win_length=win_length, n_mels=n_mels, fmin=fmin, fmax=fmax
    )
    # Convert to log scale
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db

def plot_and_save(y, S_db, sr, speaker, word, out_path):
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    
    # Waveform
    time = np.arange(len(y)) / sr
    ax[0].plot(time, y, color='blue', alpha=0.7)
    ax[0].set_title(f'Waveform: {speaker} - {word}')
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Amplitude')
    
    # Mel Spectrogram
    img = librosa.display.specshow(S_db, x_axis='time', y_axis='mel', sr=sr,
                                   fmax=8000, ax=ax[1], cmap='magma')
    fig.colorbar(img, ax=ax[1], format='%+2.0f dB')
    ax[1].set_title(f'Mel Spectrogram: {speaker} - {word}')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def preprocess_dataset():
    audio_files = []
    for spk in ['F02', 'F03', 'F04', 'F05']:
        spk_dir = os.path.join(DATA_DIR, spk)
        if os.path.exists(spk_dir):
            audio_files.extend(glob.glob(os.path.join(spk_dir, '**', '*.wav'), recursive=True))
        
        spk_flat = os.path.join(DATA_DIR, f"UASpeech_{spk}_Project", "audio")
        if os.path.exists(spk_flat):
             audio_files.extend(glob.glob(os.path.join(spk_flat, '*.wav')))
             
    audio_files = list(set(audio_files))
    
    print(f"Preprocessing {len(audio_files)} files...")
    
    # We will only generate plots for the first file of each speaker to save time
    plotted_speakers = set()
    
    for f in tqdm(audio_files):
        basename = os.path.basename(f)
        parts = os.path.splitext(basename)[0].split('_')
        if len(parts) < 3:
            continue
        speaker = parts[0]
        word = parts[2]
        
        # Output paths
        spk_out_dir = os.path.join(OUTPUT_DIR, speaker)
        os.makedirs(spk_out_dir, exist_ok=True)
        out_wav = os.path.join(spk_out_dir, basename)
        out_mel = os.path.join(spk_out_dir, basename.replace('.wav', '.npy'))
        
        # Load and resample
        y, sr = librosa.load(f, sr=TARGET_SR)
        
        # Silence trimming
        yt, _ = librosa.effects.trim(y, top_db=TOP_DB)
        
        # Loudness normalization (peak normalization)
        if len(yt) > 0:
            yt = yt / np.max(np.abs(yt))
        
        # Save processed wav
        sf.write(out_wav, yt, TARGET_SR)
        
        # Extract Mel
        S_db = extract_mel_spectrogram(yt, TARGET_SR)
        np.save(out_mel, S_db)
        
        # Plot for one example per speaker
        if speaker not in plotted_speakers:
            plot_path = os.path.join(RESULTS_DIR, f"preprocess_example_{speaker}_{word}.png")
            plot_and_save(yt, S_db, TARGET_SR, speaker, word, plot_path)
            plotted_speakers.add(speaker)

if __name__ == "__main__":
    preprocess_dataset()
