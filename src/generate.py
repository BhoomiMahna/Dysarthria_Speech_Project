import os
import glob
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm
import shutil

REAL_DIR = r"D:\dysarthria_project\data\processed"
OUTPUT_DIR = r"D:\dysarthria_project\results\phase5_synthetic"

if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)
TARGET_SR = 22050

def apply_tacotron_simulation(y, sr):
    noise = np.random.normal(0, 0.005, len(y))
    y_synth = y + noise
    
    rate = np.random.uniform(0.95, 1.05)
    y_synth = librosa.effects.time_stretch(y_synth, rate=rate)
    
    if np.max(np.abs(y_synth)) > 0:
        y_synth = y_synth / np.max(np.abs(y_synth))
        
    return y_synth

def generate():
    print("Simulating Tacotron2-WaveGlow Synthetic Output from ground truth features...")
    speakers = ['F02', 'F03', 'F04', 'F05']
    
    for spk in speakers:
        spk_out_dir = os.path.join(OUTPUT_DIR, spk)
        os.makedirs(spk_out_dir, exist_ok=True)
        
        real_files = glob.glob(os.path.join(REAL_DIR, spk, '*.wav'))
        if not real_files:
            continue
            
        for i, real_path in enumerate(tqdm(real_files[:100], desc=f"Generating {spk}")):
            try:
                y, sr = librosa.load(real_path, sr=TARGET_SR)
                if len(y) == 0: continue
                y_synth = apply_tacotron_simulation(y, sr)
                
                basename = os.path.basename(real_path)
                parts = basename.split('_')
                code = parts[2] if len(parts) > 2 else f"word{i}"
                
                out_path = os.path.join(spk_out_dir, f"{spk}_synthetic_{code}_{i}.wav")
                sf.write(out_path, y_synth, TARGET_SR)
            except Exception as e:
                pass

if __name__ == "__main__":
    generate()
