import os
import glob
import librosa
import numpy as np
import matplotlib.pyplot as plt

REAL_DIR = r"D:\dysarthria_project\data\processed"
SYNTH_DIR = r"D:\dysarthria_project\results\phase5_synthetic"
RESULTS_DIR = r"D:\dysarthria_project\results\phase8"

os.makedirs(RESULTS_DIR, exist_ok=True)
TARGET_SR = 22050

# Target words from the user's reference image
TARGET_WORDS = ['lamp', 'he', 'buffoon', 'technology', 'or', 'what']
CODE_TO_WORD = {
    'CW42': 'lamp', 'UW24': 'he', 'CW10': 'buffoon', 
    'CW96': 'technology', 'UW67': 'or', 'UW98': 'what'
}
WORD_TO_CODE = {v: k for k, v in CODE_TO_WORD.items()}

def extract_mfcc(y, sr):
    return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

def extract_mel(y, sr):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80)
    return librosa.power_to_db(S, ref=np.max)

def evaluate_dtw():
    if not os.path.exists(SYNTH_DIR):
        print("No synthetic data found.")
        return

    print("Generating 2-pane DTW plots for specific target words...")
    
    for speaker in ['F02', 'F03', 'F05']: # From user's reference
        for word in TARGET_WORDS:
            code = WORD_TO_CODE.get(word, word)
            synth_matches = glob.glob(os.path.join(SYNTH_DIR, speaker, f"*{code}*.wav"))
            real_matches = glob.glob(os.path.join(REAL_DIR, speaker, f"{speaker}_*_{code}_*.wav"))
            
            if not synth_matches or not real_matches:
                continue
                
            y_synth, _ = librosa.load(synth_matches[0], sr=TARGET_SR)
            y_real, _ = librosa.load(real_matches[0], sr=TARGET_SR)
            
            mfcc_synth = extract_mfcc(y_synth, TARGET_SR)
            mfcc_real = extract_mfcc(y_real, TARGET_SR)
            
            D, wp = librosa.sequence.dtw(X=mfcc_real, Y=mfcc_synth, metric='cosine')
            
            # The top pane is typically the Generated Mel Spectrogram, with the alignment path overlaid.
            mel_synth = extract_mel(y_synth, TARGET_SR)
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5), gridspec_kw={'height_ratios': [2, 1]})
            
            # Top pane: Spectrogram + Path
            img = librosa.display.specshow(mel_synth, x_axis='time', y_axis='mel', sr=TARGET_SR, ax=ax1, cmap='viridis')
            
            # The DTW path wp contains (x_index, y_index). We need to map frame indices to time.
            # librosa default hop length is 512
            times_x = librosa.frames_to_time(wp[:, 1], sr=TARGET_SR)
            # We want to overlay the path. In standard specshow, y is mel bins, not time.
            # If the paper overlayed path on spectrogram, they probably scaled Y to fit the height (0 to 80 bins).
            # wp[:, 0] is the Real sequence frames.
            path_y_scaled = (wp[:, 0] / np.max(wp[:, 0])) * 8192 # scaling to fmax ~8000Hz for mel axis visual overlap
            ax1.plot(times_x, path_y_scaled, color='red', linewidth=1.5, alpha=0.8)
            ax1.set_xlabel('Original') # X-axis was labeled "Original" in reference
            ax1.set_ylabel('Generated') # Y-axis was labeled "Generated"
            ax1.set_title(f'{speaker} ("{word}") | Score: {D[-1,-1]/sum(D.shape):.4f}', fontsize=8)
            
            # Bottom pane: Overlapping Waveforms
            time_real = np.arange(len(y_real)) / TARGET_SR
            time_synth = np.arange(len(y_synth)) / TARGET_SR
            
            ax2.plot(time_real, y_real, color='yellowgreen', alpha=0.8, label='Original')
            ax2.plot(time_synth, y_synth, color='indigo', alpha=0.6, label='Generated')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Amplitude (dB)')
            ax2.legend(loc='upper right', fontsize=6)
            
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, f'dtw_plot_{speaker}_{word}.png'), dpi=300)
            plt.close()
            
    print("DTW specific plots generated.")

if __name__ == "__main__":
    evaluate_dtw()
