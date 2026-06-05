import os
import glob
import soundfile as sf
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import re

DATA_DIR = r"C:\Users\DELL\My Drive\UASpeech_Project"
RESULTS_DIR = r"D:\dysarthria_project\results\phase1"

os.makedirs(RESULTS_DIR, exist_ok=True)

def parse_filename(filename):
    # UA-Speech filename format: F02_B1_CW10_M2.wav
    # Speaker: F02
    # Block: B1
    # Word: CW10 (or UW...)
    # Mic: M2
    basename = os.path.basename(filename)
    name, ext = os.path.splitext(basename)
    parts = name.split('_')
    if len(parts) >= 3:
        speaker = parts[0]
        word = parts[2]
        return speaker, word
    return None, None

def analyze_dataset():
    audio_files = []
    # Search for wav files in F02, F03, F04, F05
    for spk in ['F02', 'F03', 'F04', 'F05']:
        spk_dir = os.path.join(DATA_DIR, spk)
        if os.path.exists(spk_dir):
            audio_files.extend(glob.glob(os.path.join(spk_dir, '**', '*.wav'), recursive=True))
        
        # also check if there is a flat audio folder just in case
        spk_flat = os.path.join(DATA_DIR, f"UASpeech_{spk}_Project", "audio")
        if os.path.exists(spk_flat):
             audio_files.extend(glob.glob(os.path.join(spk_flat, '*.wav')))

    # remove duplicates
    audio_files = list(set(audio_files))
    
    if not audio_files:
        print("No audio files found! Check path.")
        return

    data = []
    
    print(f"Analyzing {len(audio_files)} files...")
    for f in tqdm(audio_files):
        speaker, word = parse_filename(f)
        if not speaker:
            continue
            
        try:
            info = sf.info(f)
            duration = info.duration
            samplerate = info.samplerate
            data.append({
                'speaker': speaker,
                'word': word,
                'duration': duration,
                'samplerate': samplerate,
                'filepath': f
            })
        except Exception as e:
            print(f"Error reading {f}: {e}")

    df = pd.DataFrame(data)
    
    # 1. Number of speakers, words, utterances
    num_speakers = df['speaker'].nunique()
    num_words = df['word'].nunique()
    num_utterances = len(df)
    
    # Intelligibility levels (standard UA-Speech mapping for these speakers)
    # F02: Low (29%), F03: Low (8%), F04: Mid (62%), F05: High (86%) - approximate from literature
    intelligibility = {
        'F02': 'Low (29%)',
        'F03': 'Low (8%)',
        'F04': 'Mid (62%)',
        'F05': 'High (86%)'
    }
    
    # 2. Duration statistics
    total_duration = df['duration'].sum() / 3600 # in hours
    mean_dur = df['duration'].mean()
    std_dur = df['duration'].std()
    
    summary_md = f"""# Dataset Analysis Summary

**Global Statistics**
- Number of Speakers: {num_speakers}
- Number of Unique Words: {num_words}
- Total Utterances: {num_utterances}
- Total Audio Duration: {total_duration:.2f} hours
- Mean Utterance Duration: {mean_dur:.2f} sec (± {std_dur:.2f} sec)

**Speaker Intelligibility Levels**
"""
    for spk, intel in intelligibility.items():
        summary_md += f"- {spk}: {intel}\n"
        
    with open(os.path.join(RESULTS_DIR, 'dataset_summary.md'), 'w') as f:
        f.write(summary_md)
        
    # 3. Speaker statistics table
    spk_stats = df.groupby('speaker').agg(
        Utterances=('word', 'count'),
        Unique_Words=('word', 'nunique'),
        Total_Duration_min=('duration', lambda x: x.sum() / 60),
        Mean_Duration_sec=('duration', 'mean')
    ).round(2)
    spk_stats['Intelligibility'] = spk_stats.index.map(intelligibility)
    
    spk_stats.to_csv(os.path.join(RESULTS_DIR, 'speaker_statistics.csv'))
    
    # 4. Histograms of audio durations
    plt.figure(figsize=(10, 6))
    plt.hist(df['duration'], bins=50, color='skyblue', edgecolor='black')
    plt.title('Distribution of Audio Durations')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig(os.path.join(RESULTS_DIR, 'duration_histogram.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Per-speaker duration boxplot
    plt.figure(figsize=(10, 6))
    df.boxplot(column='duration', by='speaker', grid=False)
    plt.title('Audio Duration by Speaker')
    plt.suptitle('')
    plt.xlabel('Speaker')
    plt.ylabel('Duration (seconds)')
    plt.savefig(os.path.join(RESULTS_DIR, 'speaker_duration_boxplot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Word frequency distribution
    word_counts = df['word'].value_counts()
    plt.figure(figsize=(15, 6))
    word_counts.head(50).plot(kind='bar', color='coral')
    plt.title('Top 50 Most Frequent Words')
    plt.xlabel('Word Code')
    plt.ylabel('Count')
    plt.xticks(rotation=90, fontsize=8)
    plt.savefig(os.path.join(RESULTS_DIR, 'word_frequency.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print("Phase 1 analysis complete. Results saved to:", RESULTS_DIR)

if __name__ == "__main__":
    analyze_dataset()
