import os
from collections import defaultdict

def analyze_filelist(filepath):
    speakers = set()
    words = defaultdict(int)
    utterance_count = 0
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('|')
            if len(parts) != 2:
                continue
            path, transcript = parts
            
            # Paths look like /content/drive/MyDrive/dysarthria_project/audio/words/M04/CW9/M04_B1_CW9_M8.wav
            # Or /content/drive/MyDrive/dysarthria_project/audio/M07/M07_B1_CW9_M6.wav
            # Or /content/drive/MyDrive/dysarthria_project/audio/control/CM10/CM10_B1_CW32_M2.wav
            
            # The filename usually starts with the speaker ID
            filename = path.split('/')[-1]
            speaker_id = filename.split('_')[0]
            
            speakers.add(speaker_id)
            words[transcript.lower()] += 1
            utterance_count += 1
            
    return speakers, words, utterance_count

train_spk, train_words, train_cnt = analyze_filelist('train_filelist.txt')
val_spk, val_words, val_cnt = analyze_filelist('val_filelist.txt')

all_speakers = train_spk.union(val_spk)
all_words = defaultdict(int)
for w, c in train_words.items():
    all_words[w] += c
for w, c in val_words.items():
    all_words[w] += c

total_utterances = train_cnt + val_cnt

print(f"Total Utterances: {total_utterances}")
print(f"Total Speakers: {len(all_speakers)}")
print(f"Speakers: {sorted(list(all_speakers))}")
print(f"Total Unique Words: {len(all_words)}")

# Top 10 words
sorted_words = sorted(all_words.items(), key=lambda x: x[1], reverse=True)
print(f"Top 10 Words: {sorted_words[:10]}")

# Since audio files are inaccessible, we can't accurately get duration statistics directly.
