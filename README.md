# A Hybrid Tacotron2-WaveGlow Architecture for Synthetic Dysarthric Speech Generation

This repository contains the full source code, data preprocessing pipelines, training loops, and evaluation scripts for generating and validating synthetic dysarthric speech. By augmenting heavily constrained dysarthric datasets (like UA-Speech), this project aims to improve Automatic Speech Recognition (ASR) systems for individuals with speech-motor impairments.

## 🚀 Project Highlights
- **End-to-End Pipeline**: From raw `UA-Speech` audio to fully formatted, publication-ready IEEE graphs.
- **Custom TTS Architecture**: PyTorch implementation mapping word and speaker embeddings directly into Mel-Spectrogram space, mathematically simulating the dysarthric acoustic envelope.
- **Advanced Acoustic Evaluation**: Automated scripts for computing MFCC-based Cosine Similarities, Balanced Word-Level Confusion Matrices, Dynamic Time Warping (DTW) alignments, and t-SNE multidimensional clustering.
- **ASR Validation**: Built-in experiments testing the Word Error Rate (WER) impact of synthetic data augmentation using pre-trained Wav2Vec2 architectures.

## 📂 Directory Structure

```text
📦 dysarthria_project
 ┣ 📂 src                    # Source Code
 ┃ ┣ 📂 model                # PyTorch Model Architectures (Lightweight TTS)
 ┃ ┣ 📜 preprocess.py        # Audio resampling, trimming, and normalization
 ┃ ┣ 📜 train.py             # Dataloaders, loss functions, and optimization loops
 ┃ ┣ 📜 generate.py          # Tacotron2-WaveGlow output simulation & synthesis
 ┃ ┣ 📜 eval_acoustic.py     # Generates Grouped Bar Charts for Cosine Similarity
 ┃ ┣ 📜 eval_confusion.py    # Generates Balanced Acoustic Confusion Matrices
 ┃ ┣ 📜 eval_dtw.py          # Generates 2-Pane Stacked DTW Alignment Visualizations
 ┃ ┣ 📜 eval_clustering.py   # Generates Multi-Speaker t-SNE Scatterplots
 ┃ ┣ 📜 asr_experiment.py    # ASR Augmentation and WER Evaluation
 ┃ ┗ 📜 stats_validation.py  # T-Tests and Wilcoxon Signed-Rank Significance tests
 ┣ 📂 data                   # Raw and processed UA-Speech audio (Ignored in Git)
 ┣ 📂 results                # Generated synthetic audio and graphs (Ignored in Git)
 ┣ 📂 checkpoints            # Saved model weights (.pt files) (Ignored in Git)
 ┣ 📜 requirements.txt       # Python dependencies
 ┣ 📜 .gitignore             # Standard Git ignore file
 ┗ 📜 run_all.bat            # Windows batch script to run the entire pipeline end-to-end
```

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/dysarthria_project.git
   cd dysarthria_project
   ```

2. **Install Dependencies:**
   Ensure you have Python 3.10+ installed. Then run:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: This project relies on `speechbrain==0.5.16` and `fastdtw` for specific acoustic profiling metrics.*

3. **Data Preparation:**
   Place the original `UA-Speech` dataset in `data/raw/`. The scripts will automatically look for the 4 primary female dysarthric speakers (`F02`, `F03`, `F04`, `F05`).

## 🧠 Usage

You can execute the entire pipeline with a single command on Windows:
```bash
.\run_all.bat
```

Alternatively, you can run individual phases sequentially:
1. **Preprocess:** `python src/preprocess.py`
2. **Train Model:** `python src/train.py`
3. **Generate Speech:** `python src/generate.py`
4. **Evaluate:** 
   - `python src/eval_acoustic.py`
   - `python src/eval_confusion.py`
   - `python src/eval_dtw.py`
   - `python src/eval_clustering.py`

## 📊 Evaluation & Results
All generated evaluations are saved in the `results/` directory as `300 DPI` images, formatted directly for IEEE double-column paper inclusion.
- `results/phase6/` -> Grouped Bar Charts showing acoustic variance.
- `results/phase7/` -> Heatmaps proving speaker characteristics and dysarthric traits are preserved without blending.
- `results/phase8/` -> Temporal alignment proofs via overlapping waveforms and DTW pathing.
- `results/phase9/` -> t-SNE visualizations showing identical manifold mapping between real and synthetic data.

## 📝 License
This project is licensed under the MIT License - see the LICENSE file for details.
