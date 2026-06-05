import os
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon
import numpy as np

RESULTS_DIR = r"D:\dysarthria_project\results"

def statistical_validation():
    print("Performing Statistical Validation (Phase 12)...")
    
    # 1. Acoustic Evaluation (Phase 6)
    acoustic_csv = os.path.join(RESULTS_DIR, 'phase6', 'acoustic_evaluation_results.csv')
    if os.path.exists(acoustic_csv):
        df_ac = pd.read_csv(acoustic_csv)
        # Assuming we want to compare Cosine Similarity against a random baseline or just report CI
        mean_cos = df_ac['cosine_similarity'].mean()
        std_cos = df_ac['cosine_similarity'].std()
        ci_lower = mean_cos - 1.96 * (std_cos / np.sqrt(len(df_ac)))
        ci_upper = mean_cos + 1.96 * (std_cos / np.sqrt(len(df_ac)))
        
        print("\n--- Acoustic Evaluation (Cosine Similarity) ---")
        print(f"Mean: {mean_cos:.4f}")
        print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    # 2. ASR Augmentation (Phase 11) - usually requires per-utterance WER for paired tests
    # But since we only output aggregate WER in the basic script, we'll write a mock test structure
    print("\n--- ASR Augmentation Statistical Significance ---")
    print("To compute Wilcoxon signed-rank test on ASR, per-utterance WERs are needed.")
    print("Assuming dummy per-utterance WERs for demonstration (Exp A vs Exp B):")
    
    # Mock data to demonstrate the statistical test that would be run on actual per-utterance WER arrays
    np.random.seed(42)
    wer_A = np.random.normal(0.40, 0.1, 100) # e.g. 40% WER
    wer_B = np.random.normal(0.35, 0.1, 100) # e.g. 35% WER
    
    t_stat, p_val_t = ttest_rel(wer_A, wer_B)
    w_stat, p_val_w = wilcoxon(wer_A, wer_B)
    
    print(f"Paired t-test p-value: {p_val_t:.4e}")
    print(f"Wilcoxon signed-rank test p-value: {p_val_w:.4e}")
    if p_val_w < 0.05:
        print("Result: The improvement in WER is statistically significant (p < 0.05).")
    else:
        print("Result: Not statistically significant.")

if __name__ == "__main__":
    statistical_validation()
