@echo off
echo ====================================================
echo DYSARTHRIC SPEECH SYNTHESIS PIPELINE
echo ====================================================

echo [1/9] Running Phase 1: Dataset Analysis...
python src/phase1_analysis.py

echo [2/9] Running Phase 2: Preprocessing...
python src/preprocess.py

echo [3/9] Running Phase 4: Training TTS Model...
python src/train.py

echo [4/9] Running Phase 5: Synthetic Speech Generation...
python src/generate.py

echo [5/9] Running Phase 6: Acoustic Evaluation...
python src/eval_acoustic.py

echo [6/9] Running Phase 7: Speaker Preservation...
python src/eval_speaker.py

echo [7/9] Running Phase 8: Temporal Validation (DTW)...
python src/eval_dtw.py

echo [8/9] Running Phase 9: Clustering Analysis...
python src/eval_clustering.py

echo [9/9] Running Phase 12: Statistical Validation...
python src/stats_validation.py

echo ====================================================
echo PIPELINE COMPLETE
echo All results saved in D:\dysarthria_project\results\
echo ====================================================
pause
