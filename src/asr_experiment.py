import os
import glob
import pandas as pd
import torch
from datasets import Dataset, Audio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, TrainingArguments, Trainer
import evaluate
from tqdm import tqdm

REAL_DIR = r"D:\dysarthria_project\data\processed"
SYNTH_DIR = r"D:\dysarthria_project\results\phase5_synthetic"
RESULTS_DIR = r"D:\dysarthria_project\results\phase11"

os.makedirs(RESULTS_DIR, exist_ok=True)

def create_dataset_csv():
    # Map words to actual text if needed, here the word is the label
    data_A = [] # Real only
    data_B = [] # Real + Synthetic
    
    # Real data
    for spk in ['F02', 'F03', 'F04', 'F05']:
        files = glob.glob(os.path.join(REAL_DIR, spk, '*.wav'))
        for f in files:
            word = os.path.basename(f).split('_')[2]
            record = {'audio': f, 'text': word.upper()}
            data_A.append(record)
            data_B.append(record)
            
    # Synthetic data
    synth_files = glob.glob(os.path.join(SYNTH_DIR, '**', '*.wav'), recursive=True)
    for f in synth_files:
        basename = os.path.basename(f)
        parts = basename.split('_')
        if len(parts) >= 3:
            word = parts[2]
            data_B.append({'audio': f, 'text': word.upper()})
            
    df_A = pd.DataFrame(data_A)
    df_B = pd.DataFrame(data_B)
    
    # Simple split (80-20)
    df_A_train = df_A.sample(frac=0.8, random_state=42)
    df_A_test = df_A.drop(df_A_train.index)
    
    df_B_train = df_B.sample(frac=0.8, random_state=42)
    # Ensure test set is only real data for fair comparison
    df_B_test = df_B[df_B['audio'].str.contains('processed')].sample(n=len(df_A_test), random_state=42)
    
    return df_A_train, df_A_test, df_B_train, df_B_test

def train_asr(df_train, df_test, experiment_name):
    print(f"Starting ASR Training: {experiment_name}")
    print(f"Train size: {len(df_train)}, Test size: {len(df_test)}")
    
    # Load dataset
    ds_train = Dataset.from_pandas(df_train)
    ds_test = Dataset.from_pandas(df_test)
    
    ds_train = ds_train.cast_column("audio", Audio(sampling_rate=16000))
    ds_test = ds_test.cast_column("audio", Audio(sampling_rate=16000))
    
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-base", 
        ctc_loss_reduction="mean", 
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer)
    )
    
    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
        batch["labels"] = processor(text=batch["text"]).input_ids
        return batch
        
    ds_train = ds_train.map(prepare_dataset, remove_columns=ds_train.column_names, num_proc=1)
    ds_test = ds_test.map(prepare_dataset, remove_columns=ds_test.column_names, num_proc=1)
    
    wer_metric = evaluate.load("wer")
    
    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}
        
    training_args = TrainingArguments(
        output_dir=os.path.join(RESULTS_DIR, experiment_name),
        group_by_length=True,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        evaluation_strategy="steps",
        num_train_epochs=1, # Very low for demonstration speed
        max_steps=50,       # Very low for demonstration speed
        fp16=False,
        save_steps=25,
        eval_steps=25,
        logging_steps=10,
        learning_rate=1e-4,
        warmup_steps=10,
        save_total_limit=2,
    )
    
    # Custom Data Collator
    from transformers import DataCollatorForCTCWithPadding
    data_collator = DataCollatorForCTCWithPadding(processor=processor, padding=True)
    
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=ds_train,
        eval_dataset=ds_test,
        tokenizer=processor.feature_extractor,
    )
    
    trainer.train()
    
    # Evaluate
    metrics = trainer.evaluate()
    print(f"WER for {experiment_name}: {metrics.get('eval_wer', 'N/A')}")
    return metrics.get('eval_wer', 'N/A')

def run_experiments():
    df_A_train, df_A_test, df_B_train, df_B_test = create_dataset_csv()
    
    if len(df_A_train) == 0:
        print("No data found for ASR.")
        return
        
    wer_A = train_asr(df_A_train, df_A_test, "Exp_A_Real_Only")
    wer_B = train_asr(df_B_train, df_B_test, "Exp_B_Real_And_Synthetic")
    
    # Save results
    results = [
        {"Experiment": "A (Real Only)", "WER": wer_A},
        {"Experiment": "B (Real + Synthetic)", "WER": wer_B}
    ]
    pd.DataFrame(results).to_csv(os.path.join(RESULTS_DIR, "asr_wer_results.csv"), index=False)
    
    print("\n=== ASR AUGMENTATION EXPERIMENT RESULTS ===")
    print(f"WER Real Only: {wer_A}")
    print(f"WER Real + Synthetic: {wer_B}")

if __name__ == "__main__":
    # Note: Requires datasets, transformers, evaluate to be installed
    pass
    # run_experiments() # Disabled by default to prevent huge downloads unless explicitly executed
