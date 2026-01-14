# Fine-Tuning Speech Emotion Recognition (SER) on Your Voice

> **Context:** You are in 2026. The most reliable "foundation" models for this are currently **SenseVoice-Small** (2024) or **Wav2Vec2-Large-Robust** (2021). 

If the calibration script didn't solve your issue, you can fine-tune a model to specifically recognize *your* emotions.

---

## 1. Data Collection (The Hard Part)

You need about **50-100 clips** of yourself (3-5 seconds each).

1.  **Structure:** Create folders for each emotion.
    ```text
    dataset/
    ├── angry/
    │   ├── angry_01.wav
    │   └── ...
    ├── calm/
    │   ├── calm_01.wav
    │   └── ...
    ├── happy/
    │   ├── happy_01.wav
    │   └── ...
    └── sad/
        ├── sad_01.wav
        └── ...
    ```
2.  **Recording:** Use the Voice Agent app to record, then move files from `recordings/` to the folders above.

---

## 2. Fine-Tuning Script (Python)

Save this as `train_ser.py`. You need a GPU (or run this on Google Colab/RunPod).

### Prerequisites
```bash
pip install transformers datasets evaluate accelerate librosa soundfile scikit-learn
```

### The Script

```python
import os
from datasets import load_dataset, Audio
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, TrainingArguments, Trainer
import evaluate
import numpy as np

# 1. Config
MODEL_ID = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"  # The base model you use now
DATASET_PATH = "./dataset"  # Your folder structure
OUTPUT_DIR = "./my_custom_ser_model"

# 2. Load Data
dataset = load_dataset("audiofolder", data_dir=DATASET_PATH)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# Split
dataset = dataset["train"].train_test_split(test_size=0.2)
labels = dataset["train"].features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label

# 3. Preprocess
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)

def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays, 
        sampling_rate=feature_extractor.sampling_rate, 
        max_length=16000*5, # 5 seconds max
        truncation=True
    )
    return inputs

encoded_dataset = dataset.map(preprocess_function, batched=True)

# 4. Metric
accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)

# 5. Training
model = AutoModelForAudioClassification.from_pretrained(
    MODEL_ID,
    num_labels=len(labels),
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True # Important for custom labels
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=4, # Low batch size for consumer GPUs
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=4,
    num_train_epochs=10,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,
)

trainer.train()

# 6. Save
trainer.save_model(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}")
```

## 3. Using the New Model

Once trained, change `EmotionDetector.MODEL_ID` in `audio/emotion_detector.py` to point to your new folder:

```python
# audio/emotion_detector.py
class EmotionDetector:
    # MODEL_ID = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
    MODEL_ID = "C:/AI/index-tts/voice_chat/my_custom_ser_model" # Your local path
```
