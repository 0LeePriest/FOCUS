# FOCUS: A Fine-Grained Customer-Oriented Sentiment Dialogue Summarization Dataset for Chinese Customer Service

Review note: anonymized repository for ACL review.  

## Contents
- `code/`: minimal training/inference scripts (BART, T5, LLaMA, DeepSeek). Some paths and checkpoints are placeholders.
- `dataset/`: curated JSON data in two styles: `refined_dataset-formatted` and `refined_dataset-free-style`, each with `train.json`, `val.json`, `test.json`.
- `result/`: selected prediction files and summary spreadsheets for qualitative inspection.

## Dataset Format (formatted style)
- `DialogueID` (int), `TimeID` (float/timestamp).
- `Dialogue`: list of turns with fields `speaker` (customer/agent), `turn` (int), `utterance` (str).
- Optional `aspect-opinion-sentiment-emotion` annotations.
- `Summ`: reference summary.

## Setup
- Python 3.9-3.11.
- GPU recommended (>=16GB for LLaMA/LoRA demos).
- Dependencies: `torch`, `transformers`, `datasets`, `peft`, `accelerate`, `rouge-chinese`, `sentencepiece`.

Install example (adjust versions to your CUDA/system):
```
pip install -U transformers datasets peft accelerate rouge-chinese sentencepiece
# Install torch per https://pytorch.org/get-started/locally/
```

## Quick Start (demo only)
The scripts are for inspection and limited demo. Some assets are withheld; adjust paths as needed.

- BART training (formatted style):
  1) `cd code`
  2) Copy data:
     - `copy ..\dataset\refined_dataset\refined_dataset-formatted\train.json .`
     - `copy ..\dataset\refined_dataset\refined_dataset-formatted\val.json .`
  3) Edit model/tokenizer paths in `bart_train.py` if needed.
  4) Run: `python bart_train.py`

- BART inference:
  1) `cd code`
  2) `copy ..\dataset\refined_dataset\refined_dataset-formatted\test.json .`
  3) Set `model_path` in `bart_inference.py` to your checkpoint.
  4) Run: `python bart_inference.py`

T5 and LLaMA LoRA scripts are provided similarly; some paths are placeholders during review.

## Reproducibility and Release Plan
- Review phase: holding back full configs, checkpoints, and some preprocessing details.
- After acceptance: release complete pipelines, hyperparameters, data preparation code, and checkpoints or detailed reproduction instructions.

## License
- Review phase: all rights reserved. Redistribution or commercial use is not permitted.



