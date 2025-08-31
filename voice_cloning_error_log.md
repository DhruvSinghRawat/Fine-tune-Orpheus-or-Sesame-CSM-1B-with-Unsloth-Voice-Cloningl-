# Voice Cloning Project – Error Log

This document tracks all errors, warnings, and issues encountered during the development and testing of the **Voice Cloning Project**, along with brief notes on potential causes and next steps.

---

## 🐛 Error Log

### 1. Installation & Environment Issues
- **Error:** `xformers==0.0.29.post3` not found / mismatch with CUDA version.
- **Cause:** Version mismatch between xformers and Colab runtime.
- **Action:** Pin xformers to compatible CUDA build or switch to nightly build.

### 2. GPU/Memory Warnings
- **Warning:** `UserWarning: GPU memory low, using CPU fallback for some ops.`
- **Cause:** Batch size too high for available VRAM.
- **Action:** Reduce `per_device_train_batch_size` or use gradient checkpointing.

### 3. Runtime Freezes / Long Execution
- **Issue:** Training takes unusually long for small dataset.
- **Cause:** Possibly high `max_steps` or inefficient dataloader.
- **Action:** Reduce steps/epochs, enable `fp16` or `bfloat16` to speed up.

### 4. Syntax & Indentation Errors
- **Error:** Misplaced `+`, `—`, `,` leading to invalid Python.
- **Action:** Re-indented and corrected operators (e.g. `model_name + time.strftime(...)`).

### 5. Missing Variables
- **Error:** `NameError: used_memory not defined`
- **Cause:** Missing `torch.cuda.memory_allocated()` call.
- **Action:** Add proper memory tracking code before printing.

### 6. Trainer Argument Errors
- **Error:** `TypeError: unexpected keyword argument 'evat_strategy'`
- **Cause:** Typo in `evaluation_strategy` argument.
- **Action:** Corrected to `evaluation_strategy="steps"`.

### 7. Unsupported Precision Flag
- **Error:** `Trainer got unexpected keyword argument 'bfi6'`
- **Cause:** Typo, should be `bf16`.
- **Action:** Fixed spelling.

### 8. Dataloader KeyErrors
- **Error:** `KeyError: 'input_ids'`
- **Cause:** Tokenizer output not properly formatted.
- **Action:** Ensure `Dataset.map()` returns dict with `input_ids` and `attention_mask`.

### 9. Trainer Logging Issues
- **Warning:** `logging_dir not set, using default runs/ folder.`
- **Action:** Set `logging_dir="./logs"` in TrainingArguments.

### 10. Memory Calculation Bugs
- **Error:** Division by zero in `% of max memory` calculation.
- **Cause:** `max_memory` not initialized.
- **Action:** Use `torch.cuda.get_device_properties(0).total_memory`.

### 11. Model Output Shape Mismatch
- **Error:** `RuntimeError: size mismatch between logits and labels`
- **Cause:** Wrong tokenizer padding/truncation.
- **Action:** Ensure `tokenizer(..., padding='max_length', truncation=True)`.

### 12. Dataset Size Imbalance
- **Issue:** Train set too small, eval set too big.
- **Action:** Split dataset using `train_test_split(test_size=0.1)`.

### 13. Trainer Crashes at Save
- **Error:** `OSError: Directory not empty: outputs/`
- **Action:** Clear `outputs/` before rerun or set `overwrite_output_dir=True`.

### 14. Mixed Precision Errors
- **Error:** `GradScaler not enabled for bf16`
- **Cause:** Colab GPU doesn't support bf16.
- **Action:** Fallback to `fp16=True`.

### 15. Logging Formatter Failures
- **Error:** `KeyError: 'train_runtime'l`
- **Cause:** Typo in dictionary key.
- **Action:** Fix to `trainer_state.metrics['train_runtime']`.

### 16. LoRA Memory Calculation Error
- **Error:** `used_memory_for_lora not defined`
- **Action:** Add explicit memory snapshot before and after LoRA adapter load.

### 17. TensorBoard Not Logging
- **Issue:** `tensorboard` logs empty.
- **Action:** Install tensorboard with `pip install tensorboard`, run `%load_ext tensorboard`.

### 18. Tokenizer Incompatibility
- **Error:** `ValueError: tokenizer length mismatch with model embedding size`
- **Action:** Resize token embeddings: `model.resize_token_embeddings(len(tokenizer))`.

### 19. Wrong Optimizer Name
- **Error:** `ValueError: Optimizer 'adamw 8bit' not recognized`
- **Action:** Use `optim="paged_adamw_8bit"` from bitsandbytes.

### 20. Data Collator Issue
- **Error:** `TypeError: 'NoneType' object is not callable`
- **Cause:** Data collator not passed or not initialized.
- **Action:** Use `DataCollatorForLanguageModeling(tokenizer, mlm=False)`.

### 21. Checkpoint Loading Failure
- **Error:** `RuntimeError: state_dict mismatch`
- **Action:** Match LoRA config with checkpoint or re-train from scratch.

### 22. Missing Model Card
- **Warning:** `Model card not found, skipping push_to_hub`
- **Action:** Add `README.md` to repo or skip hub upload.

### 23. CUDA OOM During Training
- **Action:** Reduce `batch_size`, enable gradient checkpointing, or use 4-bit quantization.

### 24. Trainer Progress Bar Stuck
- **Cause:** Multiprocessing bug in Colab.
- **Action:** Set `dataloader_num_workers=0` in TrainingArguments.

### 25. WandB Integration Error
- **Error:** `wandb not installed`
- **Action:** Install with `pip install wandb` or disable by `report_to=[]`.

### 26. UnicodeDecodeError in Dataset
- **Cause:** Corrupt audio transcript file.
- **Action:** Reload dataset with correct encoding (`utf-8`).

### 27. Mixed GPU Types
- **Error:** `RuntimeError: Attempting to run on different CUDA devices`
- **Action:** Force `.to('cuda')` on all tensors and model.

### 28. Loss Not Decreasing
- **Cause:** Learning rate too high.
- **Action:** Lower to `2e-5` and enable warmup.

### 29. Output Audio Distorted
- **Cause:** Post-processing not applied.
- **Action:** Normalize audio waveform after synthesis.

---

✅ **Next Step:** Continue capturing new errors and their resolutions here for reproducibility and better debugging in future training sessions.

