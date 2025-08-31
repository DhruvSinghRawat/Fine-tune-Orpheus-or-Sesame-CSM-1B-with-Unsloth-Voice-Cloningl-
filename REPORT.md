# Voice Cloning Assignment -- Brief Report

## What Worked

-   ✅ **Environment Setup:** Successfully set up Google Colab with GPU
    runtime and installed dependencies (PyTorch, Transformers, Unsloth,
    PEFT, TRL, etc.).\
-   ✅ **Data Loading:** Loaded and preprocessed training dataset
    correctly.\
-   ✅ **Model Initialization:** Base model loaded and LoRA
    configuration applied without issues.\
-   ✅ **Logging & Monitoring:** Integrated TensorBoard for training
    visualization.

## What Failed / Issues Faced

-   ❌ **Training Stage Blocked:** Training is not progressing due to a
    **Trainer configuration error** in `TrainingArguments` (likely
    caused by missing/incorrect parameters such as
    `evaluation_strategy`, `logging_dir`, and `fp16/bf16 flags`).\
-   ❌ **Memory Usage Calculation:** Had to fix partial code (missing
    variables like `max_memory` and incorrect syntax in f-strings).\
-   ❌ **Gradient & Batch Config:** Higher batch sizes caused CUDA
    Out-Of-Memory (OOM) errors, so gradient accumulation and batch size
    had to be tuned down.

## Suggestions / Next Steps

-   🔧 **Fix TrainingArguments Error:** Ensure correct syntax (e.g.,
    `evaluation_strategy="steps"`, `fp16=is_bfloat16_supported()`, and
    proper `logging_dir` path).\
-   🔧 **Memory Optimization:** Use smaller batch size (e.g., 8 or 4)
    with higher `gradient_accumulation_steps` to fit within VRAM.\
-   🔧 **Progressive Debugging:** Run training with a very small dataset
    (subset) to validate Trainer config before scaling to full dataset.\
-   🔧 **Check GPU Utilization:** Monitor GPU memory via `!nvidia-smi`
    during training to confirm memory allocation.\
-   📌 **Next Milestone:** Once Trainer config is fixed, re-run training
    and document training loss and validation metrics.
