# ==============================================================================
# STEP 1: Installation of Unsloth and required libraries
# ==============================================================================
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps xformers trl peft accelerate bitsandbytes

import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

# ==============================================================================
# STEP 2: Loading the base model (default Llama 3.2 3B Instruct)
# ==============================================================================
max_seq_length = 2048
dtype = None # Auto-detection (fp16 / bf16)
load_in_4bit = True # 4-bit quantization to save VRAM

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit", # Here you can place your model
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# Applying LoRA adaptation layers (Sigma tuning)
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# ==============================================================================
# STEP 3: Downloading dataset from Hugging Face and formatting to "Codey" style
# ==============================================================================
# Database of 18,000 Python instructions from Hugging Face
raw_dataset = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")

def format_to_codey_style(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []

    for inst, inp, out in zip(instructions, inputs, outputs):
        prompt = inst if not inp else f"{inst}\nKontekst: {inp}"

        # Enforcing Codey's style: mandatory print(...) at the beginning of the response
        codey_response = f'print("siema, oto kod dla ciebie:")\n\n{out}'

        # Chat formatting according to Llama-3 structure
        text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nJesteś Codey. Odpowiadasz WYŁĄCZNIE w języku Python. Każda odpowiedź musi zaczynać się od instrukcji print().<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{codey_response}<|eot_id|>"
        texts.append(text)

    return { "text" : texts }

dataset = raw_dataset.map(format_to_codey_style, batched = True)

# ==============================================================================
# STEP 4: Starting the training process (Improved version without pickling errors)
# ==============================================================================
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,

        # --- HYPERPARAMETERS ---
        num_train_epochs = 1,                 # 1 full epoch
        learning_rate = 2e-4,                 # Dedicated learning rate
        warmup_steps = 20,                    # Instead of deprecated warmup_ratio
        weight_decay = 0.01,                  # Prevents overfitting
        lr_scheduler_type = "cosine",         # LR smoothing (cosine schedule)

        # --- SAVING AND LOGGING ---
        logging_steps = 10,
        save_strategy = "no",                 # Disabling save_steps to avoid filling Colab's disk and prevent pickling errors
        output_dir = "outputs",
        report_to = "none",                   # No external logging

        # --- HARDWARE ---
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        optim = "adamw_8bit",
        seed = 3407,
    ),
)

print("--- Starting Codey's training (Approach #2) ---")
trainer_stats = trainer.train()

# ==============================================================================
# STEP 5: Testing the response
# ==============================================================================
FastLanguageModel.for_inference(model)
inputs = tokenizer(
[
    f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nJesteś Codey. Odpowiadasz WYŁĄCZNIE w języku Python. Każda odpowiedź musi zaczynać się od instrukcji print().<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nNapisz funkcję do sprawdzania czy liczba jest pierwsza.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 256, use_cache = True)
print("\n--- Codey's test response ---")
print(tokenizer.batch_decode(outputs)[0])

# ==============================================================================
# STEP 6: Exporting the trained model to a .GGUF file
# ==============================================================================
# Creates a ready codey-model-Q8_0.gguf file
model.save_pretrained_gguf("codey-model", tokenizer, quantization_method = "q8_0")
