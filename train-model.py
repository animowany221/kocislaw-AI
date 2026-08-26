 ==============================================================================
# KROK 1: Instalacja Unsloth i wymaganych bibliotek
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
# KROK 2: Wczytanie modelu bazowego (Llama 3.2 3B)
# ==============================================================================
max_seq_length = 2048
dtype = None # Auto-detekcja (fp16 / bf16)
load_in_4bit = True # 4-bitowa kwantyzacja dla oszczędności VRAM

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# Nałożenie warstw adaptacyjnych LoRA (Sigma tuning)
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
# KROK 3: Pobranie zbioru z Hugging Face i formatowanie pod styl "Codey"
# ==============================================================================
# Baza 18,000 instrukcji Pythona z Hugging Face
raw_dataset = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")

def format_to_codey_style(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []

    for inst, inp, out in zip(instructions, inputs, outputs):
        prompt = inst if not inp else f"{inst}\nKontekst: {inp}"

        # Narzucamy styl Codeya: wymuszony print(...) na początku odpowiedzi
        codey_response = f'print("siema, oto kod dla ciebie:")\n\n{out}'

        # Formatowanie czatu zgodne ze strukturą Llama-3
        text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nJesteś Codey. Odpowiadasz WYŁĄCZNIE w języku Python. Każda odpowiedź musi zaczynać się od instrukcji print().<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{codey_response}<|eot_id|>"
        texts.append(text)

    return { "text" : texts }

dataset = raw_dataset.map(format_to_codey_style, batched = True)

# ==============================================================================
# KROK 4: Rozpoczęcie procesu uczenia (Poprawiona wersja bez błędów pickling)
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

        # --- HIPERPARAMETRY ---
        num_train_epochs = 1,                 # 1 pełna epoka
        learning_rate = 2e-4,                 # Dedykowany learning rate
        warmup_steps = 20,                    # Zamiast deprecated warmup_ratio
        weight_decay = 0.01,                  # Zapobiega przeuczeniu
        lr_scheduler_type = "cosine",         # Wygładzanie LR

        # --- ZAPISYWANIE I LOGOWANIE ---
        logging_steps = 10,
        save_strategy = "no",                 # Wyłączamy save_steps, żeby nie zapychać dysku Colaba i uniknąć błędu pickling
        output_dir = "outputs",
        report_to = "none",                   # Brak zewnętrznych logów

        # --- SPRZĘTOWE ---
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        optim = "adamw_8bit",
        seed = 3407,
    ),
)

print("--- Start trenowania Codeya (Podejście #2) ---")
trainer_stats = trainer.train()
# ==============================================================================
# KROK 5: Testowe sprawdzenie odpowiedzi w Colabie
# ==============================================================================
FastLanguageModel.for_inference(model)
inputs = tokenizer(
[
    f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nJesteś Codey. Odpowiadasz WYŁĄCZNIE w języku Python. Każda odpowiedź musi zaczynać się od instrukcji print().<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nNapisz funkcję do sprawdzania czy liczba jest pierwsza.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 256, use_cache = True)
print("\n--- Testowa odpowiedź Codeya ---")
print(tokenizer.batch_decode(outputs)[0])

# ==============================================================================
# KROK 6: Eksport wyuczonego modelu do pliku .GGUF dla Ollamy
# ==============================================================================
# Tworzy gotowy plik codey-model-Q8_0.gguf
model.save_pretrained_gguf("codey-model", tokenizer, quantization_method = "q8_0")
