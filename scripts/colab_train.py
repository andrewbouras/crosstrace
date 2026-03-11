#!/usr/bin/env python3
"""
Colab Pro Training Script for Hypothesis Generation Model.

Copy this to Google Colab and run with GPU runtime (A100 preferred, V100 ok).

Steps:
1. Upload prepared/ directory to Google Drive under /MyDrive/hypothesis-training/
2. Open new Colab notebook, select GPU runtime (A100 if available)
3. Paste and run this script cell by cell

The script handles:
- Environment setup (installs Axolotl, bitsandbytes, etc.)
- Data loading from Google Drive
- QLoRA fine-tuning of Qwen2.5-7B-Instruct
- Checkpoint saving to Google Drive
- LoRA adapter merge for inference
"""

# ============================================================
# CELL 1: Environment Setup
# ============================================================
SETUP_SCRIPT = '''
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Install dependencies
!pip install -q axolotl[flash-attn] bitsandbytes accelerate peft transformers
!pip install -q sentence-transformers rouge-score

# Verify GPU
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
'''

# ============================================================
# CELL 2: Configuration
# ============================================================
CONFIG = '''
import os, json

# Paths
DRIVE_BASE = "/content/drive/MyDrive/hypothesis-training"
DATA_DIR = f"{DRIVE_BASE}/prepared"
OUTPUT_BASE = f"{DRIVE_BASE}/checkpoints"

# Verify data exists
for f in ["train_ours_only.jsonl", "train_combined.jsonl", "train_balanced.jsonl", "val.jsonl"]:
    path = f"{DATA_DIR}/{f}"
    if os.path.exists(path):
        with open(path) as fh:
            n = sum(1 for _ in fh)
        print(f"  {f}: {n} records")
    else:
        print(f"  {f}: NOT FOUND -- upload prepared/ directory to Google Drive")

# Load manifest
with open(f"{DATA_DIR}/manifest.json") as f:
    manifest = json.load(f)
print(f"\\nManifest: {json.dumps(manifest, indent=2)}")
'''

# ============================================================
# CELL 3: Run Selection
# ============================================================
RUN_SELECTION = '''
# Choose which run to execute
# Run 1: Our data only (1,180 records) -- baseline
# Run 2: Combined (6,658 records) -- full augmentation
# Run 2b: Balanced 2:1 (4,720 records) -- controlled ratio
# Run 3: Higher LoRA rank on best dataset

RUN = "1"  # Change this: "1", "2", "2b", "3"

run_configs = {
    "1": {
        "name": "run1_ours_only",
        "train_data": f"{DATA_DIR}/train_ours_only.jsonl",
        "lora_r": 16,
        "epochs": 3,
        "description": "Our 1,180 records only"
    },
    "2": {
        "name": "run2_combined",
        "train_data": f"{DATA_DIR}/train_combined.jsonl",
        "lora_r": 16,
        "epochs": 2,  # Fewer epochs for larger dataset
        "description": "Combined: 6,658 records (ours + HypoGen)"
    },
    "2b": {
        "name": "run2b_balanced",
        "train_data": f"{DATA_DIR}/train_balanced.jsonl",
        "lora_r": 16,
        "epochs": 3,
        "description": "Balanced 2:1: 4,720 records (ours 2x + HypoGen capped)"
    },
    "3": {
        "name": "run3_high_rank",
        "train_data": f"{DATA_DIR}/train_balanced.jsonl",  # Use balanced by default
        "lora_r": 32,
        "epochs": 3,
        "description": "Higher LoRA rank (r=32) on balanced data"
    },
}

cfg = run_configs[RUN]
OUTPUT_DIR = f"{OUTPUT_BASE}/{cfg['name']}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Selected: Run {RUN} -- {cfg['description']}")
print(f"Training data: {cfg['train_data']}")
print(f"LoRA rank: {cfg['lora_r']}")
print(f"Epochs: {cfg['epochs']}")
print(f"Output: {OUTPUT_DIR}")
'''

# ============================================================
# CELL 4: Training
# ============================================================
TRAINING = '''
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from datasets import load_dataset
import torch, json

# Load model in 4-bit
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
print(f"Loading {MODEL_NAME}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# Prepare for QLoRA
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=cfg["lora_r"],
    lora_alpha=cfg["lora_r"] * 2,
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules="all-linear",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Load and tokenize data
def tokenize_messages(examples):
    """Tokenize chat messages using the model's chat template."""
    texts = []
    for messages in examples["messages"]:
        if isinstance(messages, str):
            messages = json.loads(messages)
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)

    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=2048,
        padding="max_length",
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

print(f"Loading training data from {cfg['train_data']}...")
dataset = load_dataset("json", data_files={"train": cfg["train_data"]})
tokenized = dataset["train"].map(tokenize_messages, batched=True, remove_columns=dataset["train"].column_names)
print(f"Tokenized {len(tokenized)} records")

# Load validation data
val_path = f"{DATA_DIR}/val.jsonl"
if os.path.exists(val_path):
    val_dataset = load_dataset("json", data_files={"val": val_path})
    val_tokenized = val_dataset["val"].map(tokenize_messages, batched=True, remove_columns=val_dataset["val"].column_names)
    print(f"Validation: {len(val_tokenized)} records")
else:
    val_tokenized = None
    print("No validation data found")

# Training arguments
n_train = len(tokenized)
effective_batch = 4 * 8  # micro_batch * grad_accum = 32
steps_per_epoch = n_train // effective_batch
total_steps = steps_per_epoch * cfg["epochs"]
print(f"\\nTraining plan: {n_train} records, {steps_per_epoch} steps/epoch, {total_steps} total steps")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=cfg["epochs"],
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    weight_decay=0.01,
    lr_scheduler_type="linear",
    warmup_steps=10,
    logging_steps=5,
    save_steps=max(50, steps_per_epoch),  # Save at least once per epoch
    save_total_limit=3,
    eval_steps=max(50, steps_per_epoch) if val_tokenized else None,
    evaluation_strategy="steps" if val_tokenized else "no",
    bf16=True,
    gradient_checkpointing=True,
    report_to="none",  # Disable wandb etc
    dataloader_pin_memory=False,
)

from transformers import Trainer, DataCollatorForLanguageModeling

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    eval_dataset=val_tokenized,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

print(f"\\nStarting training...")
result = trainer.train()
print(f"\\nTraining complete!")
print(f"  Loss: {result.training_loss:.4f}")
print(f"  Steps: {result.global_step}")

# Save final adapter
adapter_path = f"{OUTPUT_DIR}/final_adapter"
model.save_pretrained(adapter_path)
tokenizer.save_pretrained(adapter_path)
print(f"\\nAdapter saved to {adapter_path}")
'''

# ============================================================
# CELL 5: Merge Adapter (for inference)
# ============================================================
MERGE = '''
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_PATH = f"{OUTPUT_DIR}/final_adapter"
MERGED_PATH = f"{OUTPUT_DIR}/merged"

print(f"Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

print(f"Loading adapter from {ADAPTER_PATH}...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

print(f"Merging...")
merged = model.merge_and_unload()

print(f"Saving merged model to {MERGED_PATH}...")
merged.save_pretrained(MERGED_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.save_pretrained(MERGED_PATH)
print(f"Done! Merged model at {MERGED_PATH}")
'''

# ============================================================
# CELL 6: Quick Evaluation (on Colab)
# ============================================================
EVALUATE = '''
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch, json

MERGED_PATH = f"{OUTPUT_DIR}/merged"

print("Loading merged model for evaluation...")
tokenizer = AutoTokenizer.from_pretrained(MERGED_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MERGED_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

SYSTEM_PROMPT = (
    "You are a scientific reasoning assistant. Given a description of the "
    "current state of knowledge in a research field and the conventional "
    "assumption being challenged, generate a novel hypothesis with "
    "step-by-step reasoning."
)

# Test on a few examples from our test set
test_path = f"{DATA_DIR}/test_ours.jsonl"
with open(test_path) as f:
    test_records = [json.loads(line) for line in f if line.strip()]

print(f"\\nEvaluating on {min(5, len(test_records))} test records...")

for i, record in enumerate(test_records[:5]):
    # Extract user message
    user_msg = [m["content"] for m in record["messages"] if m["role"] == "user"][0]
    ref_msg = [m["content"] for m in record["messages"] if m["role"] == "assistant"][0]

    # Generate
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=1024, temperature=0.7, top_p=0.9, do_sample=True,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    print(f"\\n{'='*60}")
    print(f"Test #{i+1}")
    print(f"{'='*60}")
    print(f"REFERENCE: {ref_msg[:200]}...")
    print(f"\\nGENERATED: {response[:200]}...")

# Generate outputs for full evaluation
print(f"\\nGenerating outputs for full test set ({len(test_records)} records)...")
generated_records = []
for i, record in enumerate(test_records):
    user_msg = [m["content"] for m in record["messages"] if m["role"] == "user"][0]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=1024, temperature=0.7, top_p=0.9, do_sample=True,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    gen_record = {
        "messages": record["messages"][:2] + [{"role": "assistant", "content": response}],
        "metadata": record.get("metadata", {}),
    }
    generated_records.append(gen_record)
    if (i+1) % 20 == 0:
        print(f"  Generated {i+1}/{len(test_records)}")

# Save generated outputs
gen_path = f"{OUTPUT_DIR}/generated_test_ours.jsonl"
with open(gen_path, "w") as f:
    for r in generated_records:
        f.write(json.dumps(r, ensure_ascii=False) + "\\n")
print(f"\\nSaved generated outputs to {gen_path}")

# Also generate for HypoGen test set
hg_test_path = f"{DATA_DIR}/test_hypogen.jsonl"
if os.path.exists(hg_test_path):
    with open(hg_test_path) as f:
        hg_test = [json.loads(line) for line in f if line.strip()]

    print(f"\\nGenerating HypoGen test outputs ({len(hg_test)} records)...")
    hg_generated = []
    for i, record in enumerate(hg_test):
        user_msg = [m["content"] for m in record["messages"] if m["role"] == "user"][0]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=1024, temperature=0.7, top_p=0.9, do_sample=True,
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)

        hg_generated.append({
            "messages": record["messages"][:2] + [{"role": "assistant", "content": response}],
            "metadata": record.get("metadata", {}),
        })

    gen_hg_path = f"{OUTPUT_DIR}/generated_test_hypogen.jsonl"
    with open(gen_hg_path, "w") as f:
        for r in hg_generated:
            f.write(json.dumps(r, ensure_ascii=False) + "\\n")
    print(f"Saved HypoGen generated outputs to {gen_hg_path}")

print("\\nDone! Run evaluate.py on the generated outputs for metrics.")
'''

# ============================================================
# CELL 7: Baseline Evaluation (no fine-tuning)
# ============================================================
BASELINE = '''
# Run this BEFORE any training to establish baseline
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, json, os

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
BASELINE_DIR = f"{OUTPUT_BASE}/baseline"
os.makedirs(BASELINE_DIR, exist_ok=True)

print(f"Loading base model {MODEL_NAME} for baseline evaluation...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

SYSTEM_PROMPT = (
    "You are a scientific reasoning assistant. Given a description of the "
    "current state of knowledge in a research field and the conventional "
    "assumption being challenged, generate a novel hypothesis with "
    "step-by-step reasoning."
)

# Generate baseline for both test sets
for test_name in ["test_ours", "test_hypogen"]:
    test_path = f"{DATA_DIR}/{test_name}.jsonl"
    if not os.path.exists(test_path):
        print(f"Skipping {test_name} -- not found")
        continue

    with open(test_path) as f:
        test_records = [json.loads(line) for line in f if line.strip()]

    print(f"\\nGenerating baseline for {test_name} ({len(test_records)} records)...")
    generated = []
    for i, record in enumerate(test_records):
        user_msg = [m["content"] for m in record["messages"] if m["role"] == "user"][0]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=1024, temperature=0.7, top_p=0.9, do_sample=True,
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)

        generated.append({
            "messages": record["messages"][:2] + [{"role": "assistant", "content": response}],
            "metadata": record.get("metadata", {}),
        })
        if (i+1) % 10 == 0:
            print(f"  Generated {i+1}/{len(test_records)}")

    out_path = f"{BASELINE_DIR}/baseline_{test_name}.jsonl"
    with open(out_path, "w") as f:
        for r in generated:
            f.write(json.dumps(r, ensure_ascii=False) + "\\n")
    print(f"Saved to {out_path}")

print("\\nBaseline generation complete!")
print("Run evaluate.py on baseline outputs to establish floor metrics.")
'''

# ============================================================
# Print instructions
# ============================================================
if __name__ == "__main__":
    print("""
COLAB PRO TRAINING INSTRUCTIONS
================================

1. Upload the 'prepared/' directory to Google Drive:
   /MyDrive/hypothesis-training/prepared/
     train_ours_only.jsonl
     train_combined.jsonl
     train_balanced.jsonl
     val.jsonl
     test_ours.jsonl
     test_hypogen.jsonl

2. Open Google Colab, select GPU runtime (A100 preferred)

3. Run the cells in order:
   Cell 1: Environment setup + mount Drive
   Cell 2: Configuration + verify data
   Cell 3: Select run (1, 2, 2b, or 3)
   Cell 7: BASELINE FIRST (run before any training!)
   Cell 4: Training (the main event)
   Cell 5: Merge adapter
   Cell 6: Quick evaluation

4. After all runs complete, download generated outputs
   and run evaluate.py locally for full metrics.

The cells above are stored as string variables. In Colab,
create separate code cells and paste the content of each
variable (SETUP_SCRIPT, CONFIG, RUN_SELECTION, etc.)

Alternatively, copy this file to Colab and exec() each cell:
    exec(SETUP_SCRIPT)
    exec(CONFIG)
    etc.
""")
