from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
# Use 4bit quantization to reduce memory usage. Can be False.
load_in_4bit = True

print("Loading unsloth/Meta-Llama-3.1-8B-bnb-4bit...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Do model patching and add fast LoRA weights
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",],
    lora_alpha=16,
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# Formatting prompt
prompt_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an ultra-fast Windows application launcher. Your ONLY job is to output the exact Windows command to fulfill the user's request. Do not output any conversational text, explanations, or formatting. Just the raw command.<|eot_id|><|start_header_id|>user<|end_header_id|>

{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{}<|eot_id|>"""


def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    outputs = examples["output"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        text = prompt_template.format(instruction, output)
        texts.append(text)
    return {"text": texts, }


print("Loading dataset launcher_dataset.jsonl...")
dataset = load_dataset(
    "json",
    data_files={
        "train": "launcher_dataset.jsonl"},
    split="train")
dataset = dataset.map(formatting_prompts_func, batched=True,)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,  # Can make training 5x faster for short sequences.
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
    ),
)

print("Starting training...")
trainer_stats = trainer.train()

print("Training complete! Exporting to 4-bit GGUF...")
# Export to 4-bit GGUF directly!
model.save_pretrained_gguf(
    "fast_launcher_model",
    tokenizer,
    quantization_method="q4_k_m")
print("GGUF export complete! You can now load fast_launcher_model/unsloth.Q4_K_M.gguf into Ollama.")
