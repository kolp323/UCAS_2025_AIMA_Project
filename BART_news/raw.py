import os
import torch
from datasets import load_dataset
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)

# === 配置参数 ===
MODEL_NAME = "facebook/bart-base"
CSV_PATH = "../data/PENS/train.tsv"  # 改成你的路径
TEXT_COLUMN = "content"
TITLE_COLUMN = "title"

# === 1. 加载数据集 ===
print("🔹 Loading dataset...")
dataset = load_dataset("csv", data_files=CSV_PATH)["train"]
# # 打印列名用于调试
# print("Columns in dataset:", dataset.column_names)
# Columns in dataset: ['UserID\tClicknewsID\tdwelltime\texposure_time\tpos\tneg\tstart\tend\tdwelltime_pos']

# # 打印第一个样本查看实际内容
# print("First example:", dataset[0])

# === 2. 加载模型与分词器 ===
print("🔹 Loading tokenizer and model...")
tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)

# === 3. 预处理函数 ===
def preprocess(example):
    model_inputs = tokenizer(
        example[TEXT_COLUMN],
        max_length=512,
        padding="max_length",
        truncation=True
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            example[TITLE_COLUMN],
            max_length=64,
            padding="max_length",
            truncation=True
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# === 4. Tokenize Dataset ===
print("🔹 Tokenizing dataset...")
tokenized_dataset = dataset.map(preprocess, batched=True)

# === 5. 训练配置 ===
training_args = TrainingArguments(
    output_dir="./bart_pens_output",
    evaluation_strategy="no",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    logging_steps=20,
    save_strategy="epoch",
    learning_rate=5e-5,
    report_to="none"
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# === 6. Trainer 微调 ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

print("🚀 Starting training...")
trainer.train()

# === 7. 保存模型 ===
print("💾 Saving model...")
model.save_pretrained("./bart_pens_model")
tokenizer.save_pretrained("./bart_pens_model")

# === 8. 推理测试函数 ===
def generate_title(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(model.device)
    summary_ids = model.generate(inputs["input_ids"], max_length=64)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# 示例
test_text = "5月27日，北京发生大规模交通拥堵，数百辆车排队超过两个小时。"
print("📝 Example:")
print("Input:", test_text)
print("Generated Title:", generate_title(test_text))
