from datasets import load_dataset

CSV_PATH = "../data/PENS/train.tsv"  # 改成你的路径

# === 加载数据集 ===
print("🔹 Loading dataset...")
dataset = load_dataset("csv", data_files=CSV_PATH, sep="\t")["train"]

# 打印列名
print("Columns in dataset:", dataset.column_names)

# 打印第一个样本查看实际内容
print("First example:", dataset[0])

