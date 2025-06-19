import json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BartForConditionalGeneration, BartTokenizer, AdamW
from rouge_score import rouge_scorer

# ============================== 数据预处理 ==============================
class NewsDataProcessor:
    def __init__(self):
        self.news_map = {}
        self.user_history = {}
    
    def load_news_data(self, path):
        """解析news.txt构建新闻内容映射"""
        with open(path) as f:
            data = json.load(f)
        for item in data:
            self.news_map[item['News ID']] = {
                'body': item['News body'],
                'headline': item['Headline']
            }
    
    def load_user_data(self, train_path, valid_path):
        """解析train/valid.txt构建用户历史"""
        for path in [train_path, valid_path]:
            with open(path) as f:
                for line in f:
                    record = json.loads(line)
                    clicks = record['ClicknewsID'].split()
                    self.user_history[record['UserID']] = clicks
    
    def load_test_data(self, test_path):
        """解析personalized_test.txt构建测试集"""
        test_data = []
        with open(test_path) as f:
            for line in f:
                record = json.loads(line)
                user = record['userid']
                clicked = record['clicknewsID'].split(',')
                targets = record['posnewID'].split(',')
                rewrites = record['rewrite_titles'].split(';;')
                
                for target_id, rewrite in zip(targets, rewrites):
                    test_data.append({
                        'user_id': user,
                        'news_id': target_id,
                        'rewrite': rewrite.strip()
                    })
        return test_data

# ============================== 数据集类 ==============================
class NewsDataset(Dataset):
    def __init__(self, data, processor, tokenizer, max_length=1024, mode='train'):
        self.data = data
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        news_id = item['news_id']
        news_data = self.processor.news_map[news_id]
        
        # 基础输入：新闻正文
        inputs = news_data['body']
        
        if self.mode == 'personalized':
            # 个性化增强：添加用户历史
            user_history = self.processor.user_history.get(item['user_id'], [])
            history_titles = [
                self.processor.news_map.get(nid, {}).get('headline', '')
                for nid in user_history[-3:]  # 取最近3条
            ]
            user_context = " | ".join(history_titles)
            inputs = f"[USER_CONTEXT] {user_context} [SEP] {inputs}"
        
        # 目标输出
        target = item['rewrite'] if self.mode == 'personalized' else news_data['headline']
        
        # Tokenization
        model_inputs = self.tokenizer(
            inputs,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                target,
                max_length=128,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )["input_ids"]
        
        return {
            'input_ids': model_inputs['input_ids'].squeeze(),
            'attention_mask': model_inputs['attention_mask'].squeeze(),
            'labels': labels.squeeze()
        }

# ============================== 训练函数 ==============================
def train_model(model, train_loader, val_loader, epochs=3, lr=5e-5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        
        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                val_loss += outputs.loss.item()
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")
    
    return model

# ============================== 评估函数 ==============================
def evaluate_rouge(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    results = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].cpu().numpy()
            
            # 生成预测
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=128,
                num_beams=4
            )
            
            # 解码文本
            preds = tokenizer.batch_decode(generated, skip_special_tokens=True)
            targets = tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            # 计算ROUGE
            for pred, target in zip(preds, targets):
                score = scorer.score(target, pred)
                results.append({
                    'rouge1': score['rouge1'].fmeasure,
                    'rouge2': score['rouge2'].fmeasure,
                    'rougeL': score['rougeL'].fmeasure
                })
    
    # 汇总结果
    avg_rouge = {
        'rouge1': sum(r['rouge1'] for r in results) / len(results),
        'rouge2': sum(r['rouge2'] for r in results) / len(results),
        'rougeL': sum(r['rougeL'] for r in results) / len(results)
    }
    return avg_rouge

# ============================== 主流程 ==============================
if __name__ == "__main__":
    # 初始化组件
    processor = NewsDataProcessor()
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    
    # 加载数据
    processor.load_news_data('../data/PENS/news.tsv')
    processor.load_user_data('../data/PENS/train.tsv', '../data/PENS/valid.tsv')
    test_data = processor.load_test_data('../data/PENS/personalized_test.tsv')
    
    # 划分数据集
    train_size = int(0.8 * len(test_data))
    train_data = test_data[:train_size]
    val_data = test_data[train_size:]
    
    # 创建数据集 - 基础训练
    base_train_set = NewsDataset(train_data, processor, tokenizer, mode='base')
    base_val_set = NewsDataset(val_data, processor, tokenizer, mode='base')
    
    # 创建数据集 - 个性化训练
    personalized_train_set = NewsDataset(train_data, processor, tokenizer, mode='personalized')
    personalized_val_set = NewsDataset(val_data, processor, tokenizer, mode='personalized')
    
    # 创建数据加载器
    base_train_loader = DataLoader(base_train_set, batch_size=8, shuffle=True)
    base_val_loader = DataLoader(base_val_set, batch_size=8)
    personalized_train_loader = DataLoader(personalized_train_set, batch_size=4, shuffle=True)
    personalized_val_loader = DataLoader(personalized_val_set, batch_size=4)
    
    # 阶段1：基础标题生成训练
    print("=== Training Base Title Generation ===")
    base_model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
    base_model = train_model(base_model, base_train_loader, base_val_loader, epochs=3)
    torch.save(base_model.state_dict(), 'base_bart_title_gen.pth')
    
    # 阶段2：个性化标题生成训练
    print("\n=== Training Personalized Title Generation ===")
    personalized_model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
    personalized_model = train_model(personalized_model, personalized_train_loader, personalized_val_loader, epochs=5)
    torch.save(personalized_model.state_dict(), 'personalized_bart_title_gen.pth')
    
    # 评估个性化模型
    print("\n=== Evaluating Personalized Model ===")
    test_set = NewsDataset(test_data, processor, tokenizer, mode='personalized')
    test_loader = DataLoader(test_set, batch_size=4)
    rouge_scores = evaluate_rouge(personalized_model, test_loader)
    
    print(f"\nFinal ROUGE Scores:")
    print(f"ROUGE-1: {rouge_scores['rouge1']:.4f}")
    print(f"ROUGE-2: {rouge_scores['rouge2']:.4f}")
    print(f"ROUGE-L: {rouge_scores['rougeL']:.4f}")
    
    # 示例：为特定用户生成标题
    def generate_personalized_title(user_id, news_id):
        news_data = processor.news_map[news_id]
        user_history = processor.user_history.get(user_id, [])
        
        # 构建个性化输入
        history_titles = [
            processor.news_map.get(nid, {}).get('headline', '')
            for nid in user_history[-3:]
        ]
        context = " | ".join(history_titles)
        input_text = f"[USER_CONTEXT] {context} [SEP] {news_data['body']}"
        
        # 生成标题
        inputs = tokenizer(input_text, return_tensors='pt', max_length=1024, truncation=True)
        generated = personalized_model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=128,
            num_beams=4
        )
        return tokenizer.decode(generated[0], skip_special_tokens=True)
    
    # 测试示例
    sample_user = test_data[0]['user_id']
    sample_news = test_data[0]['news_id']
    print(f"\nGenerated title for user {sample_user}:")
    print(generate_personalized_title(sample_user, sample_news))