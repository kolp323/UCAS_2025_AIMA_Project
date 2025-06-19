"""
数据预处理模块
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import logging
from collections import Counter, defaultdict
import re
import json

from config import DATA_PATHS, DATA_CONFIG, ensure_directories

class DataProcessor:
    """数据预处理器"""
    
    def __init__(self):
        self.logger = self._setup_logger()
        ensure_directories()
        
        # 加载原始数据路径
        self.base_data_dir = DATA_PATHS['base_data_dir']
        self.output_dir = DATA_PATHS['processed_data_dir']
        
        # 数据容器
        self.news_data = {}
        self.news_content = {}
        self.user_histories = {}
        self.user_interests = {}
        self.test_samples = []
        
        # 初始化其他属性
        self.news_index = {}
        self.category_dict = {}
        self.word_dict = {}
        self.test_users = []
        self.test_samples_raw = []
        
    def _setup_logger(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def load_original_data(self):
        """加载原始PENS数据"""
        self.logger.info("开始加载原始PENS数据...")
        
        try:
            data_dir = self.base_data_dir
            if not os.path.exists(data_dir):
                self.logger.error(f"数据目录不存在: {data_dir}")
                return False
            
            # 加载基础数据字典
            dict_path = os.path.join(data_dir, 'dict.pkl')
            if os.path.exists(dict_path):
                with open(dict_path, 'rb') as f:
                    self.news_index, self.category_dict, self.word_dict = pickle.load(f)
                self.logger.info("成功加载词典数据")
            else:
                self.logger.error(f"词典文件不存在: {dict_path}")
                return False
            
            # 加载测试用户数据
            test_users_path = os.path.join(data_dir, 'TestUsers.pkl')
            if os.path.exists(test_users_path):
                with open(test_users_path, 'rb') as f:
                    self.test_users = pickle.load(f)
                self.logger.info(f"加载测试用户数据: {len(self.test_users)} 个用户")
            else:
                self.logger.error(f"测试用户文件不存在: {test_users_path}")
                return False
            
            # 加载测试样本
            test_samples_path = os.path.join(data_dir, 'TestSamples.pkl')
            if os.path.exists(test_samples_path):
                with open(test_samples_path, 'rb') as f:
                    self.test_samples_raw = pickle.load(f)
                self.logger.info(f"加载测试样本: {len(self.test_samples_raw)} 个样本")
            else:
                self.logger.error(f"测试样本文件不存在: {test_samples_path}")
                return False
            
            # 加载新闻数据
            news_path = os.path.join(data_dir, 'news.pkl')
            if os.path.exists(news_path):
                with open(news_path, 'rb') as f:
                    self.news_content = pickle.load(f)
                self.logger.info(f"加载新闻内容: {len(self.news_content)} 条新闻")
            else:
                self.logger.error(f"新闻数据文件不存在: {news_path}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"加载原始数据失败: {str(e)}")
            return False
    
    def load_processed_tsv_data(self):
        """直接从TSV文件加载和处理数据"""
        self.logger.info("从TSV文件加载数据...")
        
        try:
            # 加载新闻数据
            news_file = os.path.join('../data/PENS', 'news.tsv')
            if os.path.exists(news_file):
                news_df = pd.read_csv(news_file, sep='\t')
                news_df.fillna(value=" ", inplace=True)
                
                # 构建新闻数据字典
                for i, row in news_df.iterrows():
                    doc_id, vert, _, title, snippet = row[:5]
                    self.news_content[doc_id] = {
                        'category': vert,
                        'title': title.lower() if isinstance(title, str) else "",
                        'body': snippet.lower() if isinstance(snippet, str) else ""
                    }
                
                self.logger.info(f"从TSV加载新闻数据: {len(self.news_content)} 条")
            
            # 加载测试数据
            test_file = os.path.join('../data/PENS', 'personalized_test.tsv')
            if os.path.exists(test_file):
                test_df = pd.read_csv(test_file, sep='\t')
                
                for i, row in test_df.iterrows():
                    user_id = i  # 使用行索引作为用户ID
                    click_history = row['clicknewsID'].split(',') if pd.notna(row['clicknewsID']) else []
                    pos_news = row['posnewID'].split(',') if pd.notna(row['posnewID']) else []
                    rewrite_titles = row['rewrite_titles'].split(';;') if pd.notna(row['rewrite_titles']) else []
                    
                    # 保存用户历史
                    self.test_users.append(click_history)
                    
                    # 创建测试样本
                    for news_id, rewrite_title in zip(pos_news, rewrite_titles):
                        if news_id.strip() and rewrite_title.strip():
                            self.test_samples_raw.append([user_id, news_id, rewrite_title.lower()])
                
                self.logger.info(f"从TSV加载测试数据: {len(self.test_users)} 个用户, {len(self.test_samples_raw)} 个样本")
            
            return True
            
        except Exception as e:
            self.logger.error(f"从TSV加载数据失败: {str(e)}")
            return False
    
    def extract_user_histories_from_tsv(self):
        """从TSV数据提取用户历史"""
        self.logger.info("从TSV数据提取用户历史...")
        
        for user_idx, click_history in enumerate(self.test_users[:DATA_CONFIG['test_samples']]):
            clicked_news_titles = []
            
            for news_id in click_history[:DATA_CONFIG['max_user_history']]:
                if news_id in self.news_content:
                    news_info = self.news_content[news_id]
                    title = news_info['title']
                    if title and title.strip():
                        clicked_news_titles.append(title.strip())
            
            if clicked_news_titles:
                self.user_histories[user_idx] = clicked_news_titles
            else:
                self.user_histories[user_idx] = ["无历史记录"]
        
        self.logger.info(f"提取用户历史完成: {len(self.user_histories)} 个用户")
        return self.user_histories
    
    def extract_user_interests_from_tsv(self):
        """从TSV数据提取用户兴趣"""
        self.logger.info("从TSV数据提取用户兴趣...")
        
        for user_idx, click_history in enumerate(self.test_users[:DATA_CONFIG['test_samples']]):
            categories = []
            
            for news_id in click_history:
                if news_id in self.news_content:
                    news_info = self.news_content[news_id]
                    category = news_info['category']
                    if category:
                        categories.append(category)
            
            if categories:
                category_counts = Counter(categories)
                top_categories = [cat for cat, count in category_counts.most_common(3)]
                self.user_interests[user_idx] = {
                    'categories': top_categories,
                    'primary_interest': top_categories[0] if top_categories else 'news'
                }
            else:
                self.user_interests[user_idx] = {
                    'categories': ['news'],
                    'primary_interest': 'news'
                }
        
        self.logger.info(f"提取用户兴趣完成: {len(self.user_interests)} 个用户")
        return self.user_interests
    
    def prepare_test_samples_from_tsv(self):
        """从TSV数据准备测试样本"""
        self.logger.info("从TSV数据准备测试样本...")
        
        test_samples = []
        
        for i, sample in enumerate(self.test_samples_raw[:DATA_CONFIG['test_samples']]):
            try:
                user_idx, news_id, rewrite_title = sample
                
                if news_id in self.news_content:
                    news_info = self.news_content[news_id]
                    
                    test_sample = {
                        'user_idx': user_idx,
                        'news_id': news_id,
                        'original_title': news_info['title'],
                        'news_body': news_info['body'][:DATA_CONFIG['max_news_content_length']],
                        'category': news_info['category'],
                        'reference_title': rewrite_title,
                        'user_history': self.user_histories.get(user_idx, ["无历史记录"]),
                        'user_interests': self.user_interests.get(user_idx, {'categories': ['news'], 'primary_interest': 'news'})
                    }
                    
                    # 过滤空标题或过短的内容
                    if test_sample['original_title'] and len(test_sample['original_title'].strip()) > 5:
                        test_samples.append(test_sample)
                
            except Exception as e:
                self.logger.warning(f"处理样本 {i} 时出错: {str(e)}")
                continue
        
        self.test_samples = test_samples
        self.logger.info(f"准备测试样本完成: {len(test_samples)} 个样本")
        return test_samples
    
    def save_processed_data(self):
        """保存预处理后的数据"""
        self.logger.info("开始保存预处理数据...")
        
        try:
            # 保存用户历史
            with open(os.path.join(self.output_dir, 'user_histories.json'), 'w', encoding='utf-8') as f:
                json.dump(self.user_histories, f, ensure_ascii=False, indent=2)
            
            # 保存用户兴趣
            with open(os.path.join(self.output_dir, 'user_interests.json'), 'w', encoding='utf-8') as f:
                json.dump(self.user_interests, f, ensure_ascii=False, indent=2)
            
            # 保存测试样本
            with open(os.path.join(self.output_dir, 'test_samples.json'), 'w', encoding='utf-8') as f:
                json.dump(self.test_samples, f, ensure_ascii=False, indent=2)
            
            # 保存统计信息
            stats = {
                'total_users': len(self.user_histories),
                'total_samples': len(self.test_samples),
                'avg_history_length': np.mean([len(hist) for hist in self.user_histories.values()]) if self.user_histories else 0,
                'categories': list(set([interest['primary_interest'] for interest in self.user_interests.values()])) if self.user_interests else []
            }
            
            with open(os.path.join(self.output_dir, 'data_stats.json'), 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            
            self.logger.info("数据保存完成")
            return True
            
        except Exception as e:
            self.logger.error(f"保存数据失败: {str(e)}")
            return False
    
    def load_processed_data(self):
        """加载已处理的数据"""
        try:
            with open(os.path.join(self.output_dir, 'test_samples.json'), 'r', encoding='utf-8') as f:
                self.test_samples = json.load(f)
            
            with open(os.path.join(self.output_dir, 'user_histories.json'), 'r', encoding='utf-8') as f:
                self.user_histories = json.load(f)
                
            with open(os.path.join(self.output_dir, 'user_interests.json'), 'r', encoding='utf-8') as f:
                self.user_interests = json.load(f)
                
            self.logger.info("成功加载已处理的数据")
            return True
        except Exception as e:
            self.logger.error(f"加载已处理数据失败: {str(e)}")
            return False
    
    def process_all_from_tsv(self):
        """从TSV文件的完整处理流程"""
        self.logger.info("开始TSV数据处理流程...")
        
        # 1. 加载TSV数据
        if not self.load_processed_tsv_data():
            return False
        
        # 2. 提取用户历史
        self.extract_user_histories_from_tsv()
        
        # 3. 提取用户兴趣
        self.extract_user_interests_from_tsv()
        
        # 4. 准备测试样本
        self.prepare_test_samples_from_tsv()
        
        # 5. 保存处理后的数据
        self.save_processed_data()
        
        self.logger.info("TSV数据处理流程完成")
        return True
    
    def process_all_from_pkl(self):
        """从PKL文件的完整处理流程"""
        self.logger.info("开始PKL数据处理流程...")
        
        # 1. 加载原始数据
        if not self.load_original_data():
            return False
        
        # 2. 提取用户历史
        # ... 这里可以调用原来的方法，但需要改进
        
        return True

    def get_test_samples(self, limit: int = None) -> List[Dict[str, Any]]:
        """获取测试样本
        
        Args:
            limit: 限制返回的样本数量，如果为None则返回所有样本
            
        Returns:
            测试样本列表
        """
        # 如果测试样本为空，尝试加载已处理的数据
        if not hasattr(self, 'test_samples') or not self.test_samples:
            if not self.load_processed_data():
                # 如果加载失败，尝试重新处理
                if not self.process_all_from_tsv():
                    self.logger.error("无法获取测试样本：数据加载和处理都失败")
                    return []
        
        if limit is None:
            return self.test_samples
        else:
            return self.test_samples[:limit]

if __name__ == "__main__":
    processor = DataProcessor()
    
    # 优先尝试TSV处理方式
    success = processor.process_all_from_tsv()
    
    if success:
        print("数据预处理完成!")
        print(f"处理了 {len(processor.test_samples)} 个测试样本")
        print(f"涉及 {len(processor.user_histories)} 个用户")
    else:
        print("数据预处理失败，请检查数据文件和路径配置") 