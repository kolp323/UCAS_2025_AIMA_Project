"""
Data preprocessing module
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
    """Data preprocessor"""
    
    def __init__(self):
        self.logger = self._setup_logger()
        ensure_directories()
        
        # Load original data path
        self.base_data_dir = DATA_PATHS['base_data_dir']
        self.output_dir = DATA_PATHS['processed_data_dir']
        
        # Data containers
        self.news_data = {}
        self.news_content = {}
        self.user_histories = {}
        self.user_interests = {}
        self.test_samples = []
        
        # Initialize other attributes
        self.news_index = {}
        self.category_dict = {}
        self.word_dict = {}
        self.test_users = []
        self.test_samples_raw = []
        
    def _setup_logger(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def load_original_data(self):
        """Load original PENS data"""
        self.logger.info("Starting to load original PENS data...")
        
        try:
            data_dir = self.base_data_dir
            if not os.path.exists(data_dir):
                self.logger.error(f"Data directory does not exist: {data_dir}")
                return False
            
            # Load basic data dictionary
            dict_path = os.path.join(data_dir, 'dict.pkl')
            if os.path.exists(dict_path):
                with open(dict_path, 'rb') as f:
                    self.news_index, self.category_dict, self.word_dict = pickle.load(f)
                self.logger.info("Successfully loaded dictionary data")
            else:
                self.logger.error(f"Dictionary file does not exist: {dict_path}")
                return False
            
            # Load test user data
            test_users_path = os.path.join(data_dir, 'TestUsers.pkl')
            if os.path.exists(test_users_path):
                with open(test_users_path, 'rb') as f:
                    self.test_users = pickle.load(f)
                self.logger.info(f"Loaded test user data: {len(self.test_users)} users")
            else:
                self.logger.error(f"Test users file does not exist: {test_users_path}")
                return False
            
            # Load test samples
            test_samples_path = os.path.join(data_dir, 'TestSamples.pkl')
            if os.path.exists(test_samples_path):
                with open(test_samples_path, 'rb') as f:
                    self.test_samples_raw = pickle.load(f)
                self.logger.info(f"Loaded test samples: {len(self.test_samples_raw)} samples")
            else:
                self.logger.error(f"Test samples file does not exist: {test_samples_path}")
                return False
            
            # Load news data
            news_path = os.path.join(data_dir, 'news.pkl')
            if os.path.exists(news_path):
                with open(news_path, 'rb') as f:
                    self.news_content = pickle.load(f)
                self.logger.info(f"Loaded news content: {len(self.news_content)} news articles")
            else:
                self.logger.error(f"News data file does not exist: {news_path}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load original data: {str(e)}")
            return False
    
    def load_processed_tsv_data(self):
        """Load and process data directly from TSV files"""
        self.logger.info("Loading data from TSV files...")
        
        try:
            # Load news data
            news_file = os.path.join('../data/PENS', 'news.tsv')
            if os.path.exists(news_file):
                news_df = pd.read_csv(news_file, sep='\t')
                news_df.fillna(value=" ", inplace=True)
                
                # Build news data dictionary
                for i, row in news_df.iterrows():
                    doc_id, vert, _, title, snippet = row[:5]
                    self.news_content[doc_id] = {
                        'category': vert,
                        'title': title.lower() if isinstance(title, str) else "",
                        'body': snippet.lower() if isinstance(snippet, str) else ""
                    }
                
                self.logger.info(f"Loaded news data from TSV: {len(self.news_content)} articles")
            
            # Load test data
            test_file = os.path.join('../data/PENS', 'personalized_test.tsv')
            if os.path.exists(test_file):
                test_df = pd.read_csv(test_file, sep='\t')
                
                for i, row in test_df.iterrows():
                    user_id = i  # Use row index as user ID
                    click_history = row['clicknewsID'].split(',') if pd.notna(row['clicknewsID']) else []
                    pos_news = row['posnewID'].split(',') if pd.notna(row['posnewID']) else []
                    rewrite_titles = row['rewrite_titles'].split(';;') if pd.notna(row['rewrite_titles']) else []
                    
                    # Save user history
                    self.test_users.append(click_history)
                    
                    # Create test samples
                    for news_id, rewrite_title in zip(pos_news, rewrite_titles):
                        if news_id.strip() and rewrite_title.strip():
                            self.test_samples_raw.append([user_id, news_id, rewrite_title.lower()])
                
                self.logger.info(f"Loaded test data from TSV: {len(self.test_users)} users, {len(self.test_samples_raw)} samples")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load data from TSV: {str(e)}")
            return False
    
    def extract_user_histories_from_tsv(self):
        """Extract user histories from TSV data"""
        self.logger.info("Extracting user histories from TSV data...")
        
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
                self.user_histories[user_idx] = ["No history available"]
        
        self.logger.info(f"User history extraction completed: {len(self.user_histories)} users")
        return self.user_histories
    
    def extract_user_interests_from_tsv(self):
        """Extract user interests from TSV data"""
        self.logger.info("Extracting user interests from TSV data...")
        
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
        
        self.logger.info(f"User interest extraction completed: {len(self.user_interests)} users")
        return self.user_interests
    
    def prepare_test_samples_from_tsv(self):
        """Prepare test samples from TSV data"""
        self.logger.info("Preparing test samples from TSV data...")
        
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
                        'user_history': self.user_histories.get(user_idx, ["No history available"]),
                        'user_interests': self.user_interests.get(user_idx, {'categories': ['news'], 'primary_interest': 'news'})
                    }
                    
                    # Filter empty titles or too short content
                    if test_sample['original_title'] and len(test_sample['original_title'].strip()) > 5:
                        test_samples.append(test_sample)
                
            except Exception as e:
                self.logger.warning(f"Error processing sample {i}: {str(e)}")
                continue
        
        self.test_samples = test_samples
        self.logger.info(f"Test sample preparation completed: {len(test_samples)} samples")
        return test_samples
    
    def save_processed_data(self):
        """Save preprocessed data"""
        self.logger.info("Starting to save preprocessed data...")
        
        try:
            # Save user histories
            with open(os.path.join(self.output_dir, 'user_histories.json'), 'w', encoding='utf-8') as f:
                json.dump(self.user_histories, f, ensure_ascii=False, indent=2)
            
            # Save user interests
            with open(os.path.join(self.output_dir, 'user_interests.json'), 'w', encoding='utf-8') as f:
                json.dump(self.user_interests, f, ensure_ascii=False, indent=2)
            
            # Save test samples
            with open(os.path.join(self.output_dir, 'test_samples.json'), 'w', encoding='utf-8') as f:
                json.dump(self.test_samples, f, ensure_ascii=False, indent=2)
            
            # Save statistics
            stats = {
                'total_users': len(self.user_histories),
                'total_samples': len(self.test_samples),
                'avg_history_length': np.mean([len(hist) for hist in self.user_histories.values()]) if self.user_histories else 0,
                'categories': list(set([interest['primary_interest'] for interest in self.user_interests.values()])) if self.user_interests else []
            }
            
            with open(os.path.join(self.output_dir, 'data_stats.json'), 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            
            self.logger.info("Data saving completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save data: {str(e)}")
            return False
    
    def load_processed_data(self):
        """Load processed data"""
        try:
            with open(os.path.join(self.output_dir, 'test_samples.json'), 'r', encoding='utf-8') as f:
                self.test_samples = json.load(f)
            
            with open(os.path.join(self.output_dir, 'user_histories.json'), 'r', encoding='utf-8') as f:
                self.user_histories = json.load(f)
                
            with open(os.path.join(self.output_dir, 'user_interests.json'), 'r', encoding='utf-8') as f:
                self.user_interests = json.load(f)
                
            self.logger.info("Successfully loaded processed data")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load processed data: {str(e)}")
            return False
    
    def process_all_from_tsv(self):
        """Complete processing pipeline from TSV files"""
        self.logger.info("Starting TSV data processing pipeline...")
        
        # 1. Load TSV data
        if not self.load_processed_tsv_data():
            return False
        
        # 2. Extract user histories
        self.extract_user_histories_from_tsv()
        
        # 3. Extract user interests
        self.extract_user_interests_from_tsv()
        
        # 4. Prepare test samples
        self.prepare_test_samples_from_tsv()
        
        # 5. Save processed data
        self.save_processed_data()
        
        self.logger.info("TSV data processing pipeline completed")
        return True
    
    def process_all_from_pkl(self):
        """Complete processing pipeline from PKL files"""
        self.logger.info("Starting PKL data processing pipeline...")
        
        # 1. Load original data
        if not self.load_original_data():
            return False
        
        # 2. Extract user histories
        # ... Here we can call original methods, but need improvements
        
        return True

    def get_test_samples(self, limit: int = None) -> List[Dict[str, Any]]:
        """Get test samples
        
        Args:
            limit: Limit the number of returned samples, if None return all samples
            
        Returns:
            List of test samples
        """
        # If test samples are empty, try to load processed data
        if not hasattr(self, 'test_samples') or not self.test_samples:
            if not self.load_processed_data():
                # If loading fails, try to reprocess
                if not self.process_all_from_tsv():
                    self.logger.error("Unable to get test samples: both data loading and processing failed")
                    return []
        
        if limit is None:
            return self.test_samples
        else:
            return self.test_samples[:limit]

if __name__ == "__main__":
    processor = DataProcessor()
    
    # Try TSV processing method first
    success = processor.process_all_from_tsv()
    
    if success:
        print("Data preprocessing completed!")
        print(f"Processed {len(processor.test_samples)} test samples")
        print(f"Involving {len(processor.user_histories)} users")
    else:
        print("Data preprocessing failed, please check data files and path configuration") 