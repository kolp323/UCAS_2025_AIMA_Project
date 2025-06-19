#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
提示词生成器模块 - 支持不同模型类型的自适应提示词生成
"""

import logging
from typing import Dict, List, Any
from config import (PROMPT_CONFIG, API_CONFIG, 
                   get_system_prompt, get_user_prompt, is_reasoning_model)

class PromptGenerator:
    """提示词生成器 - 自适应支持推理模型和聊天模型"""
    
    def __init__(self):
        """初始化提示词生成器"""
        self.current_model = API_CONFIG['model']
        self.is_reasoning_model = is_reasoning_model(self.current_model)
        
        # 设置日志
        self.logger = logging.getLogger('prompt_generator')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        self.logger.info(f"初始化提示词生成器: {self.current_model} ({'推理模型' if self.is_reasoning_model else '聊天模型'})")
    
    def get_system_prompt(self, style: str = 'default') -> str:
        """获取系统提示词"""
        return get_system_prompt(self.current_model)
    
    def get_user_prompt(self, style: str = 'focused') -> str:
        """获取用户提示词模板"""
        return get_user_prompt(self.current_model, style)
    
    def get_batch_prompt_template(self) -> str:
        """获取批量处理提示词模板"""
        if self.is_reasoning_model:
            return """Create personalized English headlines for {batch_size} news articles based on user interests.

{batch_content}

Requirements:
- Each headline should be 8-20 words
- English only
- Personalized based on user interests
- Maintain accuracy to original content

Output format:
News 1: [Headline]
News 2: [Headline]
News 3: [Headline]
... (continue for all {batch_size} articles)"""
        else:
            return """Create personalized English headlines for these {batch_size} news articles:

{batch_content}

Generate 8-20 word English headlines. Format:
News 1: [Headline]
News 2: [Headline]
News 3: [Headline]"""
    
    def generate_single_prompt(self, sample: Dict[str, Any], style: str = 'focused') -> tuple[str, str]:
        """
        为单个样本生成提示词
        
        Args:
            sample: 样本数据
            style: 提示词风格 ('focused', 'enhanced', 'creative')
        
        Returns:
            tuple: (system_prompt, user_prompt)
        """
        system_prompt = self.get_system_prompt()
        user_prompt_template = self.get_user_prompt(style)
        
        # 构建用户历史字符串
        history_str = "\n".join([f"- {title}" for title in sample['user_history'][:10]])
        
        # 构建兴趣标签字符串  
        interests = sample['user_interests']
        interest_str = f"主要兴趣: {interests['primary_interest']}, 相关类别: {', '.join(interests['categories'])}"
        
        # 格式化用户提示词
        user_prompt = user_prompt_template.format(
            user_history=history_str,
            user_interests=interest_str,
            original_title=sample['original_title'],
            news_content=sample['news_body'][:400],  # 限制新闻内容长度
            news_category=sample['category']
        )
        
        return system_prompt, user_prompt
    
    def generate_batch_prompt(self, samples: List[Dict[str, Any]]) -> tuple[str, str]:
        """
        为批量样本生成提示词
        
        Args:
            samples: 样本数据列表
        
        Returns:
            tuple: (system_prompt, user_prompt)
        """
        system_prompt = self.get_system_prompt()
        batch_template = self.get_batch_prompt_template()
        
        # 构建批量内容
        batch_content_parts = []
        for i, sample in enumerate(samples, 1):
            # 简化用户兴趣描述
            interests = sample['user_interests']
            interest_summary = f"{interests['primary_interest']} ({', '.join(interests['categories'][:3])})"
            
            # 简化历史记录
            history_summary = "; ".join(sample['user_history'][:5])
            
            content_part = f"""News {i}:
  User interests: {interest_summary}
  Original title: {sample['original_title']}
  Content: {sample['news_body'][:200]}...
  Category: {sample['category']}"""
            
            batch_content_parts.append(content_part)
        
        batch_content = "\n\n".join(batch_content_parts)
        
        user_prompt = batch_template.format(
            batch_size=len(samples),
            batch_content=batch_content
        )
        
        return system_prompt, user_prompt
    
    def get_available_styles(self) -> List[str]:
        """获取可用的提示词风格列表"""
        model_type = 'reasoning_model' if self.is_reasoning_model else 'chat_model'
        return list(PROMPT_CONFIG['user_prompts'][model_type].keys())
    
    def update_model(self, model_name: str):
        """更新当前使用的模型"""
        self.current_model = model_name
        self.is_reasoning_model = is_reasoning_model(model_name)
        self.logger.info(f"更新模型: {model_name} ({'推理模型' if self.is_reasoning_model else '聊天模型'})") 