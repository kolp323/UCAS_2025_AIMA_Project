#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prompt generator module - Supports adaptive prompt generation for different model types
"""

import logging
from typing import Dict, List, Any
from config import (PROMPT_CONFIG, API_CONFIG, 
                   get_system_prompt, get_user_prompt, is_reasoning_model)

class PromptGenerator:
    """Prompt generator - Adaptively supports reasoning models and chat models"""
    
    def __init__(self):
        """Initialize prompt generator"""
        self.current_model = API_CONFIG['model']
        self.is_reasoning_model = is_reasoning_model(self.current_model)
        
        # Setup logging
        self.logger = logging.getLogger('prompt_generator')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        self.logger.info(f"Initialize prompt generator: {self.current_model} ({'reasoning model' if self.is_reasoning_model else 'chat model'})")
    
    def get_system_prompt(self, style: str = 'default') -> str:
        """Get system prompt"""
        return get_system_prompt(self.current_model)
    
    def get_user_prompt(self, style: str = 'focused') -> str:
        """Get user prompt template"""
        return get_user_prompt(self.current_model, style)
    
    def get_batch_prompt_template(self) -> str:
        """Get batch processing prompt template"""
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
        Generate prompt for single sample
        
        Args:
            sample: Sample data
            style: Prompt style ('focused', 'enhanced', 'creative')
        
        Returns:
            tuple: (system_prompt, user_prompt)
        """
        system_prompt = self.get_system_prompt()
        user_prompt_template = self.get_user_prompt(style)
        
        # Build user history string
        history_str = "\n".join([f"- {title}" for title in sample['user_history'][:10]])
        
        # Build interest tags string
        interests = sample['user_interests']
        interest_str = f"Primary interest: {interests['primary_interest']}, Related categories: {', '.join(interests['categories'])}"
        
        # Format user prompt
        user_prompt = user_prompt_template.format(
            user_history=history_str,
            user_interests=interest_str,
            original_title=sample['original_title'],
            news_content=sample['news_body'][:400],  # Limit news content length
            news_category=sample['category']
        )
        
        return system_prompt, user_prompt
    
    def generate_batch_prompt(self, samples: List[Dict[str, Any]]) -> tuple[str, str]:
        """
        Generate prompt for batch samples
        
        Args:
            samples: List of sample data
        
        Returns:
            tuple: (system_prompt, user_prompt)
        """
        system_prompt = self.get_system_prompt()
        batch_template = self.get_batch_prompt_template()
        
        # Build batch content
        batch_content_parts = []
        for i, sample in enumerate(samples, 1):
            # Simplify user interest description
            interests = sample['user_interests']
            interest_summary = f"{interests['primary_interest']} ({', '.join(interests['categories'][:3])})"
            
            # Simplify history records
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
        """Get list of available prompt styles"""
        model_type = 'reasoning_model' if self.is_reasoning_model else 'chat_model'
        return list(PROMPT_CONFIG['user_prompts'][model_type].keys())
    
    def update_model(self, model_name: str):
        """Update currently used model"""
        self.current_model = model_name
        self.is_reasoning_model = is_reasoning_model(model_name)
        self.logger.info(f"Update model: {model_name} ({'reasoning model' if self.is_reasoning_model else 'chat model'})") 