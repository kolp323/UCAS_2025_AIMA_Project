#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LLM API client module - Encapsulates DeepSeek API calls
Supports adaptive processing for reasoning models and normal chat models
"""

import os
import json
import time
import logging
import re
from typing import Optional, Dict, Any, List
from openai import OpenAI
from config import API_CONFIG, is_reasoning_model, get_optimal_max_tokens, get_available_models, get_model_config

class LLMClient:
    """LLM API client"""
    
    def __init__(self, model_name: str = None):
        """Initialize LLM client"""
        self.api_config = API_CONFIG.copy()
        
        # Setup logging (initialize first)
        self._setup_logger()
        
        # Setup model
        if model_name and model_name in get_available_models():
            self.api_config['model'] = model_name
        elif model_name:
            self.logger.warning(f"Model {model_name} not available, using default model {self.api_config['model']}")
        
        self.client = OpenAI(
            api_key=self.api_config['api_key'],
            base_url=self.api_config['base_url']
        )
        
        # Usage status
        self.total_tokens_used = 0
        self.total_requests = 0
        self.failed_requests = 0
        
        # Determine model type
        self.is_reasoning_model = is_reasoning_model(self.api_config['model'])
        self.logger.info(f"Initialize LLM client: {self.api_config['model']} ({'reasoning model' if self.is_reasoning_model else 'chat model'})")
    
    def switch_model(self, model_name: str) -> bool:
        """Switch model"""
        if model_name in get_available_models():
            old_model = self.api_config['model']
            self.api_config['model'] = model_name
            self.is_reasoning_model = is_reasoning_model(model_name)
            self.logger.info(f"Model switched: {old_model} -> {model_name} ({'reasoning model' if self.is_reasoning_model else 'chat model'})")
            return True
        else:
            self.logger.error(f"Model {model_name} not in available list: {get_available_models()}")
            return False
    
    def get_current_model(self) -> str:
        """Get currently used model"""
        return self.api_config['model']
    
    def get_model_info(self) -> dict:
        """Get detailed information about current model"""
        return get_model_config(self.api_config['model'])

    def _setup_logger(self):
        """Setup logger"""
        self.logger = logging.getLogger('llm_client')
        # Avoid adding duplicate handlers
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
            self.logger.propagate = False  # Prevent propagation to root logger to avoid duplicate output

    def estimate_tokens(self, text: str) -> int:
        """Roughly estimate token count"""
        return int(len(text) * 1.5)

    def check_token_limit(self, tokens: int) -> bool:
        """Check if token limit is exceeded"""
        if tokens > self.api_config['max_tokens_per_request']:
            self.logger.warning(f"Estimated token count {tokens} exceeds single request limit {self.api_config['max_tokens_per_request']}")
            return False
        
        if self.total_tokens_used + tokens > self.api_config['max_tokens']:
            self.logger.warning(f"Estimated total token count will exceed limit {self.api_config['max_tokens']}")
            return False
        
        return True

    def chat_completion(self, messages, **kwargs):
        """Send chat completion request - adaptive processing for different model types"""
        # Automatically adjust max_tokens based on model type
        if 'max_tokens' not in kwargs:
            kwargs['max_tokens'] = get_optimal_max_tokens(self.api_config['model'])
        
        if not self.check_token_limit(kwargs.get('max_tokens', 100)):
            return None

        params = {
            'model': self.api_config['model'],
            'messages': messages,
            'temperature': kwargs.get('temperature', 0.7),
            'max_tokens': kwargs.get('max_tokens', 100),
        }

        for attempt in range(self.api_config['max_retries']):
            try:
                self.logger.info(f"Sending API request (attempt {attempt + 1}/{self.api_config['max_retries']})")
                
                response = self.client.chat.completions.create(**params)
                
                self.total_requests += 1
                
                # Adaptive content extraction
                extracted_content = self._extract_content_adaptive(response)
                
                if extracted_content:
                    # Record token usage
                    if hasattr(response, 'usage') and response.usage:
                        tokens_used = response.usage.total_tokens
                        self.total_tokens_used += tokens_used
                        self.logger.info(f"This request used {tokens_used} tokens, total {self.total_tokens_used} tokens")
                    
                    return extracted_content

                self.logger.warning("API response is empty or unable to extract content")
                return None

            except Exception as e:
                self.logger.error(f"API request failed (attempt {attempt + 1}): {e}")
                if attempt == self.api_config['max_retries'] - 1:
                    return None
                time.sleep(2 ** attempt)  # Exponential backoff

        return None

    def _extract_content_adaptive(self, response):
        """Adaptive content extraction - choose optimal strategy based on model type and task type"""
        message = response.choices[0].message
        
        # For reasoning models, need to distinguish task types
        if self.is_reasoning_model:
            # First check if content field has valid content
            if message.content and message.content.strip():
                content = message.content.strip()
                
                # For evaluation tasks (containing numbers and commas), prioritize content field
                if self._is_valid_evaluation_result(content):
                    cleaned_content = self.clean_generated_title(content)
                    self.logger.info(f"Get evaluation result from content field (reasoning model): {cleaned_content[:100]}...")
                    return cleaned_content
                
                # For title generation tasks, check if it's a valid title
                elif self._is_valid_title_content(content):
                    cleaned_title = self.clean_generated_title(content)
                    self.logger.info(f"Get title from content field (reasoning model): {cleaned_title}")
                    return cleaned_title
            
            # If content field is invalid or empty, consider reasoning field (only for title generation)
            if hasattr(message, 'model_extra') and message.model_extra and 'reasoning' in message.model_extra:
                reasoning = message.model_extra['reasoning']
                if reasoning:
                    # Try to extract title from reasoning process (only for title generation tasks)
                    extracted_title = self.extract_title_from_reasoning(reasoning)
                    if extracted_title:
                        self.logger.info(f"Extract title from reasoning field (reasoning model): {extracted_title}")
                        return extracted_title
                    else:
                        self.logger.warning("Unable to extract valid title from reasoning field")
                else:
                    self.logger.warning("Reasoning field is empty")
        else:
            # For non-reasoning models, prioritize content field
            if message.content and message.content.strip():
                content = message.content.strip()
                cleaned_title = self.clean_generated_title(content)
                self.logger.info(f"Get content from content field (chat model): {cleaned_title}")
                return cleaned_title
        
        # If all strategies fail
        self.logger.warning(f"Unable to extract content - content: '{message.content}', is_reasoning: {self.is_reasoning_model}, has_model_extra: {hasattr(message, 'model_extra')}")
        return None

    def _is_valid_title_content(self, content: str) -> bool:
        """Check if content is a valid title rather than reasoning process"""
        if not content:
            return False
        
        # Check if it contains reasoning keywords
        reasoning_indicators = [
            'the user', 'personalized', 'headline', 'create', 'generate',
            'we need', 'based on', 'i think', 'let me', 'alternative options'
        ]
        
        if any(indicator in content.lower() for indicator in reasoning_indicators):
            return False
        
        # Check reasonable length
        words = content.split()
        return 5 <= len(words) <= 25
    
    def _is_valid_evaluation_result(self, content: str) -> bool:
        """Check if content is a valid evaluation result (numbers, scores, etc.)"""
        if not content:
            return False
        
        content = content.strip()
        
        # Check if it contains number and comma patterns (evaluation results)
        import re
        # Look for number,number,number patterns
        number_pattern = r'^\s*(\d+(?:\.\d+)?(?:\s*,\s*\d+(?:\.\d+)?)*)\s*$'
        if re.match(number_pattern, content):
            return True
        
        # Check if it contains simple numeric output related to scoring
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if re.match(r'^\s*\d+(?:\.\d+)?(?:\s*,\s*\d+(?:\.\d+)?)+\s*$', line):
                return True
        
        return False

    def generate_personalized_title(self, sample: Dict[str, Any], system_prompt: str, user_prompt: str) -> Optional[str]:
        """Generate personalized title for single sample"""
        
        # Check if user_prompt is already a complete prompt
        if '{' not in user_prompt:
            # Already a complete prompt, use directly
            formatted_prompt = user_prompt
        else:
            # Is a template, needs formatting
            # Build user history string
            history_str = "\n".join([f"- {title}" for title in sample['user_history'][:10]])  # Limit history records
            
            # Build interest tags string
            interests = sample['user_interests']
            interest_str = f"Primary interest: {interests['primary_interest']}, Related categories: {', '.join(interests['categories'])}"
            
            # Format user prompt
            try:
                formatted_prompt = user_prompt.format(
                    user_history=history_str,
                    user_interests=interest_str,
                    original_title=sample['original_title'],
                    news_content=sample['news_body'][:400],  # Limit news content length
                    news_category=sample['category']
                )
            except KeyError as e:
                self.logger.error(f"Prompt formatting failed: {e}")
                return None
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": formatted_prompt}
        ]
        
        # Automatically adjust max_tokens based on model type
        optimal_tokens = get_optimal_max_tokens(self.api_config['model'])
        return self.chat_completion(messages, max_tokens=optimal_tokens)
    
    def generate_batch_titles(self, samples: List[Dict[str, Any]], system_prompt: str, batch_prompt_template: str) -> List[Optional[str]]:
        """Generate personalized titles in batch"""
        
        batch_content = ""
        for i, sample in enumerate(samples, 1):
            # Build user history (simplified version)
            history_summary = f"Recent reading: {', '.join(sample['user_history'][:3])}"
            interest_summary = sample['user_interests']['primary_interest']
            
            batch_content += f"""
News{i}:
- User interest: {interest_summary}
- User history: {history_summary}
- Original title: {sample['original_title']}
- News category: {sample['category']}
- Content summary: {sample['news_body'][:200]}...

"""
        
        formatted_prompt = batch_prompt_template.format(
            batch_size=len(samples),
            batch_content=batch_content
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": formatted_prompt}
        ]
        
        response = self.chat_completion(messages, max_tokens=500 * len(samples))
        
        if response:
            return self._parse_batch_response(response, len(samples))
        else:
            return [None] * len(samples)
    
    def _parse_batch_response(self, response: str, expected_count: int) -> List[Optional[str]]:
        """Parse batch response"""
        results = [None] * expected_count
        
        try:
            # Strategy 1: Parse "News X: [title]" format
            import re
            news_pattern = r'News\s+(\d+):\s*([^\n]+?)(?=\s*News\s+\d+:|$)'
            matches = re.findall(news_pattern, response, re.IGNORECASE | re.DOTALL)
            
            for match in matches:
                try:
                    idx = int(match[0]) - 1
                    if 0 <= idx < expected_count:
                        title = match[1].strip()
                        # Clean title
                        title = re.sub(r'^["\']|["\']$', '', title)  # Remove leading/trailing quotes
                        title = re.sub(r'\s+', ' ', title)  # Normalize spaces
                        results[idx] = title
                except (ValueError, IndexError):
                    continue
            
            # Strategy 2: If strategy 1 fails, try line-by-line parsing
            if all(r is None for r in results):
                lines = response.strip().split('\n')
                for line in lines:
                    line = line.strip()
                    # Match "NewsX:" or "News X:" or number-starting format
                    if re.match(r'^(?:News)\s*\d+\s*[:\-]', line, re.IGNORECASE):
                        parts = re.split(r'[:\-]', line, 1)
                        if len(parts) == 2:
                            # Extract number
                            num_match = re.search(r'\d+', parts[0])
                            if num_match:
                                try:
                                    idx = int(num_match.group()) - 1
                                    if 0 <= idx < expected_count:
                                        title = parts[1].strip()
                                        title = re.sub(r'^["\'\[\]]+|["\'\[\]]+$', '', title)
                                        results[idx] = title
                                except ValueError:
                                    continue
            
            # Log parsing results
            parsed_count = sum(1 for r in results if r is not None)
            self.logger.info(f"Batch parsing: expected {expected_count}, successfully parsed {parsed_count}")
            
            for i, result in enumerate(results):
                if result:
                    self.logger.info(f"  Batch title {i+1}: {result}")
                    
        except Exception as e:
            self.logger.error(f"Failed to parse batch response: {str(e)}")
        
        return results
    
    def parse_multiple_titles(self, response: str) -> List[str]:
        """Parse response containing multiple title options"""
        titles = []
        
        try:
            lines = response.strip().split('\n')
            for line in lines:
                line = line.strip()
                # Match formats like "1. [title]" or "1. title"
                if line and (line[0].isdigit() or line.startswith('-')):
                    # Extract title part
                    if '.' in line:
                        title = line.split('.', 1)[1].strip()
                    elif '-' in line:
                        title = line.split('-', 1)[1].strip()
                    else:
                        title = line
                    
                    # Clean format
                    title = title.replace('[', '').replace(']', '').strip()
                    if title:
                        titles.append(title)
        except Exception as e:
            self.logger.error(f"Failed to parse multiple title response: {str(e)}")
        
        return titles[:3]  # Return at most 3 titles
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return {
            'total_requests': self.total_requests,
            'failed_requests': self.failed_requests,
            'success_rate': (self.total_requests - self.failed_requests) / max(self.total_requests, 1),
            'total_tokens_used': self.total_tokens_used,
            'remaining_tokens': max(0, self.api_config['max_tokens'] - self.total_tokens_used)
        }
    
    def reset_stats(self):
        """Reset statistics"""
        self.total_tokens_used = 0
        self.total_requests = 0
        self.failed_requests = 0
    
    def clean_generated_title(self, title: str) -> str:
        """Clean generated title"""
        if not title:
            return None
            
        # Remove common prefixes and suffixes
        title = title.strip()
        
        # Remove quotes
        if title.startswith('"') and title.endswith('"'):
            title = title[1:-1]
        elif title.startswith("'") and title.endswith("'"):
            title = title[1:-1]
        
        # Remove possible numbers or markers
        title = re.sub(r'^[\d\.\-\*\+\s]+', '', title)
        
        # Remove special markers
        title = re.sub(r'\*[^*]*\*', '', title)  # Remove *markers*
        title = re.sub(r'^\w+\s*mentally[^:]*:', '', title)  # Remove "Drafts mentally:" type
        
        # Remove HTML tags
        title = re.sub(r'<[^>]+>', '', title)
        
        # Normalize spaces
        title = re.sub(r'\s+', ' ', title).strip()
        
        # Ensure first letter is capitalized
        if title:
            title = title[0].upper() + title[1:] if len(title) > 1 else title.upper()
        
        return title
    
    def extract_title_from_reasoning(self, reasoning: str) -> Optional[str]:
        """Extract actual title from reasoning field"""
        if not reasoning:
            return None
            
        lines = [line.strip() for line in reasoning.split('\n') if line.strip()]
        
        # Strategy 1: Find content in quotes (most reliable)
        quotes = re.findall(r'"([^"]*)"', reasoning)
        for quote in quotes:
            words = quote.split()
            # Check if it's reasonable title length
            if 6 <= len(words) <= 25:
                # Ensure it's not a repetition of original title or reasoning text
                if not any(phrase in quote.lower() for phrase in [
                    'the original', 'high-stakes legal', 'personalized', 'headline',
                    'user interested', 'create', 'generate', 'we need', 'based on'
                ]):
                    self.logger.info(f"Extract title from quotes: {quote}")
                    return quote
        
        # Strategy 2: Find possible titles in last few lines
        for line in reversed(lines[-15:]):  # Check last 15 lines
            # Skip obvious reasoning text
            if any(phrase in line.lower() for phrase in [
                'the user', 'we need', 'based on', 'this', 'therefore', 
                'critical instructions', 'since', 'in this context', 'i think',
                'let me', 'so the', 'given that', 'drafts mentally', 'too long',
                'trim', 'here', 'now', 'final', 'output', 'personalized',
                'headline', 'generate', 'create'
            ]) or line.startswith(('-', '*', '•', '1.', '2.', '3.')):
                continue
                
            words = line.split()
            # Check if it's reasonable title length and format
            if 6 <= len(words) <= 20:
                # Ensure it contains actual content rather than meta-description
                # Check if it contains news-related vocabulary
                news_indicators = [
                    'report', 'announce', 'reveal', 'confirm', 'break', 'launch',
                    'win', 'lose', 'face', 'plan', 'set', 'expected', 'new',
                    'first', 'major', 'top', 'best', 'worst', 'latest', 'says',
                    'shows', 'finds', 'study', 'data', 'market', 'company',
                    'government', 'court', 'legal', 'law', 'rule', 'policy'
                ]
                
                if any(indicator in line.lower() for indicator in news_indicators):
                    self.logger.info(f"Extract title from reasoning end: {line}")
                    return line
        
        # Strategy 3: Find sentences that match news title format
        for line in lines:
            # Skip obvious reasoning beginnings
            if line.startswith(('The', 'We', 'Since', 'Critical', '-', '*', 'I ', 'Let ')) or \
               any(phrase in line.lower() for phrase in [
                   'personalized', 'headline', 'user interested', 'create', 'generate',
                   'instructions', 'format', 'output', 'drafts mentally'
               ]):
                continue
                
            words = line.split()
            if 8 <= len(words) <= 18:
                # Check if it has verb and noun combinations (news title characteristics)
                has_verb = any(word.lower() in ['faces', 'wins', 'loses', 'announces', 
                                              'reveals', 'confirms', 'breaks', 'launches',
                                              'reports', 'shows', 'finds', 'says'] for word in words)
                has_noun = any(word.lower() in ['company', 'government', 'court', 'study',
                                              'market', 'report', 'news', 'data'] for word in words)
                
                if has_verb or has_noun or ':' in line:
                    self.logger.info(f"Extract title candidate from reasoning: {line}")
                    return line
        
        # Strategy 4: If all fail, find lines starting with uppercase letters (possibly titles)
        for line in lines:
            if line and line[0].isupper() and not line.startswith(('The user', 'We ', 'Since ', 'Critical')):
                words = line.split()
                if 5 <= len(words) <= 15:
                    self.logger.warning(f"Use candidate line as fallback title: {line}")
                    return line
        
        return None
    
    def is_valid_english_title(self, text: str) -> bool:
        """Check if text is a valid English title"""
        if not text or len(text.strip()) < 8:
            return False
            
        text = text.strip()
        words = text.split()
        
        # Basic checks
        return (4 <= len(words) <= 20 and 
                all(ord(char) < 256 for char in text) and
                text[0].isupper())

if __name__ == "__main__":
    # Test LLM client
    client = LLMClient()
    
    # Test simple request
    test_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Please generate a news headline: Tech company releases new product"}
    ]
    
    response = client.chat_completion(test_messages)
    print("Test response:", response)
    print("Usage statistics:", client.get_usage_stats()) 