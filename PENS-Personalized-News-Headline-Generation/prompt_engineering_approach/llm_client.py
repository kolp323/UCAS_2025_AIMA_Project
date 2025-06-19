#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LLM API客户端模块 - 封装DeepSeek API调用
支持推理模型和普通聊天模型的自适应处理
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
    """LLM API客户端"""
    
    def __init__(self, model_name: str = None):
        """初始化LLM客户端"""
        self.api_config = API_CONFIG.copy()
        
        # 设置日志（先初始化）
        self._setup_logger()
        
        # 设置模型
        if model_name and model_name in get_available_models():
            self.api_config['model'] = model_name
        elif model_name:
            self.logger.warning(f"模型 {model_name} 不可用，使用默认模型 {self.api_config['model']}")
        
        self.client = OpenAI(
            api_key=self.api_config['api_key'],
            base_url=self.api_config['base_url']
        )
        
        # 使用状态
        self.total_tokens_used = 0
        self.total_requests = 0
        self.failed_requests = 0
        
        # 判断模型类型
        self.is_reasoning_model = is_reasoning_model(self.api_config['model'])
        self.logger.info(f"初始化LLM客户端: {self.api_config['model']} ({'推理模型' if self.is_reasoning_model else '聊天模型'})")
    
    def switch_model(self, model_name: str) -> bool:
        """切换模型"""
        if model_name in get_available_models():
            old_model = self.api_config['model']
            self.api_config['model'] = model_name
            self.is_reasoning_model = is_reasoning_model(model_name)
            self.logger.info(f"模型已切换: {old_model} -> {model_name} ({'推理模型' if self.is_reasoning_model else '聊天模型'})")
            return True
        else:
            self.logger.error(f"模型 {model_name} 不在可用列表中: {get_available_models()}")
            return False
    
    def get_current_model(self) -> str:
        """获取当前使用的模型"""
        return self.api_config['model']
    
    def get_model_info(self) -> dict:
        """获取当前模型的详细信息"""
        return get_model_config(self.api_config['model'])

    def _setup_logger(self):
        """设置日志器"""
        self.logger = logging.getLogger('llm_client')
        # 避免重复添加处理器
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
            self.logger.propagate = False  # 防止向根日志器传播，避免重复输出

    def estimate_tokens(self, text: str) -> int:
        """粗略估算token数量"""
        return int(len(text) * 1.5)

    def check_token_limit(self, tokens: int) -> bool:
        """检查是否超过token限制"""
        if tokens > self.api_config['max_tokens_per_request']:
            self.logger.warning(f"预估token数 {tokens} 超过单次请求限制 {self.api_config['max_tokens_per_request']}")
            return False
        
        if self.total_tokens_used + tokens > self.api_config['max_tokens']:
            self.logger.warning(f"预估总token数将超过限制 {self.api_config['max_tokens']}")
            return False
        
        return True

    def chat_completion(self, messages, **kwargs):
        """发送聊天完成请求 - 自适应处理不同模型类型"""
        # 根据模型类型自动调整max_tokens
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
                self.logger.info(f"发送API请求 (尝试 {attempt + 1}/{self.api_config['max_retries']})")
                
                response = self.client.chat.completions.create(**params)
                
                self.total_requests += 1
                
                # 自适应提取内容
                extracted_content = self._extract_content_adaptive(response)
                
                if extracted_content:
                    # 记录token使用情况
                    if hasattr(response, 'usage') and response.usage:
                        tokens_used = response.usage.total_tokens
                        self.total_tokens_used += tokens_used
                        self.logger.info(f"本次请求使用 {tokens_used} tokens，总计 {self.total_tokens_used} tokens")
                    
                    return extracted_content

                self.logger.warning("API响应为空或无法提取内容")
                return None

            except Exception as e:
                self.logger.error(f"API请求失败 (尝试 {attempt + 1}): {e}")
                if attempt == self.api_config['max_retries'] - 1:
                    return None
                time.sleep(2 ** attempt)  # 指数退避

        return None

    def _extract_content_adaptive(self, response):
        """自适应提取内容 - 根据模型类型和任务类型选择最佳策略"""
        message = response.choices[0].message
        
        # 对于推理模型，需要区分任务类型
        if self.is_reasoning_model:
            # 首先检查content字段是否有有效内容
            if message.content and message.content.strip():
                content = message.content.strip()
                
                # 对于评估任务（包含数字和逗号的评分），优先使用content字段
                if self._is_valid_evaluation_result(content):
                    cleaned_content = self.clean_generated_title(content)
                    self.logger.info(f"从content字段获取评估结果(推理模型): {cleaned_content[:100]}...")
                    return cleaned_content
                
                # 对于标题生成任务，检查是否是有效标题
                elif self._is_valid_title_content(content):
                    cleaned_title = self.clean_generated_title(content)
                    self.logger.info(f"从content字段获取标题(推理模型): {cleaned_title}")
                    return cleaned_title
            
            # 如果content字段无效或为空，才考虑reasoning字段（仅用于标题生成）
            if hasattr(message, 'model_extra') and message.model_extra and 'reasoning' in message.model_extra:
                reasoning = message.model_extra['reasoning']
                if reasoning:
                    # 尝试从推理过程中提取标题（仅用于标题生成任务）
                    extracted_title = self.extract_title_from_reasoning(reasoning)
                    if extracted_title:
                        self.logger.info(f"从reasoning字段提取标题(推理模型): {extracted_title}")
                        return extracted_title
                    else:
                        self.logger.warning("无法从reasoning字段提取有效标题")
                else:
                    self.logger.warning("reasoning字段为空")
        else:
            # 对于非推理模型，优先使用content字段
            if message.content and message.content.strip():
                content = message.content.strip()
                cleaned_title = self.clean_generated_title(content)
                self.logger.info(f"从content字段获取内容(聊天模型): {cleaned_title}")
                return cleaned_title
        
        # 如果所有策略都失败
        self.logger.warning(f"无法提取内容 - content: '{message.content}', is_reasoning: {self.is_reasoning_model}, has_model_extra: {hasattr(message, 'model_extra')}")
        return None

    def _is_valid_title_content(self, content: str) -> bool:
        """检查content是否是有效的标题而非推理过程"""
        if not content:
            return False
        
        # 检查是否包含推理关键词
        reasoning_indicators = [
            'the user', 'personalized', 'headline', 'create', 'generate',
            'we need', 'based on', 'i think', 'let me', 'alternative options'
        ]
        
        if any(indicator in content.lower() for indicator in reasoning_indicators):
            return False
        
        # 检查长度合理性
        words = content.split()
        return 5 <= len(words) <= 25
    
    def _is_valid_evaluation_result(self, content: str) -> bool:
        """检查content是否是有效的评估结果（数字、评分等）"""
        if not content:
            return False
        
        content = content.strip()
        
        # 检查是否包含数字和逗号的模式（评估结果）
        import re
        # 查找数字,数字,数字的模式
        number_pattern = r'^\s*(\d+(?:\.\d+)?(?:\s*,\s*\d+(?:\.\d+)?)*)\s*$'
        if re.match(number_pattern, content):
            return True
        
        # 检查是否包含评分相关的简单数字输出
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if re.match(r'^\s*\d+(?:\.\d+)?(?:\s*,\s*\d+(?:\.\d+)?)+\s*$', line):
                return True
        
        return False

    def generate_personalized_title(self, sample: Dict[str, Any], system_prompt: str, user_prompt: str) -> Optional[str]:
        """为单个样本生成个性化标题"""
        
        # 检查user_prompt是否已经是完整的提示词
        if '{' not in user_prompt:
            # 已经是完整的提示词，直接使用
            formatted_prompt = user_prompt
        else:
            # 是模板，需要格式化
            # 构建用户历史字符串
            history_str = "\n".join([f"- {title}" for title in sample['user_history'][:10]])  # 限制历史记录数量
            
            # 构建兴趣标签字符串
            interests = sample['user_interests']
            interest_str = f"主要兴趣: {interests['primary_interest']}, 相关类别: {', '.join(interests['categories'])}"
            
            # 格式化用户提示词
            try:
                formatted_prompt = user_prompt.format(
                    user_history=history_str,
                    user_interests=interest_str,
                    original_title=sample['original_title'],
                    news_content=sample['news_body'][:400],  # 限制新闻内容长度
                    news_category=sample['category']
                )
            except KeyError as e:
                self.logger.error(f"提示词格式化失败: {e}")
                return None
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": formatted_prompt}
        ]
        
        # 根据模型类型自动调整max_tokens
        optimal_tokens = get_optimal_max_tokens(self.api_config['model'])
        return self.chat_completion(messages, max_tokens=optimal_tokens)
    
    def generate_batch_titles(self, samples: List[Dict[str, Any]], system_prompt: str, batch_prompt_template: str) -> List[Optional[str]]:
        """批量生成个性化标题"""
        
        batch_content = ""
        for i, sample in enumerate(samples, 1):
            # 构建用户历史（简化版）
            history_summary = f"最近阅读: {', '.join(sample['user_history'][:3])}"
            interest_summary = sample['user_interests']['primary_interest']
            
            batch_content += f"""
新闻{i}:
- 用户兴趣: {interest_summary}
- 用户历史: {history_summary}
- 原标题: {sample['original_title']}
- 新闻类别: {sample['category']}
- 内容摘要: {sample['news_body'][:200]}...

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
        """解析批量响应"""
        results = [None] * expected_count
        
        try:
            # 策略1: 解析 "News X: [标题]" 格式
            import re
            news_pattern = r'News\s+(\d+):\s*([^\n]+?)(?=\s*News\s+\d+:|$)'
            matches = re.findall(news_pattern, response, re.IGNORECASE | re.DOTALL)
            
            for match in matches:
                try:
                    idx = int(match[0]) - 1
                    if 0 <= idx < expected_count:
                        title = match[1].strip()
                        # 清理标题
                        title = re.sub(r'^["\']|["\']$', '', title)  # 移除首尾引号
                        title = re.sub(r'\s+', ' ', title)  # 规范化空格
                        results[idx] = title
                except (ValueError, IndexError):
                    continue
            
            # 策略2: 如果策略1失败，尝试按行解析
            if all(r is None for r in results):
                lines = response.strip().split('\n')
                for line in lines:
                    line = line.strip()
                    # 匹配 "新闻X:" 或 "News X:" 或数字开头的格式
                    if re.match(r'^(?:新闻|News)\s*\d+\s*[:\-]', line, re.IGNORECASE):
                        parts = re.split(r'[:\-]', line, 1)
                        if len(parts) == 2:
                            # 提取数字
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
            
            # 记录解析结果
            parsed_count = sum(1 for r in results if r is not None)
            self.logger.info(f"批量解析: 期望{expected_count}个，成功解析{parsed_count}个")
            
            for i, result in enumerate(results):
                if result:
                    self.logger.info(f"  批量标题{i+1}: {result}")
                    
        except Exception as e:
            self.logger.error(f"解析批量响应失败: {str(e)}")
        
        return results
    
    def parse_multiple_titles(self, response: str) -> List[str]:
        """解析包含多个标题选项的响应"""
        titles = []
        
        try:
            lines = response.strip().split('\n')
            for line in lines:
                line = line.strip()
                # 匹配格式如 "1. [标题]" 或 "1. 标题"
                if line and (line[0].isdigit() or line.startswith('-')):
                    # 提取标题部分
                    if '.' in line:
                        title = line.split('.', 1)[1].strip()
                    elif '-' in line:
                        title = line.split('-', 1)[1].strip()
                    else:
                        title = line
                    
                    # 清理格式
                    title = title.replace('[', '').replace(']', '').strip()
                    if title:
                        titles.append(title)
        except Exception as e:
            self.logger.error(f"解析多标题响应失败: {str(e)}")
        
        return titles[:3]  # 最多返回3个标题
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """获取使用统计"""
        return {
            'total_requests': self.total_requests,
            'failed_requests': self.failed_requests,
            'success_rate': (self.total_requests - self.failed_requests) / max(self.total_requests, 1),
            'total_tokens_used': self.total_tokens_used,
            'remaining_tokens': max(0, self.api_config['max_tokens'] - self.total_tokens_used)
        }
    
    def reset_stats(self):
        """重置统计信息"""
        self.total_tokens_used = 0
        self.total_requests = 0
        self.failed_requests = 0
    
    def clean_generated_title(self, title: str) -> str:
        """清理生成的标题"""
        if not title:
            return None
            
        # 移除常见的前缀和后缀
        title = title.strip()
        
        # 移除引号
        if title.startswith('"') and title.endswith('"'):
            title = title[1:-1]
        elif title.startswith("'") and title.endswith("'"):
            title = title[1:-1]
        
        # 移除可能的序号或标记
        title = re.sub(r'^[\d\.\-\*\+\s]+', '', title)
        
        # 移除特殊标记
        title = re.sub(r'\*[^*]*\*', '', title)  # 移除*标记*
        title = re.sub(r'^\w+\s*mentally[^:]*:', '', title)  # 移除"Drafts mentally:"类型
        
        # 移除HTML标签
        title = re.sub(r'<[^>]+>', '', title)
        
        # 规范化空格
        title = re.sub(r'\s+', ' ', title).strip()
        
        # 确保首字母大写
        if title:
            title = title[0].upper() + title[1:] if len(title) > 1 else title.upper()
        
        return title
    
    def extract_title_from_reasoning(self, reasoning: str) -> Optional[str]:
        """从reasoning字段中提取实际的标题"""
        if not reasoning:
            return None
            
        lines = [line.strip() for line in reasoning.split('\n') if line.strip()]
        
        # 策略1: 查找引号中的内容（最可靠）
        quotes = re.findall(r'"([^"]*)"', reasoning)
        for quote in quotes:
            words = quote.split()
            # 检查是否是合理的标题长度
            if 6 <= len(words) <= 25:
                # 确保不是原始标题的重复或推理文本
                if not any(phrase in quote.lower() for phrase in [
                    'the original', 'high-stakes legal', 'personalized', 'headline',
                    'user interested', 'create', 'generate', 'we need', 'based on'
                ]):
                    self.logger.info(f"从引号中提取标题: {quote}")
                    return quote
        
        # 策略2: 查找最后几行中可能的标题
        for line in reversed(lines[-15:]):  # 检查最后15行
            # 跳过明显的推理文本
            if any(phrase in line.lower() for phrase in [
                'the user', 'we need', 'based on', 'this', 'therefore', 
                'critical instructions', 'since', 'in this context', 'i think',
                'let me', 'so the', 'given that', 'drafts mentally', 'too long',
                'trim', 'here', 'now', 'final', 'output', 'personalized',
                'headline', 'generate', 'create'
            ]) or line.startswith(('-', '*', '•', '1.', '2.', '3.')):
                continue
                
            words = line.split()
            # 检查是否是合理的标题长度和格式
            if 6 <= len(words) <= 20:
                # 确保包含实际内容而不是元描述
                # 检查是否包含新闻相关词汇
                news_indicators = [
                    'report', 'announce', 'reveal', 'confirm', 'break', 'launch',
                    'win', 'lose', 'face', 'plan', 'set', 'expected', 'new',
                    'first', 'major', 'top', 'best', 'worst', 'latest', 'says',
                    'shows', 'finds', 'study', 'data', 'market', 'company',
                    'government', 'court', 'legal', 'law', 'rule', 'policy'
                ]
                
                if any(indicator in line.lower() for indicator in news_indicators):
                    self.logger.info(f"从推理末尾提取标题: {line}")
                    return line
        
        # 策略3: 查找符合新闻标题格式的句子
        for line in lines:
            # 跳过明显的推理开头
            if line.startswith(('The', 'We', 'Since', 'Critical', '-', '*', 'I ', 'Let ')) or \
               any(phrase in line.lower() for phrase in [
                   'personalized', 'headline', 'user interested', 'create', 'generate',
                   'instructions', 'format', 'output', 'drafts mentally'
               ]):
                continue
                
            words = line.split()
            if 8 <= len(words) <= 18:
                # 检查是否有动词和名词的组合（新闻标题特征）
                has_verb = any(word.lower() in ['faces', 'wins', 'loses', 'announces', 
                                              'reveals', 'confirms', 'breaks', 'launches',
                                              'reports', 'shows', 'finds', 'says'] for word in words)
                has_noun = any(word.lower() in ['company', 'government', 'court', 'study',
                                              'market', 'report', 'news', 'data'] for word in words)
                
                if has_verb or has_noun or ':' in line:
                    self.logger.info(f"从推理中提取标题候选: {line}")
                    return line
        
        # 策略4: 如果都失败了，查找包含大写字母开头的行（可能是标题）
        for line in lines:
            if line and line[0].isupper() and not line.startswith(('The user', 'We ', 'Since ', 'Critical')):
                words = line.split()
                if 5 <= len(words) <= 15:
                    self.logger.warning(f"使用候选行作为后备标题: {line}")
                    return line
        
        return None
    
    def is_valid_english_title(self, text: str) -> bool:
        """检查文本是否是有效的英文标题"""
        if not text or len(text.strip()) < 8:
            return False
            
        text = text.strip()
        words = text.split()
        
        # 基本检查
        return (4 <= len(words) <= 20 and 
                all(ord(char) < 256 for char in text) and
                text[0].isupper())

if __name__ == "__main__":
    # 测试LLM客户端
    client = LLMClient()
    
    # 测试简单请求
    test_messages = [
        {"role": "system", "content": "你是一个有用的助手。"},
        {"role": "user", "content": "请生成一个新闻标题: 科技公司发布新产品"}
    ]
    
    response = client.chat_completion(test_messages)
    print("测试响应:", response)
    print("使用统计:", client.get_usage_stats()) 