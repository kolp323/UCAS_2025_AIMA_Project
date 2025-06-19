"""
评估模块 - 包含大模型API评价和个性化效果评估
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re

# 修复ROUGE导入
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    print("警告: rouge_score包未安装，将使用简化的评估方法")
    ROUGE_AVAILABLE = False

from config import EVALUATION_CONFIG, DATA_PATHS, EVALUATION_MODEL
from llm_client import LLMClient

class Evaluator:
    """评估器"""
    
    def __init__(self, use_llm_evaluation: bool = True):
        self.logger = self._setup_logger()
        self.rouge_scorer = None
        self.use_llm_evaluation = use_llm_evaluation
        
        # 初始化LLM客户端用于评估
        if use_llm_evaluation:
            # 直接使用指定的评估模型初始化LLM客户端
            self.llm_client = LLMClient(EVALUATION_MODEL)
            self.logger.info(f"使用 {EVALUATION_MODEL} 进行LLM评估")
        
        # 初始化ROUGE评估器
        if ROUGE_AVAILABLE:
            try:
                self.rouge_scorer = rouge_scorer.RougeScorer(
                    ['rouge1', 'rouge2', 'rougeL'], 
                    use_stemmer=True
                )
                self.logger.info("ROUGE评分器初始化成功")
            except Exception as e:
                self.logger.warning(f"ROUGE评分器初始化失败: {e}")
                self.rouge_scorer = None
        
        # 评估结果存储
        self.evaluation_results = {}
        
    def _setup_logger(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def calculate_rouge_scores(self, generated_titles: List[str], reference_titles: List[str]) -> Dict[str, float]:
        """计算ROUGE分数"""
        
        if not self.rouge_scorer:
            self.logger.warning("ROUGE评估器不可用，使用简化评估")
            return self._calculate_simple_scores(generated_titles, reference_titles)
        
        try:
            # 预处理文本
            processed_generated = [self._preprocess_text_for_rouge(title) for title in generated_titles]
            processed_reference = [self._preprocess_text_for_rouge(title) for title in reference_titles]
            
            # 过滤空文本
            valid_pairs = [(g, r) for g, r in zip(processed_generated, processed_reference) 
                          if g and r and g != "empty text" and r != "empty text"]
            
            if not valid_pairs:
                self.logger.error("没有有效的文本对进行ROUGE评估")
                return {}
            
            # 计算ROUGE分数
            rouge1_scores = []
            rouge2_scores = []
            rougeL_scores = []
            
            for generated, reference in valid_pairs:
                scores = self.rouge_scorer.score(reference, generated)
                rouge1_scores.append(scores['rouge1'])
                rouge2_scores.append(scores['rouge2'])
                rougeL_scores.append(scores['rougeL'])
            
            # 计算平均分数
            rouge_results = {
                'rouge1_f': np.mean([score.fmeasure for score in rouge1_scores]),
                'rouge1_p': np.mean([score.precision for score in rouge1_scores]),
                'rouge1_r': np.mean([score.recall for score in rouge1_scores]),
                'rouge2_f': np.mean([score.fmeasure for score in rouge2_scores]),
                'rouge2_p': np.mean([score.precision for score in rouge2_scores]),
                'rouge2_r': np.mean([score.recall for score in rouge2_scores]),
                'rougeL_f': np.mean([score.fmeasure for score in rougeL_scores]),
                'rougeL_p': np.mean([score.precision for score in rougeL_scores]),
                'rougeL_r': np.mean([score.recall for score in rougeL_scores]),
            }
            
            self.logger.info(f"ROUGE评估完成，有效样本数: {len(valid_pairs)}")
            return rouge_results
            
        except Exception as e:
            self.logger.error(f"ROUGE评估失败: {str(e)}")
            return self._calculate_simple_scores(generated_titles, reference_titles)
    
    def _preprocess_text_for_rouge(self, text: str) -> str:
        """为ROUGE评估预处理文本"""
        if not text or not isinstance(text, str):
            return "empty text"
        
        # 移除多余的空格和换行
        text = ' '.join(text.split())
        
        # 确保文本不为空
        if not text.strip():
            return "empty text"
            
        return text.strip()
    
    def _calculate_simple_scores(self, generated_titles: List[str], reference_titles: List[str]) -> Dict[str, float]:
        """简化的评估方法（当ROUGE不可用时）"""
        
        scores = {}
        total_pairs = len(generated_titles)
        
        if total_pairs == 0:
            return scores
        
        # 计算字符级别的重叠
        char_overlaps = []
        word_overlaps = []
        length_ratios = []
        
        for gen, ref in zip(generated_titles, reference_titles):
            if not gen or not ref:
                continue
                
            gen_chars = set(gen.lower())
            ref_chars = set(ref.lower())
            char_overlap = len(gen_chars.intersection(ref_chars)) / max(len(gen_chars.union(ref_chars)), 1)
            char_overlaps.append(char_overlap)
            
            gen_words = set(gen.lower().split())
            ref_words = set(ref.lower().split())
            word_overlap = len(gen_words.intersection(ref_words)) / max(len(gen_words.union(ref_words)), 1)
            word_overlaps.append(word_overlap)
            
            length_ratio = min(len(gen), len(ref)) / max(len(gen), len(ref), 1)
            length_ratios.append(length_ratio)
        
        scores['simple_char_overlap'] = np.mean(char_overlaps) if char_overlaps else 0.0
        scores['simple_word_overlap'] = np.mean(word_overlaps) if word_overlaps else 0.0
        scores['length_similarity'] = np.mean(length_ratios) if length_ratios else 0.0
        
        return scores
    
    def evaluate_personalization(self, generated_titles: List[str], 
                                        user_interests: List[Dict], 
                                        news_categories: List[str],
                                        user_histories: List[List[str]]) -> Dict[str, float]:
        """个性化效果评估（基于规则）"""
        
        personalization_scores = []
        category_relevance_scores = []
        interest_alignment_scores = []
        history_relevance_scores = []
        
        for i, (title, interests, category, history) in enumerate(
            zip(generated_titles, user_interests, news_categories, user_histories)):
            
            if not title or not isinstance(interests, dict):
                continue
            
            title_lower = title.lower()
            
            # 1. 兴趣匹配度评估（改进逻辑）
            primary_interest = interests.get('primary_interest', '').lower()
            interest_categories = [cat.lower() for cat in interests.get('categories', [])]
            
            interest_score = 0.0
            
            # 类别直接匹配（权重60%）
            if category and category.lower() in interest_categories:
                interest_score += 0.6
            
            # 主要兴趣匹配（权重25%）
            if primary_interest and primary_interest in title_lower:
                interest_score += 0.25
            
            # 关键词部分匹配（权重15%）
            if primary_interest:
                primary_words = primary_interest.split()
                matched_words = sum(1 for word in primary_words if len(word) > 3 and word in title_lower)
                if primary_words:
                    interest_score += 0.15 * (matched_words / len(primary_words))
            
            personalization_scores.append(min(interest_score, 1.0))
            
            # 2. 类别相关性评估（改进）
            category_score = 0.0
            if category and interest_categories:
                if category.lower() in interest_categories:
                    category_score = 1.0  # 直接匹配
                else:
                    # 语义相关性检查
                    category_words = set(category.lower().split())
                    for cat in interest_categories:
                        cat_words = set(cat.split())
                        if category_words.intersection(cat_words):
                            category_score = max(category_score, 0.5)
            
            category_relevance_scores.append(category_score)
            
            # 3. 兴趣一致性评估（更合理）
            alignment_score = 0.0
            if category and interest_categories:
                if category.lower() in interest_categories:
                    alignment_score = 1.0
                elif 'news' in interest_categories and category in ['news', 'politics', 'business']:
                    alignment_score = 0.8  # 新闻相关类别
                elif 'finance' in interest_categories and category in ['business', 'finance', 'economy']:
                    alignment_score = 0.8  # 金融相关类别
                else:
                    alignment_score = 0.4  # 基础分
            
            interest_alignment_scores.append(alignment_score)
            
            # 4. 历史相关性评估（改进）
            history_score = 0.0
            if history and len(history) > 0:
                title_words = set(title_lower.split())
                history_words = set()
                for hist_title in history[:15]:  # 看更多历史
                    if hist_title:
                        history_words.update(hist_title.lower().split())
                
                if history_words:
                    common_words = title_words.intersection(history_words)
                    # 过滤停用词和短词
                    stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an'}
                    meaningful_words = [w for w in common_words if len(w) > 3 and w not in stop_words]
                    if title_words:
                        history_score = min(len(meaningful_words) / len(title_words), 1.0)
            
            history_relevance_scores.append(history_score)
        
        results = {
            'rule_based_personalization': np.mean(personalization_scores) if personalization_scores else 0.0,
            'rule_based_category_relevance': np.mean(category_relevance_scores) if category_relevance_scores else 0.0,
            'rule_based_interest_alignment': np.mean(interest_alignment_scores) if interest_alignment_scores else 0.0,
            'rule_based_history_relevance': np.mean(history_relevance_scores) if history_relevance_scores else 0.0
        }
        
        # 计算综合规则评分
        results['rule_based_overall'] = (
            results['rule_based_personalization'] * 0.4 +
            results['rule_based_category_relevance'] * 0.25 +
            results['rule_based_interest_alignment'] * 0.25 +
            results['rule_based_history_relevance'] * 0.1
        )
        
        return results
    
    def llm_evaluate_personalization(self, generated_titles: List[str], 
                                   user_interests: List[Dict], 
                                   news_categories: List[str],
                                   user_histories: List[List[str]], 
                                   batch_size: int = 5) -> Dict[str, float]:
        """使用LLM评估个性化效果"""
        
        if not self.use_llm_evaluation or not self.llm_client:
            self.logger.warning("LLM个性化评估不可用")
            return {}
        
        system_prompt = """你是一位专业的个性化新闻推荐评估专家。你的任务是对新闻标题的个性化程度进行评分。

评估维度 (每个维度0-10分):
1. 兴趣匹配度: 标题内容与用户兴趣的匹配程度
2. 类别相关性: 标题与用户偏好类别的相关性  
3. 历史一致性: 标题与用户历史阅读习惯的一致性
4. 个性化创新: 标题在个性化方面的创新和吸引力
5. 综合个性化程度: 整体个性化效果

严格按照要求输出：
- 只输出数字和逗号，不要任何文字说明
- 每行一个标题的评分，格式：分数1,分数2,分数3,分数4,分数5
- 分数范围0-10，可以是小数（如8.5）
- 按照标题顺序依次输出
- 不要输出任何解释、分析或其他文字

示例输出：
8.5,7.0,9.0,6.5,8.0
6.0,8.5,5.5,7.0,6.5"""

        all_scores = {
            'interest_match': [],
            'category_relevance': [],
            'history_consistency': [],
            'personalization_innovation': [],
            'overall_personalization': []
        }
        
        # 分批处理
        for i in range(0, len(generated_titles), batch_size):
            batch_titles = generated_titles[i:i+batch_size]
            batch_interests = user_interests[i:i+batch_size]
            batch_categories = news_categories[i:i+batch_size]
            # 确保user_histories不为空且长度正确
            if user_histories and len(user_histories) > i:
                batch_histories = user_histories[i:i+batch_size]
            else:
                batch_histories = [[] for _ in batch_titles]
            
            # 构建批次评估提示词
            user_prompt = self._build_personalization_evaluation_prompt(
                batch_titles, batch_interests, batch_categories, batch_histories)
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            try:
                response = self.llm_client.chat_completion(messages, max_tokens=1500)
                if response:
                    batch_scores = self._parse_personalization_evaluation_response(response)
                    for j, scores in enumerate(batch_scores):
                        if len(scores) >= 5:
                            all_scores['interest_match'].append(scores[0])
                            all_scores['category_relevance'].append(scores[1])
                            all_scores['history_consistency'].append(scores[2])
                            all_scores['personalization_innovation'].append(scores[3])
                            all_scores['overall_personalization'].append(scores[4])
                        else:
                            # 默认中等分数
                            for key in all_scores:
                                all_scores[key].append(5.0)
                else:
                    # 添加默认分数
                    for _ in range(len(batch_titles)):
                        for key in all_scores:
                            all_scores[key].append(5.0)
                            
            except Exception as e:
                self.logger.error(f"LLM个性化评估失败: {e}")
                for _ in range(len(batch_titles)):
                    for key in all_scores:
                        all_scores[key].append(5.0)
        
        # 计算平均分数（转换为0-1范围）
        results = {}
        for key, scores in all_scores.items():
            if scores:
                results[f'llm_{key}'] = np.mean(scores) / 10.0  # 转换为0-1范围
                results[f'llm_{key}_std'] = np.std(scores) / 10.0
        
        return results
    
    def _build_personalization_evaluation_prompt(self, titles: List[str], 
                                                interests: List[Dict], 
                                                categories: List[str],
                                                histories: List[List[str]]) -> str:
        """构建个性化评估提示词"""
        
        prompt = "作为个性化新闻推荐评估专家，请对以下每个新闻标题的个性化程度进行评分。\n\n"
        
        prompt += "📋 评估维度（每个维度0-10分）：\n"
        prompt += "维度1: 兴趣匹配度 - 标题与用户兴趣的匹配程度\n"
        prompt += "维度2: 类别相关性 - 标题与用户偏好类别的相关性\n"  
        prompt += "维度3: 历史一致性 - 标题与用户历史阅读的一致性\n"
        prompt += "维度4: 个性化创新 - 标题的个性化吸引力\n"
        prompt += "维度5: 综合个性化 - 整体个性化效果\n\n"
        
        prompt += "📰 待评估的新闻标题及用户信息：\n"
        
        for i, (title, interest, category, history) in enumerate(zip(titles, interests, categories, histories)):
            prompt += f"\n=== 标题 {i+1} ===\n"
            prompt += f"新闻标题: \"{title}\"\n"
            prompt += f"新闻类别: {category}\n"
            
            primary = interest.get('primary_interest', '未知')
            categories_list = interest.get('categories', [])
            prompt += f"用户兴趣: 主要兴趣={primary}, 偏好类别={categories_list}\n"
            
            if history and len(history) > 0:
                recent = history[:3]  # 显示最近3条
                prompt += f"阅读历史: {'; '.join(recent)}\n"
            else:
                prompt += "阅读历史: 无记录\n"
        
        prompt += "\n⚠️ 输出要求：\n"
        prompt += "- 必须按照标题顺序（1到N）依次评分\n"
        prompt += "- 每行输出一个标题的评分，格式：维度1分数,维度2分数,维度3分数,维度4分数,维度5分数\n"
        prompt += "- 只输出数字和逗号，不要任何文字说明\n"
        prompt += "- 分数范围：0-10（可以是小数，如8.5）\n\n"
        
        prompt += "示例输出：\n"
        prompt += "8,7,9,6,8\n"
        prompt += "6,8,5,7,6\n\n"
        
        prompt += "请开始评分（按顺序对每个标题输出一行评分）：\n"
        
        return prompt
    
    def _parse_personalization_evaluation_response(self, response: str) -> List[List[float]]:
        """解析LLM个性化评估响应，支持多种格式包括缺少首个分数的情况"""
        
        import re
        
        self.logger.debug(f"解析个性化评估响应: {response[:200]}...")
        
        try:
            cleaned_text = response.strip()
            score_segments = []
            
            # 方法1：按行分割
            lines = [line.strip() for line in cleaned_text.split('\n') if line.strip()]
            
            for line in lines:
                if not line.strip():
                    continue
                
                # 跳过明显的说明文字
                if any(keyword in line.lower() for keyword in ['标题', '评估', '维度', '分数', '输出', '示例', '作为', '请']):
                    continue
                
                # 查找所有数字（包括小数）
                numbers = re.findall(r'\d+(?:\.\d+)?', line)
                
                if len(numbers) >= 3:  # 至少3个数字才认为是有效的分数行
                    try:
                        scores = [float(num) for num in numbers]
                        # 限制在合理范围内
                        scores = [max(0.0, min(10.0, score)) for score in scores]
                        score_segments.append(scores)
                        self.logger.debug(f"从行 '{line}' 解析出个性化分数: {scores}")
                    except ValueError:
                        continue
            
            # 方法2：处理连续的分数段（如果按行分割失败）
            if not score_segments:
                # 按空格分割可能的分数段
                potential_segments = re.split(r'\s+', cleaned_text)
                
                for segment in potential_segments:
                    segment = segment.strip()
                    if not segment:
                        continue
                    
                    # 处理以逗号开头的分数段（缺少第一个分数）
                    if segment.startswith(','):
                        # 添加默认分数5.0
                        segment = '5.0' + segment
                    
                    # 查找分数
                    numbers = re.findall(r'\d+(?:\.\d+)?', segment)
                    
                    if len(numbers) >= 3:  # 至少3个数字
                        try:
                            scores = [float(num) for num in numbers]
                            scores = [max(0.0, min(10.0, score)) for score in scores]
                            score_segments.append(scores)
                            self.logger.debug(f"从段 '{segment}' 解析出个性化分数: {scores}")
                        except ValueError:
                            continue
            
            # 方法3：直接解析所有数字
            if not score_segments:
                all_numbers = re.findall(r'\d+(?:\.\d+)?', cleaned_text)
                
                if len(all_numbers) >= 5:  # 至少5个数字
                    try:
                        # 将数字按5个一组分组
                        for i in range(0, len(all_numbers), 5):
                            if i + 4 < len(all_numbers):  # 确保有完整的5个数字
                                group = all_numbers[i:i+5]
                                scores = [float(num) for num in group]
                                scores = [max(0.0, min(10.0, score)) for score in scores]
                                score_segments.append(scores)
                    except ValueError:
                        pass
            
            if not score_segments:
                self.logger.warning("没有解析出任何个性化分数，返回默认值")
                return [[5.0, 5.0, 5.0, 5.0, 5.0]]
            
            # 确保每个分数段都有5个值
            normalized_scores = []
            for scores in score_segments:
                if len(scores) < 5:
                    # 不足5个则用平均值补充
                    avg_score = np.mean(scores) if scores else 5.0
                    scores.extend([avg_score] * (5 - len(scores)))
                elif len(scores) > 5:
                    # 超过5个则截取前5个
                    scores = scores[:5]
                
                normalized_scores.append(scores)
            
            self.logger.info(f"个性化评估解析完成，共{len(normalized_scores)}组分数")
            return normalized_scores
            
        except Exception as e:
            self.logger.error(f"解析个性化评估响应失败: {str(e)}")
            return [[5.0, 5.0, 5.0, 5.0, 5.0]]
    
    def llm_evaluate_quality(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """使用LLM评估标题质量（修复版本）"""
        
        self.logger.info("开始LLM质量评估...")
        
        generated_titles = results.get('generated_titles', [])
        reference_titles = results.get('reference_titles', [])
        # 修复：统一使用news_contents字段名
        news_bodies = results.get('news_contents', []) or results.get('news_bodies', [])
        
        # 检查数据完整性，提供更详细的警告信息
        if not generated_titles:
            self.logger.warning("缺少生成的标题数据")
            return {
                'llm_quality_score': 0.0,
                'llm_quality_scores': []
            }
        
        if not reference_titles:
            self.logger.warning("缺少参考标题数据")
            return {
                'llm_quality_score': 0.0,
                'llm_quality_scores': []
            }
            
        if not news_bodies:
            self.logger.warning("缺少新闻内容数据")
            return {
                'llm_quality_score': 0.0,
                'llm_quality_scores': []
            }
        
        # 检查数据长度一致性
        if len(generated_titles) != len(reference_titles) or len(generated_titles) != len(news_bodies):
            self.logger.warning(f"数据长度不一致: 生成标题{len(generated_titles)}, 参考标题{len(reference_titles)}, 新闻内容{len(news_bodies)}")
            # 截取到最短长度
            min_length = min(len(generated_titles), len(reference_titles), len(news_bodies))
            generated_titles = generated_titles[:min_length]
            reference_titles = reference_titles[:min_length]
            news_bodies = news_bodies[:min_length]
            self.logger.info(f"已调整数据长度为: {min_length}")
        
        self.logger.info(f"LLM质量评估准备就绪，共{len(generated_titles)}个样本")
        
        # 批量处理，每批5个
        batch_size = 5
        all_scores = []
        
        for i in range(0, len(generated_titles), batch_size):
            batch_generated = generated_titles[i:i+batch_size]
            batch_reference = reference_titles[i:i+batch_size]
            batch_bodies = news_bodies[i:i+batch_size]
            
            self.logger.info(f"正在评估批次 {i//batch_size + 1}，包含 {len(batch_generated)} 个标题")
            
            try:
                batch_scores = self._llm_evaluate_batch_quality(
                    batch_generated, batch_reference, batch_bodies
                )
                
                if batch_scores:
                    # batch_scores是二维列表，每个子列表包含5个维度分数
                    all_scores.extend(batch_scores)
                    self.logger.info(f"批次评估成功，获得 {len(batch_scores)} 个分数")
                else:
                    # 如果批次评估失败，添加默认分数
                    default_scores = [[0.5, 0.5, 0.5, 0.5, 0.5] for _ in range(len(batch_generated))]
                    all_scores.extend(default_scores)
                    self.logger.warning(f"批次评估失败，使用默认分数")
                    
            except Exception as e:
                self.logger.error(f"批次评估出错: {e}")
                # 添加默认分数
                default_scores = [[0.5, 0.5, 0.5, 0.5, 0.5] for _ in range(len(batch_generated))]
                all_scores.extend(default_scores)
        
        if not all_scores:
            self.logger.warning("没有获得任何LLM质量评分，返回默认值")
            return {
                'llm_quality_score': 0.0,
                'llm_quality_scores': []
            }
        
        # 计算每个标题的综合分数（五个维度的平均值）
        title_scores = []
        for score_list in all_scores:
            if isinstance(score_list, list) and len(score_list) >= 5:
                # 取前5个维度分数的平均值
                avg_score = sum(score_list[:5]) / 5.0
                # 确保在0-1范围内
                normalized_score = max(0.0, min(1.0, avg_score))
                title_scores.append(normalized_score)
            else:
                # 如果格式不对，使用默认分数
                title_scores.append(0.5)
        
        # 计算总体平均分
        overall_score = sum(title_scores) / len(title_scores) if title_scores else 0.0
        
        self.logger.info(f"LLM质量评估完成，平均分: {overall_score:.3f}")
        
        return {
            'llm_quality_score': overall_score,
            'llm_quality_scores': title_scores
        }
    
    def _llm_evaluate_batch_quality(self, generated_titles: List[str], 
                                     reference_titles: List[str],
                                     news_bodies: List[str]) -> List[List[float]]:
        """批次LLM质量评估"""
        
        system_prompt = """你是一位专业的新闻标题质量评估专家。请对生成的新闻标题进行质量评分。

评估维度 (每个维度0-10分):
1. 准确性: 标题是否准确反映新闻内容
2. 吸引力: 标题是否能吸引读者注意
3. 清晰度: 标题表达是否清晰易懂
4. 简洁性: 标题长度和表达是否简洁
5. 综合质量: 整体标题质量

严格按照要求输出：
- 只输出数字和逗号，不要任何文字说明
- 每行一个标题的评分，格式：分数1,分数2,分数3,分数4,分数5
- 分数范围0-10，可以是小数（如8.5）
- 按照标题顺序依次输出
- 不要输出任何解释、分析或其他文字

示例输出：
8.5,7.0,9.0,6.5,8.0
6.0,8.5,5.5,7.0,6.5"""
        
        user_prompt = "请对以下新闻标题进行质量评估：\n\n"
        
        for i, (gen, orig, content) in enumerate(zip(generated_titles, reference_titles, news_bodies)):
            user_prompt += f"=== 标题 {i+1} ===\n"
            user_prompt += f"原标题: {orig}\n"
            user_prompt += f"生成标题: {gen}\n"
            user_prompt += f"新闻内容: {content[:200]}...\n\n"
        
        user_prompt += "请按照标题顺序，每行输出一个标题的评分（5个维度分数，用逗号分隔）："
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = self.llm_client.chat_completion(messages, max_tokens=1000)
            if response:
                batch_scores = self._parse_llm_evaluation_response(response, len(generated_titles))
                if batch_scores:  # 确保解析成功
                    return batch_scores
                else:
                    # 解析失败，使用默认分数
                    return [[0.5, 0.5, 0.5, 0.5, 0.5]] * len(generated_titles)
            else:
                # API调用失败，使用默认分数
                return [[0.5, 0.5, 0.5, 0.5, 0.5]] * len(generated_titles)
            
        except Exception as e:
            self.logger.error(f"LLM质量评估失败: {e}")
            # 返回默认分数矩阵
            return [[0.5, 0.5, 0.5, 0.5, 0.5]] * len(generated_titles)
    
    def _parse_llm_evaluation_response(self, response_text: str, expected_count: int) -> List[List[float]]:
        """解析LLM评估响应，支持多种格式，特别处理缺少首个分数的情况"""
        
        self.logger.debug(f"解析LLM响应: {response_text[:200]}...")
        
        try:
            # 清理响应文本
            cleaned_text = response_text.strip()
            
            # 处理缺少第一个分数的情况，如：",8.0,6.0,5.5,6.8 6.0,6.5,5.0,7.0,6.1"
            # 使用正则表达式查找所有可能的分数段
            import re
            
            # 查找所有可能的分数模式
            # 包括：逗号分隔、空格分隔、以逗号开头的分数段
            score_segments = []
            
            # 方法1：按行分割
            lines = [line.strip() for line in cleaned_text.split('\n') if line.strip()]
            
            for line in lines:
                if not line.strip():
                    continue
                
                # 跳过明显的说明文字
                if any(keyword in line.lower() for keyword in ['标题', '评估', '维度', '分数', '输出', '示例']):
                    continue
                
                # 处理可能的分数行
                # 查找所有数字（包括小数）
                numbers = re.findall(r'\d+(?:\.\d+)?', line)
                
                if len(numbers) >= 3:  # 至少3个数字才认为是有效的分数行
                    try:
                        scores = [float(num) for num in numbers]
                        # 限制在合理范围内
                        scores = [max(0.0, min(10.0, score)) for score in scores]
                        score_segments.append(scores)
                        self.logger.debug(f"从行 '{line}' 解析出分数: {scores}")
                    except ValueError:
                        continue
            
            # 方法2：处理连续的分数段（如果按行分割失败）
            if not score_segments:
                # 尝试查找连续的数字分数段
                # 匹配类似 ",8.0,6.0,5.5,6.8 6.0,6.5,5.0,7.0,6.1" 的格式
                
                # 首先按空格或换行分割可能的分数段
                potential_segments = re.split(r'\s+', cleaned_text)
                
                for segment in potential_segments:
                    segment = segment.strip()
                    if not segment:
                        continue
                    
                    # 处理以逗号开头的分数段（缺少第一个分数）
                    if segment.startswith(','):
                        # 移除开头的逗号，添加默认分数
                        segment = '5.0' + segment  # 添加默认分数5.0
                    
                    # 查找分数
                    numbers = re.findall(r'\d+(?:\.\d+)?', segment)
                    
                    if len(numbers) >= 3:  # 至少3个数字
                        try:
                            scores = [float(num) for num in numbers]
                            scores = [max(0.0, min(10.0, score)) for score in scores]
                            score_segments.append(scores)
                            self.logger.debug(f"从段 '{segment}' 解析出分数: {scores}")
                        except ValueError:
                            continue
            
            # 方法3：如果还是没有找到，尝试直接解析整个文本
            if not score_segments:
                # 查找所有数字
                all_numbers = re.findall(r'\d+(?:\.\d+)?', cleaned_text)
                
                if len(all_numbers) >= 5:  # 至少5个数字
                    try:
                        # 将数字按5个一组分组
                        for i in range(0, len(all_numbers), 5):
                            if i + 4 < len(all_numbers):  # 确保有完整的5个数字
                                group = all_numbers[i:i+5]
                                scores = [float(num) for num in group]
                                scores = [max(0.0, min(10.0, score)) for score in scores]
                                score_segments.append(scores)
                    except ValueError:
                        pass
            
            if not score_segments:
                self.logger.warning("没有解析出任何分数，返回默认值")
                return [[5.0, 5.0, 5.0, 5.0, 5.0]] * expected_count
            
            # 确保每个分数段都有5个值
            normalized_scores = []
            for scores in score_segments:
                if len(scores) < 5:
                    # 不足5个则用平均值补充
                    avg_score = np.mean(scores) if scores else 5.0
                    scores.extend([avg_score] * (5 - len(scores)))
                elif len(scores) > 5:
                    # 超过5个则截取前5个
                    scores = scores[:5]
                
                # 转换为0-1范围
                normalized = [score / 10.0 for score in scores]
                normalized = [max(0.0, min(1.0, score)) for score in normalized]
                normalized_scores.append(normalized)
            
            # 调整数量以匹配期望值
            if len(normalized_scores) > expected_count:
                normalized_scores = normalized_scores[:expected_count]
            elif len(normalized_scores) < expected_count:
                # 不足则复制最后一组
                last_scores = normalized_scores[-1] if normalized_scores else [0.5, 0.5, 0.5, 0.5, 0.5]
                while len(normalized_scores) < expected_count:
                    normalized_scores.append(last_scores.copy())
            
            self.logger.info(f"解析完成，共{len(normalized_scores)}组分数，每组{len(normalized_scores[0])}个值")
            return normalized_scores
            
        except Exception as e:
            self.logger.error(f"解析LLM评估响应失败: {str(e)}")
            # 返回默认分数矩阵
            return [[0.5, 0.5, 0.5, 0.5, 0.5]] * expected_count
    
    def comprehensive_evaluation(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """综合评估"""
        
        eval_results = {}
        
        # 基础统计
        generated_titles = results.get('generated_titles', [])
        reference_titles = results.get('reference_titles', [])
        user_interests = results.get('user_interests', [])
        news_categories = results.get('news_categories', [])
        user_histories = results.get('user_histories', [])
        original_titles = results.get('original_titles', [])
        news_contents = results.get('news_contents', [])
        
        eval_results['basic_stats'] = self._calculate_basic_stats(generated_titles, reference_titles)
        
        # ROUGE评估
        if generated_titles and reference_titles:
            eval_results['rouge_scores'] = self.calculate_rouge_scores(generated_titles, reference_titles)
        
        # 个性化评估（规则基础）- 暂时禁用，因为效果不佳
        # if generated_titles and user_interests and news_categories:
        #     eval_results['rule_based_personalization'] = self.evaluate_personalization(
        #         generated_titles, user_interests, news_categories, user_histories)
        
        # 个性化评估（LLM）
        if self.use_llm_evaluation and generated_titles and user_interests and news_categories:
            self.logger.info("开始LLM个性化评估...")
            eval_results['llm_personalization'] = self.llm_evaluate_personalization(
                generated_titles, user_interests, news_categories, user_histories)
        
        # 标题质量评估
        eval_results['title_quality'] = self._evaluate_title_quality(generated_titles)
        
        # LLM质量评估
        if self.use_llm_evaluation and generated_titles and original_titles and news_contents:
            self.logger.info("开始LLM质量评估...")
            eval_results['llm_evaluation'] = self.llm_evaluate_quality(results)
        
        # 计算综合评分
        comprehensive_scores = self._calculate_comprehensive_score(eval_results)
        eval_results['comprehensive_scores'] = comprehensive_scores
        eval_results['overall_score'] = comprehensive_scores.get('final_comprehensive_score', 0.0)
        
        # 保存评估结果到实例属性
        self.evaluation_results = eval_results
        
        return eval_results
    
    def _calculate_basic_stats(self, generated_titles: List[str], reference_titles: List[str]) -> Dict[str, Any]:
        """计算基础统计信息"""
        
        stats = {
            'total_samples': len(generated_titles),
            'valid_samples': len([t for t in generated_titles if t and t.strip()]),
            'success_rate': 0.0,
            'avg_generated_length': 0.0,
            'avg_reference_length': 0.0
        }
        
        if stats['total_samples'] > 0:
            stats['success_rate'] = stats['valid_samples'] / stats['total_samples']
            
            valid_generated = [t for t in generated_titles if t and t.strip()]
            if valid_generated:
                stats['avg_generated_length'] = np.mean([len(t) for t in valid_generated])
            
            valid_reference = [t for t in reference_titles if t and t.strip()]
            if valid_reference:
                stats['avg_reference_length'] = np.mean([len(t) for t in valid_reference])
        
        return stats
    
    def _evaluate_title_quality(self, generated_titles: List[str]) -> Dict[str, float]:
        """评估标题质量（基于规则）"""
        
        if not generated_titles:
            return {}
        
        length_scores = []
        diversity_score = 0.0
        
        valid_titles = [t for t in generated_titles if t and t.strip()]
        
        if not valid_titles:
            return {}
        
        # 长度合理性评估
        for title in valid_titles:
            length = len(title)
            if 20 <= length <= 80:  # 理想长度范围
                length_scores.append(1.0)
            elif 10 <= length <= 100:  # 可接受范围
                length_scores.append(0.7)
            else:
                length_scores.append(0.3)
        
        # 多样性评估
        unique_titles = set(valid_titles)
        diversity_score = len(unique_titles) / len(valid_titles)
        
        return {
            'length_reasonableness': np.mean(length_scores),
            'title_diversity': diversity_score,
            'average_length': np.mean([len(t) for t in valid_titles])
        }
    
    def _calculate_comprehensive_score(self, eval_results: Dict[str, Any]) -> Dict[str, float]:
        """计算综合评分（分别计算ROUGE和LLM综合得分，并加权）"""
        
        scores_breakdown = {
            'rouge_score': 0.0,
            'llm_quality_score': 0.0,
            'llm_personalization_score': 0.0,
            'rule_personalization_score': 0.0,
            'title_quality_score': 0.0
        }
        
        # 1. ROUGE评分
        rouge_scores = eval_results.get('rouge_scores', {})
        if rouge_scores:
            rouge_avg = np.mean([
                rouge_scores.get('rouge1_f', 0),
                rouge_scores.get('rouge2_f', 0),
                rouge_scores.get('rougeL_f', 0)
            ])
            scores_breakdown['rouge_score'] = rouge_avg
        
        # 2. LLM质量评分（已经是0-1范围）
        llm_eval = eval_results.get('llm_evaluation', {})
        if llm_eval:
            llm_quality_score = llm_eval.get('llm_quality_score', 0)
            scores_breakdown['llm_quality_score'] = llm_quality_score  # 已经是0-1范围，无需转换
        
        # 3. LLM个性化评分
        llm_personalization = eval_results.get('llm_personalization', {})
        if llm_personalization:
            llm_personal_score = llm_personalization.get('llm_overall_personalization', 0)
            scores_breakdown['llm_personalization_score'] = llm_personal_score
        
        # 4. 标题质量评分
        title_quality = eval_results.get('title_quality', {})
        if title_quality:
            quality_avg = np.mean([
                title_quality.get('length_reasonableness', 0),
                title_quality.get('title_diversity', 0)
            ])
            scores_breakdown['title_quality_score'] = quality_avg
        
        # 计算加权综合分数（移除规则个性化评估后重新分配权重）
        # ROUGE权重35%, LLM质量权重30%, LLM个性化权重25%, 标题质量权重10%
        final_score = (
            scores_breakdown['rouge_score'] * 0.35 +
            scores_breakdown['llm_quality_score'] * 0.30 +
            scores_breakdown['llm_personalization_score'] * 0.25 +
            scores_breakdown['title_quality_score'] * 0.10
        )
        
        # 确保最终分数在0-1范围内
        final_score = min(max(final_score, 0.0), 1.0)
        scores_breakdown['final_comprehensive_score'] = final_score
        
        return scores_breakdown
    
    def generate_evaluation_report(self, eval_results: Dict[str, Any], save_path: Optional[str] = None) -> str:
        """生成详细评估报告（统一权重体系）"""
        
        report = "=" * 80 + "\n"
        report += "个性化新闻标题生成 - 详细评估报告\n"
        report += "=" * 80 + "\n\n"
        
        # 基本统计
        basic_stats = eval_results.get('basic_stats', {})
        report += "📊 基本统计信息:\n"
        report += f"├─ 总样本数: {basic_stats.get('total_samples', 0)}\n"
        report += f"├─ 有效样本数: {basic_stats.get('valid_samples', 0)}\n"
        report += f"└─ 成功率: {basic_stats.get('success_rate', 0):.2%}\n\n"
        
        # ROUGE评分（自动评估）
        rouge_scores = eval_results.get('rouge_scores', {})
        if rouge_scores:
            report += "📝 ROUGE评分 (自动评估 - 与参考标题的相似性):\n"
            report += f"├─ ROUGE-1 F-Score: {rouge_scores.get('rouge1_f', 0):.4f}\n"
            report += f"├─ ROUGE-2 F-Score: {rouge_scores.get('rouge2_f', 0):.4f}\n"
            report += f"└─ ROUGE-L F-Score: {rouge_scores.get('rougeL_f', 0):.4f}\n\n"
        
        # 调试信息：LLM质量评估
        llm_eval = eval_results.get('llm_evaluation', {})
        if llm_eval:
            report += "🤖 LLM质量评估详情:\n"
            llm_quality_score = llm_eval.get('llm_quality_score', 0)
            report += f"├─ 质量得分 (0-1): {llm_quality_score:.4f}\n"
            report += f"├─ 等效10分制: {llm_quality_score * 10:.2f}/10\n"
            report += f"└─ 评分标准差: {llm_eval.get('llm_quality_std', 0):.4f}\n\n"
        else:
            report += "⚠️ 警告: LLM质量评估未运行或失败\n\n"
        
        # LLM个性化评估
        llm_personalization = eval_results.get('llm_personalization', {})
        if llm_personalization:
            report += "🎯 LLM个性化评估 (大模型评分 - 个性化程度):\n"
            report += f"├─ 兴趣匹配度: {llm_personalization.get('llm_interest_match', 0):.4f}\n"
            report += f"├─ 类别相关性: {llm_personalization.get('llm_category_relevance', 0):.4f}\n"
            report += f"├─ 历史一致性: {llm_personalization.get('llm_history_consistency', 0):.4f}\n"
            report += f"├─ 个性化创新: {llm_personalization.get('llm_personalization_innovation', 0):.4f}\n"
            report += f"└─ LLM综合个性化: {llm_personalization.get('llm_overall_personalization', 0):.4f}\n\n"
        
        # 标题质量（规则评估）
        title_quality = eval_results.get('title_quality', {})
        if title_quality:
            report += "✨ 标题质量评估 (自动评估 - 基于规则):\n"
            report += f"├─ 长度合理性: {title_quality.get('length_reasonableness', 0):.4f}\n"
            report += f"├─ 标题多样性: {title_quality.get('title_diversity', 0):.4f}\n"
            report += f"└─ 平均标题长度: {title_quality.get('average_length', 0):.1f} 字符\n\n"
        
        # 综合评分详细展示（统一权重体系）
        comprehensive_scores = eval_results.get('comprehensive_scores', {})
        if comprehensive_scores:
            report += "🏆 综合评分详情 (加权计算):\n"
            report += f"├─ ROUGE得分 (权重35%): {comprehensive_scores.get('rouge_score', 0):.4f}\n"
            report += f"├─ LLM质量得分 (权重30%): {comprehensive_scores.get('llm_quality_score', 0):.4f}\n"
            report += f"├─ LLM个性化得分 (权重25%): {comprehensive_scores.get('llm_personalization_score', 0):.4f}\n"
            report += f"├─ 标题质量得分 (权重10%): {comprehensive_scores.get('title_quality_score', 0):.4f}\n"
            report += f"└─ 📈 最终综合得分: {comprehensive_scores.get('final_comprehensive_score', 0):.4f}\n\n"
        else:
            overall_score = eval_results.get('overall_score', 0)
            if isinstance(overall_score, dict):
                final_score = overall_score.get('final_comprehensive_score', 0)
            else:
                final_score = overall_score
            report += f"🏆 综合评分: {final_score:.4f}\n\n"
        
        # 评分说明
        report += "📋 评分说明:\n"
        report += "• ROUGE评分 (35%): 衡量生成标题与参考标题的词汇重叠度\n"
        report += "• LLM质量评分 (30%): 大模型从准确性、吸引力、清晰度等维度评分\n"
        report += "• LLM个性化评分 (25%): 大模型从个性化角度评估标题质量\n"
        report += "• 标题质量评分 (10%): 基于长度合理性、多样性等规则评估\n"
        report += "• 最终得分: 各项评分的加权平均（范围0-1）\n\n"
        
        # 评级
        final_score = 0.0
        if comprehensive_scores:
            final_score = comprehensive_scores.get('final_comprehensive_score', 0)
        else:
            overall_score = eval_results.get('overall_score', 0)
            if isinstance(overall_score, dict):
                final_score = overall_score.get('final_comprehensive_score', 0)
            else:
                final_score = overall_score
        
        report += "🎯 说明: 本评估集成了ROUGE自动评估和LLM智能评估，全面反映标题生成质量\n"
        
        # 保存报告
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
            self.logger.info(f"评估报告已保存到: {save_path}")
        
        return report
    
    def save_detailed_results(self, save_path: str):
        """保存详细评估结果到文件"""
        
        if not hasattr(self, 'evaluation_results') or not self.evaluation_results:
            self.logger.warning("没有评估结果可保存")
            return False
        
        try:
            # 创建输出目录
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # 准备保存数据
            save_data = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'evaluation_results': self.evaluation_results
            }
            
            # 保存到JSON文件
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2, default=str)
            
            self.logger.info(f"详细评估结果已保存到: {save_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"保存详细评估结果失败: {str(e)}")
            return False
    
    def create_comparison_chart(self, save_path: Optional[str] = None):
        """创建评估结果对比图表"""
        
        if not hasattr(self, 'evaluation_results') or not self.evaluation_results:
            self.logger.warning("没有评估结果可用于生成图表")
            return False
        
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            eval_results = self.evaluation_results
            
            # 创建2x2的子图
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('个性化新闻标题生成 - 评估结果', fontsize=16)
            
            # 1. ROUGE分数
            rouge_scores = eval_results.get('rouge_scores', {})
            if rouge_scores:
                rouge_metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
                rouge_values = [
                    rouge_scores.get('rouge1_f', 0),
                    rouge_scores.get('rouge2_f', 0),
                    rouge_scores.get('rougeL_f', 0)
                ]
                
                axes[0, 0].bar(rouge_metrics, rouge_values, color='skyblue', alpha=0.8)
                axes[0, 0].set_title('ROUGE F-Scores')
                axes[0, 0].set_ylabel('Score')
                axes[0, 0].set_ylim(0, 1)
                for i, v in enumerate(rouge_values):
                    axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
            
            # 2. LLM评估（质量+个性化）
            llm_eval = eval_results.get('llm_evaluation', {})
            llm_personalization = eval_results.get('llm_personalization', {})
            if llm_eval or llm_personalization:
                llm_labels = ['质量评分', '个性化评分']
                llm_values = [
                    llm_eval.get('llm_quality_score', 0),
                    llm_personalization.get('llm_overall_personalization', 0)
                ]
                
                axes[0, 1].bar(llm_labels, llm_values, color='lightcoral', alpha=0.8)
                axes[0, 1].set_title('LLM评估')
                axes[0, 1].set_ylabel('Score')
                axes[0, 1].set_ylim(0, 1)
                for i, v in enumerate(llm_values):
                    axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
            else:
                axes[0, 1].text(0.5, 0.5, 'LLM评估数据\n不可用', ha='center', va='center', 
                               transform=axes[0, 1].transAxes, fontsize=12)
                axes[0, 1].set_title('LLM评估')
            
            # 3. 标题质量
            title_quality = eval_results.get('title_quality', {})
            if title_quality:
                quality_labels = ['长度\n合理性', '标题\n多样性']
                quality_values = [
                    title_quality.get('length_reasonableness', 0),
                    title_quality.get('title_diversity', 0)
                ]
                
                axes[1, 0].bar(quality_labels, quality_values, color='lightgreen', alpha=0.8)
                axes[1, 0].set_title('标题质量')
                axes[1, 0].set_ylabel('Score')
                axes[1, 0].set_ylim(0, 1)
                for i, v in enumerate(quality_values):
                    axes[1, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
            else:
                axes[1, 0].text(0.5, 0.5, '标题质量数据\n不可用', ha='center', va='center', 
                               transform=axes[1, 0].transAxes, fontsize=12)
                axes[1, 0].set_title('标题质量')
            
            # 4. 综合评分饼图（修复负值问题）
            comprehensive_scores = eval_results.get('comprehensive_scores', {})
            if comprehensive_scores:
                overall_score = comprehensive_scores.get('final_comprehensive_score', 0)
            else:
                overall_score = eval_results.get('overall_score', 0)
                if isinstance(overall_score, dict):
                    overall_score = overall_score.get('final_comprehensive_score', 0)
            
            # 确保分数在0-1范围内
            overall_score = max(0.0, min(1.0, overall_score))
            remaining_score = 1.0 - overall_score
            
            # 确保两个值都不为负
            if overall_score < 0 or remaining_score < 0:
                overall_score = 0.5
                remaining_score = 0.5
            
            colors = ['gold', 'lightgray']
            wedge_sizes = [overall_score, remaining_score]
            
            axes[1, 1].pie(wedge_sizes, 
                           labels=['已达成', '待提升'], 
                           colors=colors,
                           autopct='%1.1f%%',
                           startangle=90)
            axes[1, 1].set_title(f'综合评分: {overall_score:.3f}')
            
            plt.tight_layout()
            
            # 保存图表
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"评估图表已保存到: {save_path}")
            
            plt.close()
            return True
            
        except ImportError as e:
            self.logger.warning(f"无法生成图表，缺少matplotlib: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"生成评估图表失败: {str(e)}")
            return False

if __name__ == "__main__":
    # 测试评估器
    evaluator = Evaluator()
    
    # 模拟数据
    test_results = {
        'generated_titles': [
            '科技巨头推出AI新品',
            '智能手机性能大升级',
            '新技术引领行业变革'
        ],
        'reference_titles': [
            '科技公司发布新产品',
            '手机厂商推出旗舰机',
            '技术创新推动发展'
        ],
        'user_interests': [
            {'primary_interest': 'Technology', 'categories': ['AI', 'Mobile']},
            {'primary_interest': 'Technology', 'categories': ['Hardware']},
            {'primary_interest': 'Business', 'categories': ['Innovation']}
        ],
        'news_categories': ['Technology', 'Technology', 'Business']
    }
    
    # 执行评估
    results = evaluator.comprehensive_evaluation(test_results)
    
    # 生成报告
    report = evaluator.generate_evaluation_report(results)
    print(report)
    
    # 保存结果
    evaluator.save_detailed_results('./test_evaluation_results.json') 