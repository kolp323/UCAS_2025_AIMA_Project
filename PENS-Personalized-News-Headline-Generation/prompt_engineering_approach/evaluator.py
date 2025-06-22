"""
Evaluation module - Contains LLM API evaluation and personalization effect assessment
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

# Fix ROUGE import
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    print("Warning: rouge_score package not installed, will use simplified evaluation methods")
    ROUGE_AVAILABLE = False

from config import EVALUATION_CONFIG, DATA_PATHS, EVALUATION_MODEL
from llm_client import LLMClient

class Evaluator:
    """Evaluator"""
    
    def __init__(self, use_llm_evaluation: bool = True):
        self.logger = self._setup_logger()
        self.rouge_scorer = None
        self.use_llm_evaluation = use_llm_evaluation
        
        # Initialize LLM client for evaluation
        if use_llm_evaluation:
            # Directly use specified evaluation model to initialize LLM client
            self.llm_client = LLMClient(EVALUATION_MODEL)
            self.logger.info(f"Using {EVALUATION_MODEL} for LLM evaluation")
        
        # Initialize ROUGE evaluator
        if ROUGE_AVAILABLE:
            try:
                self.rouge_scorer = rouge_scorer.RougeScorer(
                    ['rouge1', 'rouge2', 'rougeL'], 
                    use_stemmer=True
                )
                self.logger.info("ROUGE scorer initialized successfully")
            except Exception as e:
                self.logger.warning(f"ROUGE scorer initialization failed: {e}")
                self.rouge_scorer = None
        
        # Evaluation results storage
        self.evaluation_results = {}
        
    def _setup_logger(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def calculate_rouge_scores(self, generated_titles: List[str], reference_titles: List[str]) -> Dict[str, float]:
        """Calculate ROUGE scores"""
        
        if not self.rouge_scorer:
            self.logger.warning("ROUGE evaluator not available, using simplified evaluation")
            return self._calculate_simple_scores(generated_titles, reference_titles)
        
        try:
            # Preprocess text
            processed_generated = [self._preprocess_text_for_rouge(title) for title in generated_titles]
            processed_reference = [self._preprocess_text_for_rouge(title) for title in reference_titles]
            
            # Filter empty text
            valid_pairs = [(g, r) for g, r in zip(processed_generated, processed_reference) 
                          if g and r and g != "empty text" and r != "empty text"]
            
            if not valid_pairs:
                self.logger.error("No valid text pairs for ROUGE evaluation")
                return {}
            
            # Calculate ROUGE scores
            rouge1_scores = []
            rouge2_scores = []
            rougeL_scores = []
            
            for generated, reference in valid_pairs:
                scores = self.rouge_scorer.score(reference, generated)
                rouge1_scores.append(scores['rouge1'])
                rouge2_scores.append(scores['rouge2'])
                rougeL_scores.append(scores['rougeL'])
            
            # Calculate average scores
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
            
            self.logger.info(f"ROUGE evaluation completed, valid samples: {len(valid_pairs)}")
            return rouge_results
            
        except Exception as e:
            self.logger.error(f"ROUGE evaluation failed: {str(e)}")
            return self._calculate_simple_scores(generated_titles, reference_titles)
    
    def _preprocess_text_for_rouge(self, text: str) -> str:
        """Preprocess text for ROUGE evaluation"""
        if not text or not isinstance(text, str):
            return "empty text"
        
        # Remove extra spaces and line breaks
        text = ' '.join(text.split())
        
        # Ensure text is not empty
        if not text.strip():
            return "empty text"
            
        return text.strip()
    
    def _calculate_simple_scores(self, generated_titles: List[str], reference_titles: List[str]) -> Dict[str, float]:
        """Simplified evaluation method (when ROUGE is not available)"""
        
        scores = {}
        total_pairs = len(generated_titles)
        
        if total_pairs == 0:
            return scores
        
        # Calculate character-level overlap
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
        """Personalization effectiveness evaluation (rule-based)"""
        
        personalization_scores = []
        category_relevance_scores = []
        interest_alignment_scores = []
        history_relevance_scores = []
        
        for i, (title, interests, category, history) in enumerate(
            zip(generated_titles, user_interests, news_categories, user_histories)):
            
            if not title or not isinstance(interests, dict):
                continue
            
            title_lower = title.lower()
            
            # 1. Interest matching evaluation (improved logic)
            primary_interest = interests.get('primary_interest', '').lower()
            interest_categories = [cat.lower() for cat in interests.get('categories', [])]
            
            interest_score = 0.0
            
            # Direct category matching (weight 60%)
            if category and category.lower() in interest_categories:
                interest_score += 0.6
            
            # Primary interest matching (weight 25%)
            if primary_interest and primary_interest in title_lower:
                interest_score += 0.25
            
            # Partial keyword matching (weight 15%)
            if primary_interest:
                primary_words = primary_interest.split()
                matched_words = sum(1 for word in primary_words if len(word) > 3 and word in title_lower)
                if primary_words:
                    interest_score += 0.15 * (matched_words / len(primary_words))
            
            personalization_scores.append(min(interest_score, 1.0))
            
            # 2. Category relevance evaluation (improved)
            category_score = 0.0
            if category and interest_categories:
                if category.lower() in interest_categories:
                    category_score = 1.0  # Direct match
                else:
                    # Semantic relevance check
                    category_words = set(category.lower().split())
                    for cat in interest_categories:
                        cat_words = set(cat.split())
                        if category_words.intersection(cat_words):
                            category_score = max(category_score, 0.5)
            
            category_relevance_scores.append(category_score)
            
            # 3. Interest alignment evaluation (more reasonable)
            alignment_score = 0.0
            if category and interest_categories:
                if category.lower() in interest_categories:
                    alignment_score = 1.0
                elif 'news' in interest_categories and category in ['news', 'politics', 'business']:
                    alignment_score = 0.8  # News-related categories
                elif 'finance' in interest_categories and category in ['business', 'finance', 'economy']:
                    alignment_score = 0.8  # Finance-related categories
                else:
                    alignment_score = 0.4  # Base score
            
            interest_alignment_scores.append(alignment_score)
            
            # 4. History relevance evaluation (improved)
            history_score = 0.0
            if history and len(history) > 0:
                title_words = set(title_lower.split())
                history_words = set()
                for hist_title in history[:15]:  # Look at more history
                    if hist_title:
                        history_words.update(hist_title.lower().split())
                
                if history_words:
                    common_words = title_words.intersection(history_words)
                    # Filter stop words and short words
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
        
        # Calculate comprehensive rule-based score
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
        """Use LLM to evaluate personalization effectiveness"""
        
        if not self.use_llm_evaluation or not self.llm_client:
            self.logger.warning("LLM personalization evaluation not available")
            return {}
        
        system_prompt = """You are a professional personalized news recommendation evaluation expert. Your task is to score the personalization level of news headlines.

Evaluation dimensions (each dimension 0-10 points):
1. Interest matching: Degree of match between headline content and user interests
2. Category relevance: Relevance of headline to user preferred categories
3. History consistency: Consistency with user's historical reading habits
4. Personalization innovation: Innovation and attractiveness in personalization
5. Overall personalization: Overall personalization effectiveness

Strict output requirements:
- Only output numbers and commas, no text explanations
- One line per headline evaluation, format: score1,score2,score3,score4,score5
- Score range 0-10, can be decimal (e.g., 8.5)
- Output in order of headlines
- Do not output any explanations, analysis, or other text

Example output:
8.5,7.0,9.0,6.5,8.0
6.0,8.5,5.5,7.0,6.5"""

        all_scores = {
            'interest_match': [],
            'category_relevance': [],
            'history_consistency': [],
            'personalization_innovation': [],
            'overall_personalization': []
        }
        
        # Process in batches
        for i in range(0, len(generated_titles), batch_size):
            batch_titles = generated_titles[i:i+batch_size]
            batch_interests = user_interests[i:i+batch_size]
            batch_categories = news_categories[i:i+batch_size]
            # Ensure user_histories is not empty and has correct length
            if user_histories and len(user_histories) > i:
                batch_histories = user_histories[i:i+batch_size]
            else:
                batch_histories = [[] for _ in batch_titles]
            
            # Build batch evaluation prompt
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
                            # Default medium scores
                            for key in all_scores:
                                all_scores[key].append(5.0)
                else:
                    # Add default scores
                    for _ in range(len(batch_titles)):
                        for key in all_scores:
                            all_scores[key].append(5.0)
                            
            except Exception as e:
                self.logger.error(f"LLM personalization evaluation failed: {e}")
                for _ in range(len(batch_titles)):
                    for key in all_scores:
                        all_scores[key].append(5.0)
        
        # Calculate average scores (convert to 0-1 range)
        results = {}
        for key, scores in all_scores.items():
            if scores:
                results[f'llm_{key}'] = np.mean(scores) / 10.0  # Convert to 0-1 range
                results[f'llm_{key}_std'] = np.std(scores) / 10.0
        
        return results
    
    def _build_personalization_evaluation_prompt(self, titles: List[str], 
                                                interests: List[Dict], 
                                                categories: List[str],
                                                histories: List[List[str]]) -> str:
        """Build personalization evaluation prompt"""
        
        prompt = "As a personalized news recommendation evaluation expert, please score the personalization level of each news headline below.\n\n"
        
        prompt += "Evaluation dimensions (each dimension 0-10 points):\n"
        prompt += "Dimension 1: Interest matching - Degree of match between headline and user interests\n"
        prompt += "Dimension 2: Category relevance - Relevance of headline to user preferred categories\n"  
        prompt += "Dimension 3: History consistency - Consistency with user's historical reading\n"
        prompt += "Dimension 4: Personalization innovation - Personalized attractiveness of headline\n"
        prompt += "Dimension 5: Overall personalization - Overall personalization effectiveness\n\n"
        
        prompt += "News headlines and user information to be evaluated:\n"
        
        for i, (title, interest, category, history) in enumerate(zip(titles, interests, categories, histories)):
            prompt += f"\n=== Title {i+1} ===\n"
            prompt += f"News headline: \"{title}\"\n"
            prompt += f"News category: {category}\n"
            
            primary = interest.get('primary_interest', 'Unknown')
            categories_list = interest.get('categories', [])
            prompt += f"User interests: Primary interest={primary}, Preferred categories={categories_list}\n"
            
            if history and len(history) > 0:
                recent = history[:3]  # Show latest 3 items
                prompt += f"Reading history: {'; '.join(recent)}\n"
            else:
                prompt += "Reading history: No records\n"
        
        prompt += "\nOutput requirements:\n"
        prompt += "- Must score in order of headlines (1 to N)\n"
        prompt += "- Output one line per headline evaluation, format: dimension1_score,dimension2_score,dimension3_score,dimension4_score,dimension5_score\n"
        prompt += "- Only output numbers and commas, no text explanations\n"
        prompt += "- Score range: 0-10 (can be decimal, e.g., 8.5)\n\n"
        
        prompt += "Example output:\n"
        prompt += "8,7,9,6,8\n"
        prompt += "6,8,5,7,6\n\n"
        
        prompt += "Please start scoring (output one line of scores for each headline in order):\n"
        
        return prompt
    
    def _parse_personalization_evaluation_response(self, response: str) -> List[List[float]]:
        """Parse LLM personalization evaluation response, supporting multiple formats including missing first score"""
        
        import re
        
        self.logger.debug(f"Parse personalization evaluation response: {response[:200]}...")
        
        try:
            cleaned_text = response.strip()
            score_segments = []
            
            # Method 1: Split by lines
            lines = [line.strip() for line in cleaned_text.split('\n') if line.strip()]
            
            for line in lines:
                if not line.strip():
                    continue
                
                # Skip obvious explanatory text
                if any(keyword in line.lower() for keyword in ['title', 'evaluation', 'dimension', 'score', 'output', 'example', 'as', 'please']):
                    continue
                
                # Find all numbers (including decimals)
                numbers = re.findall(r'\d+(?:\.\d+)?', line)
                
                if len(numbers) >= 3:  # At least 3 numbers to consider as valid score line
                    try:
                        scores = [float(num) for num in numbers]
                        # Limit to reasonable range
                        scores = [max(0.0, min(10.0, score)) for score in scores]
                        score_segments.append(scores)
                        self.logger.debug(f"Parsed personalization scores from line '{line}': {scores}")
                    except ValueError:
                        continue
            
            # Method 2: Process continuous score segments (if line splitting fails)
            if not score_segments:
                # Split possible score segments by space
                potential_segments = re.split(r'\s+', cleaned_text)
                
                for segment in potential_segments:
                    segment = segment.strip()
                    if not segment:
                        continue
                    
                    # Handle segments starting with comma (missing first score)
                    if segment.startswith(','):
                        # Add default score 5.0
                        segment = '5.0' + segment
                    
                    # Find scores
                    numbers = re.findall(r'\d+(?:\.\d+)?', segment)
                    
                    if len(numbers) >= 3:  # At least 3 numbers
                        try:
                            scores = [float(num) for num in numbers]
                            scores = [max(0.0, min(10.0, score)) for score in scores]
                            score_segments.append(scores)
                            self.logger.debug(f"Parsed personalization scores from segment '{segment}': {scores}")
                        except ValueError:
                            continue
            
            # Method 3: Parse all numbers directly
            if not score_segments:
                all_numbers = re.findall(r'\d+(?:\.\d+)?', cleaned_text)
                
                if len(all_numbers) >= 5:  # At least 5 numbers
                    try:
                        # Group numbers into sets of 5
                        for i in range(0, len(all_numbers), 5):
                            if i + 4 < len(all_numbers):  # Ensure complete set of 5 numbers
                                group = all_numbers[i:i+5]
                                scores = [float(num) for num in group]
                                scores = [max(0.0, min(10.0, score)) for score in scores]
                                score_segments.append(scores)
                    except ValueError:
                        pass
            
            if not score_segments:
                self.logger.warning("No personalization scores parsed, returning default values")
                return [[5.0, 5.0, 5.0, 5.0, 5.0]]
            
            # Ensure each score segment has 5 values
            normalized_scores = []
            for scores in score_segments:
                if len(scores) < 5:
                    # If less than 5, supplement with average value
                    avg_score = np.mean(scores) if scores else 5.0
                    scores.extend([avg_score] * (5 - len(scores)))
                elif len(scores) > 5:
                    # If more than 5, take first 5
                    scores = scores[:5]
                
                normalized_scores.append(scores)
            
            self.logger.info(f"Personalization evaluation parsing complete, {len(normalized_scores)} score groups")
            return normalized_scores
            
        except Exception as e:
            self.logger.error(f"Failed to parse personalization evaluation response: {str(e)}")
            return [[5.0, 5.0, 5.0, 5.0, 5.0]]
    
    def llm_evaluate_quality(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to evaluate title quality (fixed version)"""
        
        self.logger.info("Starting LLM quality evaluation...")
        
        generated_titles = results.get('generated_titles', [])
        reference_titles = results.get('reference_titles', [])
        # Fix: Unified use of news_contents field name
        news_bodies = results.get('news_contents', []) or results.get('news_bodies', [])
        
        # Check data integrity, provide more detailed warning information
        if not generated_titles:
            self.logger.warning("Missing generated title data")
            return {
                'llm_quality_score': 0.0,
                'llm_quality_scores': []
            }
        
        if not reference_titles:
            self.logger.warning("Missing reference title data")
            return {
                'llm_quality_score': 0.0,
                'llm_quality_scores': []
            }
            
        if not news_bodies:
            self.logger.warning("Missing news content data")
            return {
                'llm_quality_score': 0.0,
                'llm_quality_scores': []
            }
        
        # Check data length consistency
        if len(generated_titles) != len(reference_titles) or len(generated_titles) != len(news_bodies):
            self.logger.warning(f"Data length inconsistent: generated titles {len(generated_titles)}, reference titles {len(reference_titles)}, news content {len(news_bodies)}")
            # Truncate to shortest length
            min_length = min(len(generated_titles), len(reference_titles), len(news_bodies))
            generated_titles = generated_titles[:min_length]
            reference_titles = reference_titles[:min_length]
            news_bodies = news_bodies[:min_length]
            self.logger.info(f"Adjusted data length to: {min_length}")
        
        self.logger.info(f"LLM quality evaluation ready, {len(generated_titles)} samples total")
        
        # Batch processing, 5 per batch
        batch_size = 5
        all_scores = []
        
        for i in range(0, len(generated_titles), batch_size):
            batch_generated = generated_titles[i:i+batch_size]
            batch_reference = reference_titles[i:i+batch_size]
            batch_bodies = news_bodies[i:i+batch_size]
            
            self.logger.info(f"Evaluating batch {i//batch_size + 1}, containing {len(batch_generated)} titles")
            
            try:
                batch_scores = self._llm_evaluate_batch_quality(
                    batch_generated, batch_reference, batch_bodies
                )
                
                if batch_scores:
                    # batch_scores is a 2D list, each sublist contains 5 dimension scores
                    all_scores.extend(batch_scores)
                    self.logger.info(f"Batch evaluation successful, obtained {len(batch_scores)} scores")
                else:
                    # If batch evaluation fails, add default scores
                    default_scores = [[0.5, 0.5, 0.5, 0.5, 0.5] for _ in range(len(batch_generated))]
                    all_scores.extend(default_scores)
                    self.logger.warning(f"Batch evaluation failed, using default scores")
                    
            except Exception as e:
                self.logger.error(f"Batch evaluation error: {e}")
                # Add default scores
                default_scores = [[0.5, 0.5, 0.5, 0.5, 0.5] for _ in range(len(batch_generated))]
                all_scores.extend(default_scores)
        
        if not all_scores:
            self.logger.warning("No LLM quality scores obtained, returning default values")
            return {
                'llm_quality_score': 0.0,
                'llm_quality_scores': []
            }
        
        # Calculate comprehensive score for each title (average of five dimensions)
        title_scores = []
        for score_list in all_scores:
            if isinstance(score_list, list) and len(score_list) >= 5:
                # Take average of first 5 dimension scores
                avg_score = sum(score_list[:5]) / 5.0
                # Ensure in 0-1 range
                normalized_score = max(0.0, min(1.0, avg_score))
                title_scores.append(normalized_score)
            else:
                # If format is incorrect, use default score
                title_scores.append(0.5)
        
        # Calculate overall average score
        overall_score = sum(title_scores) / len(title_scores) if title_scores else 0.0
        
        self.logger.info(f"LLM quality evaluation complete, average score: {overall_score:.3f}")
        
        return {
            'llm_quality_score': overall_score,
            'llm_quality_scores': title_scores
        }
    
    def _llm_evaluate_batch_quality(self, generated_titles: List[str], 
                                     reference_titles: List[str],
                                     news_bodies: List[str]) -> List[List[float]]:
        """Batch LLM quality evaluation"""
        
        system_prompt = """You are a professional news headline quality evaluation expert. Please score the quality of generated news headlines.

Evaluation dimensions (each dimension 0-10 points):
1. Accuracy: Whether the headline accurately reflects the news content
2. Attractiveness: Whether the headline can attract readers' attention
3. Clarity: Whether the headline expression is clear and understandable
4. Conciseness: Whether the headline length and expression are concise
5. Overall quality: Overall headline quality

Strict output requirements:
- Only output numbers and commas, no text explanations
- One line per headline evaluation, format: score1,score2,score3,score4,score5
- Score range 0-10, can be decimal (e.g., 8.5)
- Output in order of headlines
- Do not output any explanations, analysis, or other text

Example output:
8.5,7.0,9.0,6.5,8.0
6.0,8.5,5.5,7.0,6.5"""
        
        user_prompt = "Please evaluate the quality of the following news headlines:\n\n"
        
        for i, (gen, orig, content) in enumerate(zip(generated_titles, reference_titles, news_bodies)):
            user_prompt += f"=== Title {i+1} ===\n"
            user_prompt += f"Original title: {orig}\n"
            user_prompt += f"Generated title: {gen}\n"
            user_prompt += f"News content: {content[:200]}...\n\n"
        
        user_prompt += "Please output scores for each title in order (5 dimension scores separated by commas):"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = self.llm_client.chat_completion(messages, max_tokens=1000)
            if response:
                batch_scores = self._parse_llm_evaluation_response(response, len(generated_titles))
                if batch_scores:  # Ensure parsing success
                    return batch_scores
                else:
                    # Parsing failed, use default scores
                    return [[0.5, 0.5, 0.5, 0.5, 0.5]] * len(generated_titles)
            else:
                # API call failed, use default scores
                return [[0.5, 0.5, 0.5, 0.5, 0.5]] * len(generated_titles)
            
        except Exception as e:
            self.logger.error(f"LLM quality evaluation failed: {e}")
            # Return default score matrix
            return [[0.5, 0.5, 0.5, 0.5, 0.5]] * len(generated_titles)
    
    def _parse_llm_evaluation_response(self, response_text: str, expected_count: int) -> List[List[float]]:
        """Parse LLM evaluation response, supporting multiple formats, especially handling missing first score"""
        
        self.logger.debug(f"Parsing LLM response: {response_text[:200]}...")
        
        try:
            # Clean response text
            cleaned_text = response_text.strip()
            
            # Handle case with missing first score, e.g.: ",8.0,6.0,5.5,6.8 6.0,6.5,5.0,7.0,6.1"
            # Use regular expressions to find all possible score segments
            import re
            
            # Find all possible score patterns
            # Including: comma-separated, space-separated, segments starting with comma
            score_segments = []
            
            # Method 1: Split by lines
            lines = [line.strip() for line in cleaned_text.split('\n') if line.strip()]
            
            for line in lines:
                if not line.strip():
                    continue
                
                # Skip obvious explanatory text
                if any(keyword in line.lower() for keyword in ['title', 'evaluation', 'dimension', 'score', 'output', 'example']):
                    continue
                
                # Handle possible score lines
                # Find all numbers (including decimals)
                numbers = re.findall(r'\d+(?:\.\d+)?', line)
                
                if len(numbers) >= 3:  # At least 3 numbers to consider as valid score line
                    try:
                        scores = [float(num) for num in numbers]
                        # Limit to reasonable range
                        scores = [max(0.0, min(10.0, score)) for score in scores]
                        score_segments.append(scores)
                        self.logger.debug(f"Parsed scores from line '{line}': {scores}")
                    except ValueError:
                        continue
            
            # Method 2: Process continuous score segments (if line splitting fails)
            if not score_segments:
                # Try to find continuous numeric score segments
                # Match formats like ",8.0,6.0,5.5,6.8 6.0,6.5,5.0,7.0,6.1"
                
                # First split possible score segments by space or newline
                potential_segments = re.split(r'\s+', cleaned_text)
                
                for segment in potential_segments:
                    segment = segment.strip()
                    if not segment:
                        continue
                    
                    # Handle segments starting with comma (missing first score)
                    if segment.startswith(','):
                        # Remove leading comma, add default score
                        segment = '5.0' + segment  # Add default score 5.0
                    
                    # Find scores
                    numbers = re.findall(r'\d+(?:\.\d+)?', segment)
                    
                    if len(numbers) >= 3:  # At least 3 numbers
                        try:
                            scores = [float(num) for num in numbers]
                            scores = [max(0.0, min(10.0, score)) for score in scores]
                            score_segments.append(scores)
                            self.logger.debug(f"Parsed scores from segment '{segment}': {scores}")
                        except ValueError:
                            continue
            
            # Method 3: If still not found, try parsing entire text directly
            if not score_segments:
                # Find all numbers
                all_numbers = re.findall(r'\d+(?:\.\d+)?', cleaned_text)
                
                if len(all_numbers) >= 5:  # At least 5 numbers
                    try:
                        # Group numbers into sets of 5
                        for i in range(0, len(all_numbers), 5):
                            if i + 4 < len(all_numbers):  # Ensure complete set of 5 numbers
                                group = all_numbers[i:i+5]
                                scores = [float(num) for num in group]
                                scores = [max(0.0, min(10.0, score)) for score in scores]
                                score_segments.append(scores)
                    except ValueError:
                        pass
            
            if not score_segments:
                self.logger.warning("No scores parsed, returning default values")
                return [[5.0, 5.0, 5.0, 5.0, 5.0]] * expected_count
            
            # Ensure each score segment has 5 values
            normalized_scores = []
            for scores in score_segments:
                if len(scores) < 5:
                    # If less than 5, supplement with average value
                    avg_score = np.mean(scores) if scores else 5.0
                    scores.extend([avg_score] * (5 - len(scores)))
                elif len(scores) > 5:
                    # If more than 5, take first 5
                    scores = scores[:5]
                
                # Convert to 0-1 range
                normalized = [score / 10.0 for score in scores]
                normalized = [max(0.0, min(1.0, score)) for score in normalized]
                normalized_scores.append(normalized)
            
            # Adjust quantity to match expected value
            if len(normalized_scores) > expected_count:
                normalized_scores = normalized_scores[:expected_count]
            elif len(normalized_scores) < expected_count:
                # If insufficient, duplicate last group
                last_scores = normalized_scores[-1] if normalized_scores else [0.5, 0.5, 0.5, 0.5, 0.5]
                while len(normalized_scores) < expected_count:
                    normalized_scores.append(last_scores.copy())
            
            self.logger.info(f"Parsing complete, {len(normalized_scores)} score groups, {len(normalized_scores[0])} values per group")
            return normalized_scores
            
        except Exception as e:
            self.logger.error(f"Failed to parse LLM evaluation response: {str(e)}")
            # Return default score matrix
            return [[0.5, 0.5, 0.5, 0.5, 0.5]] * expected_count
    
    def comprehensive_evaluation(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive evaluation"""
        
        eval_results = {}
        
        # Basic statistics
        generated_titles = results.get('generated_titles', [])
        reference_titles = results.get('reference_titles', [])
        user_interests = results.get('user_interests', [])
        news_categories = results.get('news_categories', [])
        user_histories = results.get('user_histories', [])
        original_titles = results.get('original_titles', [])
        news_contents = results.get('news_contents', [])
        
        eval_results['basic_stats'] = self._calculate_basic_stats(generated_titles, reference_titles)
        
        # ROUGE evaluation
        if generated_titles and reference_titles:
            eval_results['rouge_scores'] = self.calculate_rouge_scores(generated_titles, reference_titles)
        
        # Personalization evaluation (rule-based) - temporarily disabled due to poor performance
        # if generated_titles and user_interests and news_categories:
        #     eval_results['rule_based_personalization'] = self.evaluate_personalization(
        #         generated_titles, user_interests, news_categories, user_histories)
        
        # Personalization evaluation (LLM)
        if self.use_llm_evaluation and generated_titles and user_interests and news_categories:
            self.logger.info("Starting LLM personalization evaluation...")
            eval_results['llm_personalization'] = self.llm_evaluate_personalization(
                generated_titles, user_interests, news_categories, user_histories)
        
        # Title quality evaluation
        eval_results['title_quality'] = self._evaluate_title_quality(generated_titles)
        
        # LLM quality evaluation
        if self.use_llm_evaluation and generated_titles and original_titles and news_contents:
            self.logger.info("Starting LLM quality evaluation...")
            eval_results['llm_evaluation'] = self.llm_evaluate_quality(results)
        
        # Calculate comprehensive score
        comprehensive_scores = self._calculate_comprehensive_score(eval_results)
        eval_results['comprehensive_scores'] = comprehensive_scores
        eval_results['overall_score'] = comprehensive_scores.get('final_comprehensive_score', 0.0)
        
        # Save evaluation results to instance attribute
        self.evaluation_results = eval_results
        
        return eval_results
    
    def _calculate_basic_stats(self, generated_titles: List[str], reference_titles: List[str]) -> Dict[str, Any]:
        """Calculate basic statistics"""
        
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
        """Evaluate title quality (rule-based)"""
        
        if not generated_titles:
            return {}
        
        length_scores = []
        diversity_score = 0.0
        
        valid_titles = [t for t in generated_titles if t and t.strip()]
        
        if not valid_titles:
            return {}
        
        # Length reasonableness evaluation
        for title in valid_titles:
            length = len(title)
            if 20 <= length <= 80:  # Ideal length range
                length_scores.append(1.0)
            elif 10 <= length <= 100:  # Acceptable range
                length_scores.append(0.7)
            else:
                length_scores.append(0.3)
        
        # Diversity evaluation
        unique_titles = set(valid_titles)
        diversity_score = len(unique_titles) / len(valid_titles)
        
        return {
            'length_reasonableness': np.mean(length_scores),
            'title_diversity': diversity_score,
            'average_length': np.mean([len(t) for t in valid_titles])
        }
    
    def _calculate_comprehensive_score(self, eval_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate comprehensive score (separately calculate ROUGE and LLM scores, then weight them)"""
        
        scores_breakdown = {
            'rouge_score': 0.0,
            'llm_quality_score': 0.0,
            'llm_personalization_score': 0.0,
            'rule_personalization_score': 0.0,
            'title_quality_score': 0.0
        }
        
        # 1. ROUGE score
        rouge_scores = eval_results.get('rouge_scores', {})
        if rouge_scores:
            rouge_avg = np.mean([
                rouge_scores.get('rouge1_f', 0),
                rouge_scores.get('rouge2_f', 0),
                rouge_scores.get('rougeL_f', 0)
            ])
            scores_breakdown['rouge_score'] = rouge_avg
        
        # 2. LLM quality score (already in 0-1 range)
        llm_eval = eval_results.get('llm_evaluation', {})
        if llm_eval:
            llm_quality_score = llm_eval.get('llm_quality_score', 0)
            scores_breakdown['llm_quality_score'] = llm_quality_score  # Already in 0-1 range, no conversion needed
        
        # 3. LLM personalization score
        llm_personalization = eval_results.get('llm_personalization', {})
        if llm_personalization:
            llm_personal_score = llm_personalization.get('llm_overall_personalization', 0)
            scores_breakdown['llm_personalization_score'] = llm_personal_score
        
        # 4. Title quality score
        title_quality = eval_results.get('title_quality', {})
        if title_quality:
            quality_avg = np.mean([
                title_quality.get('length_reasonableness', 0),
                title_quality.get('title_diversity', 0)
            ])
            scores_breakdown['title_quality_score'] = quality_avg
        
        # Calculate weighted comprehensive score (redistributed weights after removing rule-based personalization evaluation)
        # ROUGE weight 35%, LLM quality weight 30%, LLM personalization weight 25%, title quality weight 10%
        final_score = (
            scores_breakdown['rouge_score'] * 0.35 +
            scores_breakdown['llm_quality_score'] * 0.30 +
            scores_breakdown['llm_personalization_score'] * 0.25 +
            scores_breakdown['title_quality_score'] * 0.10
        )
        
        # Ensure final score is in 0-1 range
        final_score = min(max(final_score, 0.0), 1.0)
        scores_breakdown['final_comprehensive_score'] = final_score
        
        return scores_breakdown
    
    def generate_evaluation_report(self, eval_results: Dict[str, Any], save_path: Optional[str] = None) -> str:
        """Generate detailed evaluation report (unified weight system)"""
        
        report = "=" * 80 + "\n"
        report += "Personalized News Headline Generation - Detailed Evaluation Report\n"
        report += "=" * 80 + "\n\n"
        
        # Basic statistics
        basic_stats = eval_results.get('basic_stats', {})
        report += "Basic Statistics:\n"
        report += f"├─ Total samples: {basic_stats.get('total_samples', 0)}\n"
        report += f"├─ Valid samples: {basic_stats.get('valid_samples', 0)}\n"
        report += f"└─ Success rate: {basic_stats.get('success_rate', 0):.2%}\n\n"
        
        # ROUGE score (automatic evaluation)
        rouge_scores = eval_results.get('rouge_scores', {})
        if rouge_scores:
            report += "ROUGE Scores (Automatic Evaluation - Similarity to Reference Titles):\n"
            report += f"├─ ROUGE-1 F-Score: {rouge_scores.get('rouge1_f', 0):.4f}\n"
            report += f"├─ ROUGE-2 F-Score: {rouge_scores.get('rouge2_f', 0):.4f}\n"
            report += f"└─ ROUGE-L F-Score: {rouge_scores.get('rougeL_f', 0):.4f}\n\n"
        
        # Debug info: LLM quality evaluation
        llm_eval = eval_results.get('llm_evaluation', {})
        if llm_eval:
            report += "LLM Quality Evaluation Details:\n"
            llm_quality_score = llm_eval.get('llm_quality_score', 0)
            report += f"├─ Quality score (0-1): {llm_quality_score:.4f}\n"
            report += f"├─ Equivalent 10-point scale: {llm_quality_score * 10:.2f}/10\n"
            report += f"└─ Score standard deviation: {llm_eval.get('llm_quality_std', 0):.4f}\n\n"
        else:
            report += "Warning: LLM quality evaluation not run or failed\n\n"
        
        # LLM personalization evaluation
        llm_personalization = eval_results.get('llm_personalization', {})
        if llm_personalization:
            report += "LLM Personalization Evaluation (Large Model Scoring - Personalization Degree):\n"
            report += f"├─ Interest matching: {llm_personalization.get('llm_interest_match', 0):.4f}\n"
            report += f"├─ Category relevance: {llm_personalization.get('llm_category_relevance', 0):.4f}\n"
            report += f"├─ History consistency: {llm_personalization.get('llm_history_consistency', 0):.4f}\n"
            report += f"├─ Personalization innovation: {llm_personalization.get('llm_personalization_innovation', 0):.4f}\n"
            report += f"└─ LLM overall personalization: {llm_personalization.get('llm_overall_personalization', 0):.4f}\n\n"
        
        # Title quality (rule evaluation)
        title_quality = eval_results.get('title_quality', {})
        if title_quality:
            report += "Title Quality Evaluation (Automatic Evaluation - Rule-based):\n"
            report += f"├─ Length reasonableness: {title_quality.get('length_reasonableness', 0):.4f}\n"
            report += f"├─ Title diversity: {title_quality.get('title_diversity', 0):.4f}\n"
            report += f"└─ Average title length: {title_quality.get('average_length', 0):.1f} characters\n\n"
        
        # Comprehensive score detailed display (unified weight system)
        comprehensive_scores = eval_results.get('comprehensive_scores', {})
        if comprehensive_scores:
            report += "Comprehensive Score Details (Weighted Calculation):\n"
            report += f"├─ ROUGE score (weight 35%): {comprehensive_scores.get('rouge_score', 0):.4f}\n"
            report += f"├─ LLM quality score (weight 30%): {comprehensive_scores.get('llm_quality_score', 0):.4f}\n"
            report += f"├─ LLM personalization score (weight 25%): {comprehensive_scores.get('llm_personalization_score', 0):.4f}\n"
            report += f"├─ Title quality score (weight 10%): {comprehensive_scores.get('title_quality_score', 0):.4f}\n"
            report += f"└─ Final comprehensive score: {comprehensive_scores.get('final_comprehensive_score', 0):.4f}\n\n"
        else:
            overall_score = eval_results.get('overall_score', 0)
            if isinstance(overall_score, dict):
                final_score = overall_score.get('final_comprehensive_score', 0)
            else:
                final_score = overall_score
            report += f"Comprehensive score: {final_score:.4f}\n\n"
        
        # Scoring explanation
        report += "Scoring Explanation:\n"
        report += "• ROUGE score (35%): Measures lexical overlap between generated titles and reference titles\n"
        report += "• LLM quality score (30%): Large model evaluation from accuracy, attractiveness, clarity dimensions\n"
        report += "• LLM personalization score (25%): Large model evaluation of title quality from personalization perspective\n"
        report += "• Title quality score (10%): Rule-based evaluation based on length reasonableness, diversity, etc.\n"
        report += "• Final score: Weighted average of all scores (range 0-1)\n\n"
        
        # Rating
        final_score = 0.0
        if comprehensive_scores:
            final_score = comprehensive_scores.get('final_comprehensive_score', 0)
        else:
            overall_score = eval_results.get('overall_score', 0)
            if isinstance(overall_score, dict):
                final_score = overall_score.get('final_comprehensive_score', 0)
            else:
                final_score = overall_score
        
        report += "Note: This evaluation integrates ROUGE automatic evaluation and LLM intelligent evaluation, comprehensively reflecting title generation quality\n"
        
        # Save report
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
            self.logger.info(f"Evaluation report saved to: {save_path}")
        
        return report
    
    def save_detailed_results(self, save_path: str):
        """Save detailed evaluation results to file"""
        
        if not hasattr(self, 'evaluation_results') or not self.evaluation_results:
            self.logger.warning("No evaluation results to save")
            return False
        
        try:
            # Create output directory
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Prepare save data
            save_data = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'evaluation_results': self.evaluation_results
            }
            
            # Save to JSON file
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2, default=str)
            
            self.logger.info(f"Detailed evaluation results saved to: {save_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save detailed evaluation results: {str(e)}")
            return False
    
    def create_comparison_chart(self, save_path: Optional[str] = None):
        """Create evaluation results comparison chart"""
        
        if not hasattr(self, 'evaluation_results') or not self.evaluation_results:
            self.logger.warning("No evaluation results available for chart generation")
            return False
        
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Set font for Chinese characters
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            eval_results = self.evaluation_results
            
            # Create 2x2 subplots
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Personalized News Headline Generation - Evaluation Results', fontsize=16)
            
            # 1. ROUGE scores
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
            
            # 2. LLM evaluation (quality + personalization)
            llm_eval = eval_results.get('llm_evaluation', {})
            llm_personalization = eval_results.get('llm_personalization', {})
            if llm_eval or llm_personalization:
                llm_labels = ['Quality Score', 'Personalization Score']
                llm_values = [
                    llm_eval.get('llm_quality_score', 0),
                    llm_personalization.get('llm_overall_personalization', 0)
                ]
                
                axes[0, 1].bar(llm_labels, llm_values, color='lightcoral', alpha=0.8)
                axes[0, 1].set_title('LLM Evaluation')
                axes[0, 1].set_ylabel('Score')
                axes[0, 1].set_ylim(0, 1)
                for i, v in enumerate(llm_values):
                    axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
            else:
                axes[0, 1].text(0.5, 0.5, 'LLM Evaluation Data\nNot Available', ha='center', va='center', 
                               transform=axes[0, 1].transAxes, fontsize=12)
                axes[0, 1].set_title('LLM Evaluation')
            
            # 3. Title quality
            title_quality = eval_results.get('title_quality', {})
            if title_quality:
                quality_labels = ['Length\nReasonableness', 'Title\nDiversity']
                quality_values = [
                    title_quality.get('length_reasonableness', 0),
                    title_quality.get('title_diversity', 0)
                ]
                
                axes[1, 0].bar(quality_labels, quality_values, color='lightgreen', alpha=0.8)
                axes[1, 0].set_title('Title Quality')
                axes[1, 0].set_ylabel('Score')
                axes[1, 0].set_ylim(0, 1)
                for i, v in enumerate(quality_values):
                    axes[1, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
            else:
                axes[1, 0].text(0.5, 0.5, 'Title Quality Data\nNot Available', ha='center', va='center', 
                               transform=axes[1, 0].transAxes, fontsize=12)
                axes[1, 0].set_title('Title Quality')
            
            # 4. Comprehensive score pie chart (fixed negative value issue)
            comprehensive_scores = eval_results.get('comprehensive_scores', {})
            if comprehensive_scores:
                overall_score = comprehensive_scores.get('final_comprehensive_score', 0)
            else:
                overall_score = eval_results.get('overall_score', 0)
                if isinstance(overall_score, dict):
                    overall_score = overall_score.get('final_comprehensive_score', 0)
            
            # Ensure score is in 0-1 range
            overall_score = max(0.0, min(1.0, overall_score))
            remaining_score = 1.0 - overall_score
            
            # Ensure both values are not negative
            if overall_score < 0 or remaining_score < 0:
                overall_score = 0.5
                remaining_score = 0.5
            
            colors = ['gold', 'lightgray']
            wedge_sizes = [overall_score, remaining_score]
            
            axes[1, 1].pie(wedge_sizes, 
                           labels=['Achieved', 'To Improve'], 
                           colors=colors,
                           autopct='%1.1f%%',
                           startangle=90)
            axes[1, 1].set_title(f'Comprehensive Score: {overall_score:.3f}')
            
            plt.tight_layout()
            
            # Save chart
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Evaluation chart saved to: {save_path}")
            
            plt.close()
            return True
            
        except ImportError as e:
            self.logger.warning(f"Cannot generate chart, matplotlib missing: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"Failed to generate evaluation chart: {str(e)}")
            return False

if __name__ == "__main__":
    # Test evaluator
    evaluator = Evaluator()
    
    # Mock data
    test_results = {
        'generated_titles': [
            'Tech Giants Launch AI Products',
            'Smartphone Performance Major Upgrade',
            'New Technology Leads Industry Change'
        ],
        'reference_titles': [
            'Tech Companies Release New Products',
            'Phone Manufacturers Launch Flagship',
            'Technology Innovation Drives Development'
        ],
        'user_interests': [
            {'primary_interest': 'Technology', 'categories': ['AI', 'Mobile']},
            {'primary_interest': 'Technology', 'categories': ['Hardware']},
            {'primary_interest': 'Business', 'categories': ['Innovation']}
        ],
        'news_categories': ['Technology', 'Technology', 'Business']
    }
    
    # Execute evaluation
    results = evaluator.comprehensive_evaluation(test_results)
    
    # Generate report
    report = evaluator.generate_evaluation_report(results)
    print(report)
    
    # Save results
    evaluator.save_detailed_results('./test_evaluation_results.json') 