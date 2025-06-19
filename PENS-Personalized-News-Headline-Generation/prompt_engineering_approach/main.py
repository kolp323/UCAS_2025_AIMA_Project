"""
主程序 - 基于提示词工程的个性化新闻标题生成
"""

import os
import json
import logging
import argparse
from datetime import datetime
from tqdm import tqdm
from typing import List, Dict, Any

from config import ensure_directories, DATA_CONFIG, PROMPT_CONFIG, API_CONFIG, DATA_PATHS
from data_processor import DataProcessor
from llm_client import LLMClient
from prompt_generator import PromptGenerator
from evaluator import Evaluator

class PersonalizedTitleGenerator:
    """个性化标题生成器主类"""
    
    def __init__(self, config_override: Dict = None):
        self.logger = self._setup_logger()
        
        # 应用配置覆盖
        if config_override:
            self.config = config_override
        else:
            self.config = {
                'data': DATA_CONFIG,
                'prompt': PROMPT_CONFIG,
                'api': API_CONFIG,
                'paths': DATA_PATHS
            }
        
        # 初始化组件
        self.data_processor = DataProcessor()
        self.llm_client = LLMClient()
        self.prompt_generator = PromptGenerator()
        self.evaluator = Evaluator()
        
        # 结果存储
        self.results = {
            'generated_titles': [],
            'reference_titles': [],
            'user_interests': [],
            'news_categories': [],
            'generation_metadata': []
        }
        
        ensure_directories()
    
    def _setup_logger(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def load_or_process_data(self, force_reprocess: bool = False) -> bool:
        """加载或处理数据"""
        
        processed_data_path = os.path.join(self.config['paths']['processed_data_dir'], 'test_samples.json')
        
        if not force_reprocess and os.path.exists(processed_data_path):
            self.logger.info("检测到已处理的数据，直接加载...")
            success = self.data_processor.load_processed_data()
            if success:
                return True
            else:
                self.logger.warning("加载已处理数据失败，重新处理...")
        
        self.logger.info("开始数据预处理...")
        success = self.data_processor.process_all_from_tsv()
        
        if not success:
            self.logger.error("数据预处理失败")
            return False
        
        self.logger.info(f"数据预处理完成，共 {len(self.data_processor.test_samples)} 个样本")
        return True
    
    def generate_titles_single(self, samples: List[Dict[str, Any]], 
                              start_idx: int = 0, end_idx: int = None) -> List[str]:
        """单个样本逐一生成标题"""
        
        if end_idx is None:
            end_idx = len(samples)
        
        generated_titles = []
        
        self.logger.info(f"开始单个生成模式，处理样本 {start_idx} 到 {end_idx}")
        
        for i, sample in enumerate(tqdm(samples[start_idx:end_idx], desc="生成标题")):
            try:
                # 生成个性化提示词
                system_prompt, user_prompt = self.prompt_generator.generate_single_prompt(
                    sample, 
                    style='enhanced'
                )
                
                # 调用LLM生成标题
                response = self.llm_client.generate_personalized_title(
                    sample, 
                    system_prompt, 
                    user_prompt
                )
                
                if response:
                    # 解析多个标题选项，选择第一个
                    title_options = self.llm_client.parse_multiple_titles(response)
                    if title_options:
                        generated_title = title_options[0]
                    else:
                        # 如果解析失败，使用原始响应的第一行
                        generated_title = response.split('\n')[0].strip()
                        generated_title = generated_title.replace('标题:', '').replace(':', '').strip()
                    
                    generated_titles.append(generated_title)
                    
                    # 保存元数据
                    self.results['generation_metadata'].append({
                        'sample_idx': start_idx + i,
                        'prompt_style': 'enhanced',
                        'response_raw': response,
                        'title_options': title_options if 'title_options' in locals() else [generated_title],
                        'selected_title': generated_title
                    })
                else:
                    generated_titles.append("生成失败")
                    self.results['generation_metadata'].append({
                        'sample_idx': start_idx + i,
                        'error': 'API调用失败'
                    })
                
                # 检查API使用情况
                usage_stats = self.llm_client.get_usage_stats()
                if usage_stats['remaining_tokens'] <= 1000:
                    self.logger.warning("API token即将用尽，停止生成")
                    break
                    
            except Exception as e:
                self.logger.error(f"处理样本 {start_idx + i} 时出错: {str(e)}")
                generated_titles.append("处理异常")
                self.results['generation_metadata'].append({
                    'sample_idx': start_idx + i,
                    'error': str(e)
                })
        
        return generated_titles
    
    def generate_titles_batch(self, samples: List[Dict[str, Any]], 
                             batch_size: int = None) -> List[str]:
        """批量生成标题"""
        
        if batch_size is None:
            batch_size = self.config['data']['batch_size']
        
        generated_titles = []
        
        self.logger.info(f"开始批量生成模式，批次大小: {batch_size}")
        
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i+batch_size]
            
            try:
                self.logger.info(f"处理批次 {i//batch_size + 1}, 样本 {i} 到 {i+len(batch)}")
                
                # 生成批量提示词
                system_prompt, batch_prompt = self.prompt_generator.generate_batch_prompt(batch)
                
                # 调用LLM批量生成
                batch_titles = self.llm_client.generate_batch_titles(
                    batch,
                    system_prompt,
                    batch_prompt
                )
                
                # 处理批量结果
                for j, title in enumerate(batch_titles):
                    if title:
                        generated_titles.append(title)
                    else:
                        generated_titles.append("批量生成失败")
                    
                    self.results['generation_metadata'].append({
                        'sample_idx': i + j,
                        'batch_id': i//batch_size,
                        'title': title or "批量生成失败"
                    })
                
                # 检查API使用情况
                usage_stats = self.llm_client.get_usage_stats()
                if usage_stats['remaining_tokens'] <= 2000:
                    self.logger.warning("API token即将用尽，停止生成")
                    break
                    
            except Exception as e:
                self.logger.error(f"处理批次 {i//batch_size + 1} 时出错: {str(e)}")
                # 为该批次的所有样本添加错误标记
                for j in range(len(batch)):
                    generated_titles.append("批次处理异常")
                    self.results['generation_metadata'].append({
                        'sample_idx': i + j,
                        'error': str(e)
                    })
        
        return generated_titles
    
    def run_generation(self, mode: str = 'single', max_samples: int = None, 
                      force_reprocess: bool = False) -> bool:
        """运行标题生成流程"""
        
        self.logger.info("=" * 60)
        self.logger.info("开始个性化新闻标题生成")
        self.logger.info("=" * 60)
        
        # 1. 数据准备
        self.logger.info("步骤 1: 数据准备")
        if not self.load_or_process_data(force_reprocess):
            return False
        
        # 限制样本数量
        samples = self.data_processor.test_samples
        if max_samples:
            samples = samples[:max_samples]
            self.logger.info(f"限制处理样本数为: {max_samples}")
        
        # 2. 标题生成
        self.logger.info(f"步骤 2: 标题生成 (模式: {mode})")
        
        if mode == 'single':
            generated_titles = self.generate_titles_single(samples)
        elif mode == 'batch':
            generated_titles = self.generate_titles_batch(samples)
        else:
            self.logger.error(f"不支持的生成模式: {mode}")
            return False
        
        # 3. 整理结果
        self.logger.info("步骤 3: 整理结果")
        self.results['generated_titles'] = generated_titles
        self.results['reference_titles'] = [s['reference_title'] for s in samples[:len(generated_titles)]]
        self.results['user_interests'] = [s['user_interests'] for s in samples[:len(generated_titles)]]
        self.results['news_categories'] = [s['category'] for s in samples[:len(generated_titles)]]
        # 添加LLM质量评估所需的数据
        self.results['original_titles'] = [s['original_title'] for s in samples[:len(generated_titles)]]
        self.results['news_contents'] = [s['news_body'] for s in samples[:len(generated_titles)]]
        self.results['user_histories'] = [s['user_history'] for s in samples[:len(generated_titles)]]
        
        # 4. 保存结果
        self.save_results()
        
        # 5. 打印API使用统计
        usage_stats = self.llm_client.get_usage_stats()
        self.logger.info("API使用统计:")
        self.logger.info(f"  总请求数: {usage_stats['total_requests']}")
        self.logger.info(f"  失败请求数: {usage_stats['failed_requests']}")
        self.logger.info(f"  成功率: {usage_stats['success_rate']:.2%}")
        self.logger.info(f"  总token使用: {usage_stats['total_tokens_used']}")
        self.logger.info(f"  剩余token: {usage_stats['remaining_tokens']}")
        
        self.logger.info(f"标题生成完成，共生成 {len(generated_titles)} 个标题")
        return True
    
    def save_results(self):
        """保存生成结果"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存详细结果
        results_path = os.path.join(
            self.config['paths']['generated_titles_dir'], 
            f'generation_results_{timestamp}.json'
        )
        
        try:
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"生成结果已保存到: {results_path}")
            
            # 保存简化的标题对比文件
            comparison_path = os.path.join(
                self.config['paths']['generated_titles_dir'],
                f'title_comparison_{timestamp}.txt'
            )
            
            with open(comparison_path, 'w', encoding='utf-8') as f:
                f.write("个性化新闻标题生成结果对比\n")
                f.write("=" * 60 + "\n\n")
                
                for i, (gen, ref) in enumerate(zip(
                    self.results['generated_titles'], 
                    self.results['reference_titles']
                )):
                    f.write(f"样本 {i+1}:\n")
                    f.write(f"生成标题: {gen}\n")
                    f.write(f"参考标题: {ref}\n")
                    f.write("-" * 40 + "\n")
            
            self.logger.info(f"标题对比文件已保存到: {comparison_path}")
            
        except Exception as e:
            self.logger.error(f"保存结果失败: {str(e)}")
    
    def run_evaluation(self):
        """运行评估流程"""
        
        if not self.results['generated_titles']:
            self.logger.error("没有生成结果可供评估")
            return False
        
        self.logger.info("开始评估生成结果...")
        
        # 执行综合评估
        evaluation_results = self.evaluator.comprehensive_evaluation(self.results)
        
        # 生成评估报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(
            self.config['paths']['evaluation_results_dir'],
            f'evaluation_report_{timestamp}.txt'
        )
        
        report = self.evaluator.generate_evaluation_report(evaluation_results, report_path)
        print("\n" + report)
        
        # 保存详细评估结果
        detailed_results_path = os.path.join(
            self.config['paths']['evaluation_results_dir'],
            f'detailed_evaluation_{timestamp}.json'
        )
        self.evaluator.save_detailed_results(detailed_results_path)
        
        # 生成评估图表
        try:
            chart_path = os.path.join(
                self.config['paths']['evaluation_results_dir'],
                f'evaluation_chart_{timestamp}.png'
            )
            self.evaluator.create_comparison_chart(save_path=chart_path)
        except Exception as e:
            self.logger.warning(f"生成评估图表失败: {e}")
        
        return True

def main():
    """主函数"""
    
    parser = argparse.ArgumentParser(description='基于提示词工程的个性化新闻标题生成')
    parser.add_argument('--mode', choices=['single', 'batch'], default='single',
                       help='生成模式: single(逐个生成) 或 batch(批量生成)')
    parser.add_argument('--max-samples', type=int, default=50,
                       help='最大处理样本数量')
    parser.add_argument('--force-reprocess', action='store_true',
                       help='强制重新处理数据')
    parser.add_argument('--skip-generation', action='store_true',
                       help='跳过生成，仅运行评估')
    parser.add_argument('--skip-evaluation', action='store_true',
                       help='跳过评估，仅运行生成')
    
    args = parser.parse_args()
    
    # 创建生成器实例
    generator = PersonalizedTitleGenerator()
    
    success = True
    
    # 运行生成流程
    if not args.skip_generation:
        success = generator.run_generation(
            mode=args.mode,
            max_samples=args.max_samples,
            force_reprocess=args.force_reprocess
        )
    
    # 运行评估流程
    if success and not args.skip_evaluation:
        generator.run_evaluation()
    
    if success:
        print("\n🎉 程序执行完成！")
        print(f"📁 查看结果文件夹: {DATA_PATHS['output_dir']}")
    else:
        print("\n❌ 程序执行失败，请检查日志信息")

if __name__ == "__main__":
    main() 