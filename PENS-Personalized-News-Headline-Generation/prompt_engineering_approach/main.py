"""
Main program - Personalized News Headline Generation Based on Prompt Engineering
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
    """Main class for personalized title generator"""
    
    def __init__(self, config_override: Dict = None):
        self.logger = self._setup_logger()
        
        # Apply configuration override
        if config_override:
            self.config = config_override
        else:
            self.config = {
                'data': DATA_CONFIG,
                'prompt': PROMPT_CONFIG,
                'api': API_CONFIG,
                'paths': DATA_PATHS
            }
        
        # Initialize components
        self.data_processor = DataProcessor()
        self.llm_client = LLMClient()
        self.prompt_generator = PromptGenerator()
        self.evaluator = Evaluator()
        
        # Results storage
        self.results = {
            'generated_titles': [],
            'reference_titles': [],
            'user_interests': [],
            'news_categories': [],
            'generation_metadata': []
        }
        
        ensure_directories()
    
    def _setup_logger(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def load_or_process_data(self, force_reprocess: bool = False) -> bool:
        """Load or process data"""
        
        processed_data_path = os.path.join(self.config['paths']['processed_data_dir'], 'test_samples.json')
        
        if not force_reprocess and os.path.exists(processed_data_path):
            self.logger.info("Detected processed data, loading directly...")
            success = self.data_processor.load_processed_data()
            if success:
                return True
            else:
                self.logger.warning("Failed to load processed data, reprocessing...")
        
        self.logger.info("Starting data preprocessing...")
        success = self.data_processor.process_all_from_tsv()
        
        if not success:
            self.logger.error("Data preprocessing failed")
            return False
        
        self.logger.info(f"Data preprocessing completed, total {len(self.data_processor.test_samples)} samples")
        return True
    
    def generate_titles_single(self, samples: List[Dict[str, Any]], 
                              start_idx: int = 0, end_idx: int = None) -> List[str]:
        """Generate titles one by one for individual samples"""
        
        if end_idx is None:
            end_idx = len(samples)
        
        generated_titles = []
        
        self.logger.info(f"Starting single generation mode, processing samples {start_idx} to {end_idx}")
        
        for i, sample in enumerate(tqdm(samples[start_idx:end_idx], desc="Generating titles")):
            try:
                # Generate personalized prompt
                system_prompt, user_prompt = self.prompt_generator.generate_single_prompt(
                    sample, 
                    style='enhanced'
                )
                
                # Call LLM to generate title
                response = self.llm_client.generate_personalized_title(
                    sample, 
                    system_prompt, 
                    user_prompt
                )
                
                if response:
                    # Parse multiple title options, select the first one
                    title_options = self.llm_client.parse_multiple_titles(response)
                    if title_options:
                        generated_title = title_options[0]
                    else:
                        # If parsing fails, use the first line of raw response
                        generated_title = response.split('\n')[0].strip()
                        generated_title = generated_title.replace('Title:', '').replace(':', '').strip()
                    
                    generated_titles.append(generated_title)
                    
                    # Save metadata
                    self.results['generation_metadata'].append({
                        'sample_idx': start_idx + i,
                        'prompt_style': 'enhanced',
                        'response_raw': response,
                        'title_options': title_options if 'title_options' in locals() else [generated_title],
                        'selected_title': generated_title
                    })
                else:
                    generated_titles.append("Generation failed")
                    self.results['generation_metadata'].append({
                        'sample_idx': start_idx + i,
                        'error': 'API call failed'
                    })
                
                # Check API usage status
                usage_stats = self.llm_client.get_usage_stats()
                if usage_stats['remaining_tokens'] <= 1000:
                    self.logger.warning("API tokens almost exhausted, stopping generation")
                    break
                    
            except Exception as e:
                self.logger.error(f"Error processing sample {start_idx + i}: {str(e)}")
                generated_titles.append("Processing exception")
                self.results['generation_metadata'].append({
                    'sample_idx': start_idx + i,
                    'error': str(e)
                })
        
        return generated_titles
    
    def generate_titles_batch(self, samples: List[Dict[str, Any]], 
                             batch_size: int = None) -> List[str]:
        """Generate titles in batch"""
        
        if batch_size is None:
            batch_size = self.config['data']['batch_size']
        
        generated_titles = []
        
        self.logger.info(f"Starting batch generation mode, batch size: {batch_size}")
        
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i+batch_size]
            
            try:
                self.logger.info(f"Processing batch {i//batch_size + 1}, samples {i} to {i+len(batch)}")
                
                # Generate batch prompt
                system_prompt, batch_prompt = self.prompt_generator.generate_batch_prompt(batch)
                
                # Call LLM for batch generation
                batch_titles = self.llm_client.generate_batch_titles(
                    batch,
                    system_prompt,
                    batch_prompt
                )
                
                # Process batch results
                for j, title in enumerate(batch_titles):
                    if title:
                        generated_titles.append(title)
                    else:
                        generated_titles.append("Batch generation failed")
                    
                    self.results['generation_metadata'].append({
                        'sample_idx': i + j,
                        'batch_id': i//batch_size,
                        'title': title or "Batch generation failed"
                    })
                
                # Check API usage status
                usage_stats = self.llm_client.get_usage_stats()
                if usage_stats['remaining_tokens'] <= 2000:
                    self.logger.warning("API tokens almost exhausted, stopping generation")
                    break
                    
            except Exception as e:
                self.logger.error(f"Error processing batch {i//batch_size + 1}: {str(e)}")
                # Add error markers for all samples in this batch
                for j in range(len(batch)):
                    generated_titles.append("Batch processing exception")
                    self.results['generation_metadata'].append({
                        'sample_idx': i + j,
                        'error': str(e)
                    })
        
        return generated_titles
    
    def run_generation(self, mode: str = 'single', max_samples: int = None, 
                      force_reprocess: bool = False) -> bool:
        """Run title generation process"""
        
        self.logger.info("=" * 60)
        self.logger.info("Starting personalized news headline generation")
        self.logger.info("=" * 60)
        
        # 1. Data preparation
        self.logger.info("Step 1: Data preparation")
        if not self.load_or_process_data(force_reprocess):
            return False
        
        # Limit sample count
        samples = self.data_processor.test_samples
        if max_samples:
            samples = samples[:max_samples]
            self.logger.info(f"Limited processing sample count to: {max_samples}")
        
        # 2. Title generation
        self.logger.info(f"Step 2: Title generation (mode: {mode})")
        
        if mode == 'single':
            generated_titles = self.generate_titles_single(samples)
        elif mode == 'batch':
            generated_titles = self.generate_titles_batch(samples)
        else:
            self.logger.error(f"Unsupported generation mode: {mode}")
            return False
        
        # 3. Organize results
        self.logger.info("Step 3: Organize results")
        self.results['generated_titles'] = generated_titles
        self.results['reference_titles'] = [s['reference_title'] for s in samples[:len(generated_titles)]]
        self.results['user_interests'] = [s['user_interests'] for s in samples[:len(generated_titles)]]
        self.results['news_categories'] = [s['category'] for s in samples[:len(generated_titles)]]
        # Add data required for LLM quality evaluation
        self.results['original_titles'] = [s['original_title'] for s in samples[:len(generated_titles)]]
        self.results['news_contents'] = [s['news_body'] for s in samples[:len(generated_titles)]]
        self.results['user_histories'] = [s['user_history'] for s in samples[:len(generated_titles)]]
        
        # 4. Save results
        self.save_results()
        
        # 5. Print API usage statistics
        usage_stats = self.llm_client.get_usage_stats()
        self.logger.info("API usage statistics:")
        self.logger.info(f"  Total requests: {usage_stats['total_requests']}")
        self.logger.info(f"  Failed requests: {usage_stats['failed_requests']}")
        self.logger.info(f"  Success rate: {usage_stats['success_rate']:.2%}")
        self.logger.info(f"  Total tokens used: {usage_stats['total_tokens_used']}")
        self.logger.info(f"  Remaining tokens: {usage_stats['remaining_tokens']}")
        
        self.logger.info(f"Title generation completed, generated {len(generated_titles)} titles")
        return True
    
    def save_results(self):
        """Save generation results"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_path = os.path.join(
            self.config['paths']['generated_titles_dir'], 
            f'generation_results_{timestamp}.json'
        )
        
        try:
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Generation results saved to: {results_path}")
            
            # Save simplified title comparison file
            comparison_path = os.path.join(
                self.config['paths']['generated_titles_dir'],
                f'title_comparison_{timestamp}.txt'
            )
            
            with open(comparison_path, 'w', encoding='utf-8') as f:
                f.write("Personalized News Headline Generation Results Comparison\n")
                f.write("=" * 60 + "\n\n")
                
                for i, (gen, ref) in enumerate(zip(
                    self.results['generated_titles'], 
                    self.results['reference_titles']
                )):
                    f.write(f"Sample {i+1}:\n")
                    f.write(f"Generated title: {gen}\n")
                    f.write(f"Reference title: {ref}\n")
                    f.write("-" * 40 + "\n")
            
            self.logger.info(f"Title comparison file saved to: {comparison_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {str(e)}")
    
    def run_evaluation(self):
        """Run evaluation process"""
        
        if not self.results['generated_titles']:
            self.logger.error("No generation results available for evaluation")
            return False
        
        self.logger.info("Starting evaluation of generation results...")
        
        # Execute comprehensive evaluation
        evaluation_results = self.evaluator.comprehensive_evaluation(self.results)
        
        # Generate evaluation report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(
            self.config['paths']['evaluation_results_dir'],
            f'evaluation_report_{timestamp}.txt'
        )
        
        report = self.evaluator.generate_evaluation_report(evaluation_results, report_path)
        print("\n" + report)
        
        # Save detailed evaluation results
        detailed_results_path = os.path.join(
            self.config['paths']['evaluation_results_dir'],
            f'detailed_evaluation_{timestamp}.json'
        )
        self.evaluator.save_detailed_results(detailed_results_path)
        
        # Generate evaluation charts
        try:
            chart_path = os.path.join(
                self.config['paths']['evaluation_results_dir'],
                f'evaluation_chart_{timestamp}.png'
            )
            self.evaluator.create_comparison_chart(save_path=chart_path)
        except Exception as e:
            self.logger.warning(f"Failed to generate evaluation chart: {e}")
        
        return True

def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description='Personalized News Headline Generation Based on Prompt Engineering')
    parser.add_argument('--mode', choices=['single', 'batch'], default='single',
                       help='Generation mode: single (generate one by one) or batch (batch generation)')
    parser.add_argument('--max-samples', type=int, default=50,
                       help='Maximum number of samples to process')
    parser.add_argument('--force-reprocess', action='store_true',
                       help='Force reprocessing of data')
    parser.add_argument('--skip-generation', action='store_true',
                       help='Skip generation, only run evaluation')
    parser.add_argument('--skip-evaluation', action='store_true',
                       help='Skip evaluation, only run generation')
    
    args = parser.parse_args()
    
    # Create generator instance
    generator = PersonalizedTitleGenerator()
    
    success = True
    
    # Run generation process
    if not args.skip_generation:
        success = generator.run_generation(
            mode=args.mode,
            max_samples=args.max_samples,
            force_reprocess=args.force_reprocess
        )
    
    # Run evaluation process
    if success and not args.skip_evaluation:
        generator.run_evaluation()
    
    if success:
        print("\nProgram execution completed!")
        print(f"Check results folder: {DATA_PATHS['output_dir']}")
    else:
        print("\nProgram execution failed, please check log information")

if __name__ == "__main__":
    main() 