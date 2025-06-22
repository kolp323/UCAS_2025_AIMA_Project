"""
Configuration file - Personalized News Headline Generation Based on Prompt Engineering
"""

import os

# Import API configuration
try:
    from api_config import (
        API_BASE_CONFIG, 
        MODELS_CONFIG, 
        DEFAULT_MODEL, 
        EVALUATION_MODEL,
        get_model_config, 
        get_available_models, 
        is_reasoning_model as _is_reasoning_model
    )
except ImportError:
    print("Warning: api_config.py file not found, please copy from api_config_template.py and configure your API key")
    from api_config_template import (
        API_BASE_CONFIG, 
        MODELS_CONFIG, 
        DEFAULT_MODEL, 
        EVALUATION_MODEL,
        get_model_config, 
        get_available_models, 
        is_reasoning_model as _is_reasoning_model
    )

# API configuration (backward compatibility)
API_CONFIG = {
    'base_url': API_BASE_CONFIG['base_url'],
    'api_key': API_BASE_CONFIG['api_key'],
    'model': 'deepseek-chat-v3-0324',  # Currently used model
    'max_tokens': 50000,  # Total token limit
    'max_tokens_per_request': 3000,  # Single request token limit
    'max_retries': API_BASE_CONFIG['max_retries'],
    'timeout': API_BASE_CONFIG['timeout'],
    'temperature': 0.7
}

# Data path configuration
DATA_PATHS = {
    'base_data_dir': '../data2',  # Preprocessed PENS data directory (containing pkl files)
    'raw_data_dir': '../data',  # Original PENS data directory
    'output_dir': './outputs',
    'processed_data_dir': './outputs/processed_data',
    'generated_titles_dir': './outputs/generated_titles',
    'evaluation_results_dir': './outputs/evaluation_results'
}

# Data processing configuration
DATA_CONFIG = {
    'max_user_history': 20,  # Maximum user history records
    'max_news_content_length': 500,  # Maximum news content length
    'min_title_length': 5,  # Minimum title length
    'max_title_length': 50,  # Maximum title length
    'batch_size': 10,  # Batch processing size
    'test_samples': 200  # Number of test samples
}

# Prompt configuration
PROMPT_CONFIG = {
    'system_prompts': {
        'reasoning_model': """You are an AI expert at creating personalized news headlines. Your task is to analyze user preferences and generate engaging, personalized English headlines that match their interests.

Key requirements:
- Generate ONLY English headlines (no Chinese)
- Headlines should be 8-20 words long
- Make headlines personally relevant to the user's interests
- Maintain accuracy to the original news content
- Use engaging, attention-grabbing language
- Return ONLY the final headline text, no explanations""",

        'chat_model': """Create a personalized English news headline based on the user's interests and browsing history.

Requirements:
- English only
- 8-20 words
- Engaging and relevant to user interests
- Accurate to the news content
- Return only the headline, no other text"""
    },
    
    'user_prompts': {
        'reasoning_model': {
            'focused': """Based on this user's interests and reading history, create a personalized English headline for the news below.

User's browsing history (recent articles they read):
{user_history}

User's interests: {user_interests}

Original news:
Title: {original_title}
Content: {news_content}
Category: {news_category}

Generate an engaging 8-20 word English headline that would appeal to this specific user. Consider their interests and reading patterns to make it personally relevant.""",

            'enhanced': """Analyze this user's profile and create a compelling personalized headline.

User Profile:
- Reading History: {user_history}
- Primary Interests: {user_interests}

News to personalize:
- Original: {original_title}
- Content: {news_content}
- Category: {news_category}

Create a personalized English headline (8-20 words) that:
1. Matches their interests
2. Uses engaging language
3. Maintains accuracy
4. Would catch their attention based on their reading history""",

            'creative': """Transform this news into a personalized headline for a specific user.

User Context:
{user_history}

Interests: {user_interests}

News:
{original_title}
{news_content}
Category: {news_category}

Generate a creative, personalized English headline (8-20 words) that speaks directly to this user's interests and reading preferences."""
        },

        'chat_model': {
            'focused': """Create a personalized headline for this user:

Interests: {user_interests}
Recent reads: {user_history}

News: {original_title}
Content: {news_content}

Generate an 8-20 word English headline that appeals to their interests.""",

            'enhanced': """User profile:
- Interests: {user_interests}  
- Recent articles: {user_history}

News: {original_title}
Content: {news_content}

Create an engaging 8-20 word English headline personalized for this user.""",

            'creative': """Personalize this headline for a user interested in: {user_interests}

Their recent reads: {user_history}

Original: {original_title}
Content: {news_content}

Return an engaging 8-20 word English headline."""
        }
    }
}

# Evaluation configuration
EVALUATION_CONFIG = {
    'rouge_metrics': ['rouge-1', 'rouge-2', 'rouge-l'],
    'use_stemmer': True,
    'alpha': 0.5,  # Weight parameter for F-score
    'evaluation_samples': 100  # Number of evaluation samples
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': './outputs/logs/prompt_engineering.log'
}

# Model configuration
MODEL_CONFIG = {
    'reasoning_models': [
        'deepseek-r1-0528',
        'deepseek-r1',
        'qwen-plus-r1', 
        'claude-thinking'
    ],
    'chat_models': [
        'deepseek-chat-v3-0324',
        'deepseek-chat',
        'gpt-4',
        'claude-3.5-sonnet'
    ]
}

# Evaluation model configuration - modified to non-reasoning model
EVALUATION_MODEL = 'deepseek-chat-v3-0324'  # Changed to chat model to avoid complex parsing of reasoning models

def is_reasoning_model(model_name: str) -> bool:
    """Check if it is a reasoning model"""
    return model_name in MODEL_CONFIG['reasoning_models']

def get_optimal_max_tokens(model_name: str) -> int:
    """Get optimal max_tokens setting based on model type"""
    config = get_model_config(model_name)
    return config.get('max_tokens', 200)

def get_system_prompt(model_name: str) -> str:
    """Get system prompt based on model type"""
    if is_reasoning_model(model_name):
        return PROMPT_CONFIG['system_prompts']['reasoning_model']
    else:
        return PROMPT_CONFIG['system_prompts']['chat_model']

def get_user_prompt(model_name: str, style: str = 'focused') -> str:
    """Get user prompt based on model type and style"""
    if is_reasoning_model(model_name):
        return PROMPT_CONFIG['user_prompts']['reasoning_model'].get(style, 
               PROMPT_CONFIG['user_prompts']['reasoning_model']['focused'])
    else:
        return PROMPT_CONFIG['user_prompts']['chat_model'].get(style,
               PROMPT_CONFIG['user_prompts']['chat_model']['focused'])

def set_current_model(model_name: str):
    """Set the currently used model"""
    if model_name in get_available_models():
        API_CONFIG['model'] = model_name
        return True
    else:
        print(f"Model {model_name} is not in the available model list: {get_available_models()}")
        return False

# Create necessary directories
def ensure_directories():
    """Ensure all necessary directories exist"""
    dirs_to_create = [
        DATA_PATHS['output_dir'],
        DATA_PATHS['processed_data_dir'],
        DATA_PATHS['generated_titles_dir'],
        DATA_PATHS['evaluation_results_dir'],
        os.path.dirname(LOGGING_CONFIG['file'])
    ]
    
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)

if __name__ == "__main__":
    ensure_directories()
    print("Configuration loaded successfully, directory structure created") 