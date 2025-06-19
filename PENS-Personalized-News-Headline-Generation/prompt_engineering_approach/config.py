"""
配置文件 - 基于提示词工程的个性化新闻标题生成
"""

import os

# 导入API配置
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
    print("警告: 未找到api_config.py文件，请从api_config_template.py复制并配置您的API密钥")
    from api_config_template import (
        API_BASE_CONFIG, 
        MODELS_CONFIG, 
        DEFAULT_MODEL, 
        EVALUATION_MODEL,
        get_model_config, 
        get_available_models, 
        is_reasoning_model as _is_reasoning_model
    )

# API配置（向后兼容）
API_CONFIG = {
    'base_url': API_BASE_CONFIG['base_url'],
    'api_key': API_BASE_CONFIG['api_key'],
    'model': 'deepseek-chat-v3-0324',  # 当前使用的模型
    'max_tokens': 50000,  # 总token限制
    'max_tokens_per_request': 3000,  # 单次请求token限制
    'max_retries': API_BASE_CONFIG['max_retries'],
    'timeout': API_BASE_CONFIG['timeout'],
    'temperature': 0.7
}

# 数据路径配置
DATA_PATHS = {
    'base_data_dir': '../data2',  # 预处理后的PENS数据目录（包含pkl文件）
    'raw_data_dir': '../data',  # 原始PENS数据目录
    'output_dir': './outputs',
    'processed_data_dir': './outputs/processed_data',
    'generated_titles_dir': './outputs/generated_titles',
    'evaluation_results_dir': './outputs/evaluation_results'
}

# 数据处理配置
DATA_CONFIG = {
    'max_user_history': 20,  # 最大用户历史记录数
    'max_news_content_length': 500,  # 新闻内容最大长度
    'min_title_length': 5,  # 最小标题长度
    'max_title_length': 50,  # 最大标题长度
    'batch_size': 10,  # 批处理大小
    'test_samples': 200  # 测试样本数量
}

# 提示词配置
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

# 评估配置
EVALUATION_CONFIG = {
    'rouge_metrics': ['rouge-1', 'rouge-2', 'rouge-l'],
    'use_stemmer': True,
    'alpha': 0.5,  # F-score的权重参数
    'evaluation_samples': 100  # 评估样本数量
}

# 日志配置
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': './outputs/logs/prompt_engineering.log'
}

# 模型配置
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

# 评估模型配置 - 修改为非推理模型
EVALUATION_MODEL = 'deepseek-chat-v3-0324'  # 改为聊天模型，避免推理模型的复杂解析

def is_reasoning_model(model_name: str) -> bool:
    """判断是否为推理模型"""
    return model_name in MODEL_CONFIG['reasoning_models']

def get_optimal_max_tokens(model_name: str) -> int:
    """根据模型类型获取最优的max_tokens设置"""
    config = get_model_config(model_name)
    return config.get('max_tokens', 200)

def get_system_prompt(model_name: str) -> str:
    """根据模型类型获取系统提示词"""
    if is_reasoning_model(model_name):
        return PROMPT_CONFIG['system_prompts']['reasoning_model']
    else:
        return PROMPT_CONFIG['system_prompts']['chat_model']

def get_user_prompt(model_name: str, style: str = 'focused') -> str:
    """根据模型类型和风格获取用户提示词"""
    if is_reasoning_model(model_name):
        return PROMPT_CONFIG['user_prompts']['reasoning_model'].get(style, 
               PROMPT_CONFIG['user_prompts']['reasoning_model']['focused'])
    else:
        return PROMPT_CONFIG['user_prompts']['chat_model'].get(style,
               PROMPT_CONFIG['user_prompts']['chat_model']['focused'])

def set_current_model(model_name: str):
    """设置当前使用的模型"""
    if model_name in get_available_models():
        API_CONFIG['model'] = model_name
        return True
    else:
        print(f"模型 {model_name} 不在可用模型列表中: {get_available_models()}")
        return False

# 创建必要的目录
def ensure_directories():
    """确保所有必要的目录存在"""
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
    print("配置加载完成，目录结构已创建") 