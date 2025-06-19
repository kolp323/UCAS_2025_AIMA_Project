"""
API配置模板文件
复制此文件为 api_config.py 并填入您的API密钥
"""

# API基础配置
API_BASE_CONFIG = {
    'base_url': 'https://api.chavapa.com/v1',
    'api_key': 'YOUR_API_KEY_HERE',  # 请替换为您的API密钥
    'max_retries': 3,
    'timeout': 60
}

# 支持的模型配置
MODELS_CONFIG = {
    'deepseek-r1-0528': {
        'name': 'deepseek-r1-0528',
        'type': 'reasoning',
        'max_tokens': 500,
        'temperature': 0.7,
        'description': '推理模型，支持复杂逻辑推理'
    },
    'deepseek-chat-v3-0324': {
        'name': 'deepseek-chat-v3-0324', 
        'type': 'chat',
        'max_tokens': 200,
        'temperature': 0.7,
        'description': '聊天模型，适合对话生成'
    }
}

# 默认模型
DEFAULT_MODEL = 'deepseek-chat-v3-0324'

EVALUATION_MODEL = 'deepseek-chat-v3-0324'

def get_model_config(model_name: str) -> dict:
    """获取指定模型的配置"""
    return MODELS_CONFIG.get(model_name, MODELS_CONFIG[DEFAULT_MODEL])

def get_available_models() -> list:
    """获取所有可用模型列表"""
    return list(MODELS_CONFIG.keys())

def is_reasoning_model(model_name: str) -> bool:
    """判断是否为推理模型"""
    config = get_model_config(model_name)
    return config.get('type') == 'reasoning' 