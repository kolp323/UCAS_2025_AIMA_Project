"""
API configuration template file
Copy this file as api_config.py and fill in your API key
"""

# Basic API configuration
API_BASE_CONFIG = {
    'base_url': 'https://api.chavapa.com/v1',
    'api_key': 'YOUR_API_KEY_HERE',  # Please replace with your API key
    'max_retries': 3,
    'timeout': 60
}

# Supported model configurations
MODELS_CONFIG = {
    'deepseek-r1-0528': {
        'name': 'deepseek-r1-0528',
        'type': 'reasoning',
        'max_tokens': 500,
        'temperature': 0.7,
        'description': 'Reasoning model, supports complex logical reasoning'
    },
    'deepseek-chat-v3-0324': {
        'name': 'deepseek-chat-v3-0324', 
        'type': 'chat',
        'max_tokens': 200,
        'temperature': 0.7,
        'description': 'Chat model, suitable for dialogue generation'
    }
}

# Default model
DEFAULT_MODEL = 'deepseek-chat-v3-0324'

EVALUATION_MODEL = 'deepseek-chat-v3-0324'

def get_model_config(model_name: str) -> dict:
    """Get configuration for specified model"""
    return MODELS_CONFIG.get(model_name, MODELS_CONFIG[DEFAULT_MODEL])

def get_available_models() -> list:
    """Get list of all available models"""
    return list(MODELS_CONFIG.keys())

def is_reasoning_model(model_name: str) -> bool:
    """Check if it is a reasoning model"""
    config = get_model_config(model_name)
    return config.get('type') == 'reasoning' 