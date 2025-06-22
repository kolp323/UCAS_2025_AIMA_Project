# Prompt Engineering-based Personalized News Headline Generation System

## Project Overview

This project adopts a prompt engineering technical approach, implementing personalized news headline generation through calling large language model APIs, serving as an alternative implementation scheme for the PENS project. This system supports multi-model comparison, improved personalization evaluation, and LLM quality assessment.

## Main Features

- **Secure API Configuration Management**: Sensitive information stored independently, secure and controllable
- **Multi-model Support**: Supports dynamic model switching and performance comparison
- **Enhanced Personalization Evaluation**: Scientific evaluation of personalization effects across 4 dimensions
- **LLM Quality Assessment**: Uses reasoning models to objectively evaluate headline quality
- **Comprehensive Scoring System**: Multi-metric weighted evaluation, comprehensively reflecting system performance

## Technical Architecture

### Core Modules

1. **API Configuration Module**
   - `api_config.py` - Sensitive information configuration (added to .gitignore)
   - `api_config_template.py` - Configuration template
   - Supports multi-model configuration and dynamic switching

2. **Data Processing Module** (`data_processor.py`)
   - Extracts user historical click sequences from existing PENS data
   - Analyzes user interest tags and historical preferences
   - Intelligent data preprocessing and cleaning

3. **Prompt Engineering Module** (`prompt_generator.py`)
   - Adaptive prompt design
   - Supports different prompt strategies for reasoning models and chat models
   - Optimizes token usage, controls API call costs

4. **LLM Client Module** (`llm_client.py`)
   - Supports multi-model dynamic switching
   - Intelligent content extraction and error handling
   - API quota management and rate limiting

5. **Evaluation Module** (`evaluator.py`)
   - 4-dimensional personalization evaluation
   - LLM quality evaluation functionality
   - Scientific comprehensive scoring system

6. **Main Program Module** (`main.py`)
   - Complete generation process
   - Batch processing and result management

## File Structure

```
prompt_engineering_approach/
├── README.md                           # Project description
├── requirements.txt                    # Dependencies
├── api_config_template.py              # API configuration template
├── config.py                          # Main configuration file
├── data_processor.py                  # Data processor
├── prompt_generator.py                # Prompt generator
├── llm_client.py                      # LLM API client
├── evaluator.py                       # Evaluator
├── main.py                            # Main program
├── prompt_engineering_pipeline.ipynb  # Complete process demonstration
└── outputs/                           # Output directory
    ├── processed_data/                # Preprocessed data
    ├── generated_titles/              # Generated headlines
    └── evaluation_results/            # Evaluation results
```

## Quick Start

### 1. Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
# Copy configuration template
cp api_config_template.py api_config.py

# Edit configuration file, replace YOUR_API_KEY_HERE with your actual API key
```

### 3. Run System

**Method 1: Using Jupyter Notebook (Recommended)**
```bash
jupyter notebook prompt_engineering_pipeline.ipynb
```

**Method 2: Using Python Scripts**
```bash
# Run main program
python main.py
```

## Supported Models

Currently supports the following models (configurable in `api_config.py`):

- **deepseek-chat-v3-0324** (default) - Chat model, suitable for headline generation
- **deepseek-r1-0528** - Reasoning model, used for quality evaluation

Users can add more models in the configuration file as needed.

## Evaluation System

### Enhanced Personalization Evaluation

1. **Interest Matching** (40% weight) - Degree of headline matching with user interests
2. **Category Relevance** (20% weight) - Relevance of headline to news category
3. **Interest Consistency** (25% weight) - Consistency between user interests and news category
4. **Historical Relevance** (15% weight) - Considers user historical reading preferences

### LLM Quality Evaluation

Uses reasoning models for 5-dimensional evaluation:
- Accuracy, Attractiveness, Clarity, Reasonableness, Innovation

### Comprehensive Scoring

- ROUGE Score: 25%
- Personalization Effect: 35%
- Headline Quality: 20%
- LLM Evaluation: 20%

## Performance

### Evaluation Metric Improvements

| Evaluation Metric | Original Version | Enhanced Version | Improvement Description |
|---------|--------|--------|----------|
| Personalization Matching | ~0.01 | ~0.35+ | Scientific evaluation algorithm |
| Category Relevance | ~0.00 | ~0.42+ | Added semantic matching |
| Headline Quality Evaluation | Rule-based | LLM evaluation | More objective and accurate |
| Comprehensive Score | 0.31 | 0.45+ | Multi-dimensional evaluation |

### System Advantages

- **Security**: Secure API key storage, no accidental leakage
- **Efficiency**: Intelligent batch processing, controllable costs
- **Accuracy**: Improved evaluation methods more scientific and objective
- **Flexibility**: Multi-model support, convenient for horizontal comparison
- **Comprehensiveness**: Multi-dimensional evaluation, comprehensive reflection of system performance

## Usage Recommendations

1. **First Use**: Run complete Jupyter Notebook process
2. **Daily Development**: Use main.py for batch processing
3. **Model Comparison**: Add new models in configuration file for comparison
4. **Effect Optimization**: Adjust prompt design based on evaluation results

## Technical Details

### API Configuration Security

- Sensitive information stored in independent files
- Automatically added to .gitignore, preventing accidental commits
- Supports environment variable configuration

### Multi-model Support

```python
# Dynamic model switching
from llm_client import LLMClient
client = LLMClient(model_name="deepseek-r1-0528")
client.switch_model("deepseek-chat-v3-0324")
```

### Enhanced Evaluation

```python
# Using evaluator
from evaluator import Evaluator
evaluator = Evaluator(use_llm_evaluation=True)
results = evaluator.comprehensive_evaluation(data)
```

## Future Plans

- [ ] Support more large model APIs
- [ ] Implement online learning and feedback optimization
- [ ] Add multi-language support
- [ ] Develop Web interface
- [ ] Integrate more evaluation metrics

## Update Log

### v2.0.0 (Current Version)
- Refactored API configuration management, improved security
- Added multi-model support and dynamic switching
- Improved personalization evaluation algorithm (4-dimensional evaluation)
- Introduced LLM quality evaluation functionality
- Upgraded comprehensive scoring system
- Optimized Jupyter Notebook process

### v1.0.0 (Original Version)
- Basic personalized headline generation functionality
- ROUGE evaluation support
- Simple personalization effect evaluation

## Contribution Guidelines

Welcome to submit Issues and Pull Requests to improve this project!

## License

This project uses the MIT license. 