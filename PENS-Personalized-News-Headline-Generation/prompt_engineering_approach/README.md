# 基于提示词工程的个性化新闻标题生成系统

## 🚀 项目概述

本项目采用提示词工程的技术路线，通过调用大语言模型API实现个性化新闻标题生成，作为PENS项目的替代实现方案。本系统支持多模型比较、改进的个性化评估和LLM质量评估。

## 🎯 主要特性

- **🔐 安全的API配置管理**：敏感信息独立存储，安全可控
- **🔧 多模型支持**：支持动态模型切换和性能对比
- **📊 增强版个性化评估**：4维度科学评估个性化效果
- **🤖 LLM质量评估**：使用推理模型客观评估标题质量
- **🎯 综合评分体系**：多指标加权评估，全面反映系统性能

## 📁 技术架构

### 核心模块

1. **API配置模块**
   - `api_config.py` - 敏感信息配置（已加入.gitignore）
   - `api_config_template.py` - 配置模板
   - 支持多模型配置和动态切换

2. **数据处理模块** (`data_processor.py`)
   - 从原有PENS数据中提取用户历史点击序列
   - 分析用户兴趣标签和历史偏好
   - 智能数据预处理和清洗

3. **提示词工程模块** (`prompt_generator.py`)
   - 自适应提示词设计
   - 支持推理模型和聊天模型不同的提示词策略
   - 优化token使用，控制API调用成本

4. **LLM客户端模块** (`llm_client.py`)
   - 支持多模型动态切换
   - 智能内容提取和错误处理
   - API配额管理和速率限制

5. **评估模块** (`evaluator.py`)
       - 4维度个性化评估
   - LLM质量评估功能
   - 科学的综合评分体系

6. **主程序模块** (`main.py`)
   - 完整的生成流程
   - 批量处理和结果管理

## 📂 文件结构

```
prompt_engineering_approach/
├── README.md                           # 项目说明
├── requirements.txt                    # 依赖包
├── api_config_template.py              # API配置模板
├── config.py                          # 主配置文件
├── data_processor.py                  # 数据处理器
├── prompt_generator.py                # 提示词生成器
├── llm_client.py                      # LLM API客户端
├── evaluator.py                       # 评估器
├── main.py                            # 主程序
├── prompt_engineering_pipeline.ipynb  # 完整流程演示
└── outputs/                           # 输出目录
    ├── processed_data/                # 预处理数据
    ├── generated_titles/              # 生成的标题
    └── evaluation_results/            # 评估结果
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt
```

### 2. 配置API密钥

```bash
# 复制配置模板
cp api_config_template.py api_config.py

# 编辑配置文件，将YOUR_API_KEY_HERE替换为您的实际API密钥
```

### 3. 运行系统

**方式1：使用Jupyter Notebook（推荐）**
```bash
jupyter notebook prompt_engineering_pipeline.ipynb
```

**方式2：使用Python脚本**
```bash
# 运行主程序
python main.py
```

## 🔧 支持的模型

当前支持以下模型（可在`api_config.py`中配置）：

- **deepseek-chat-v3-0324** (默认) - 聊天模型，适合标题生成
- **deepseek-r1-0528** - 推理模型，用于质量评估

用户可以根据需要在配置文件中添加更多模型。

## 📊 评估体系

### 增强版个性化评估

1. **兴趣匹配度** (40%权重) - 标题与用户兴趣的匹配程度
2. **类别相关性** (20%权重) - 标题与新闻类别的相关性  
3. **兴趣一致性** (25%权重) - 用户兴趣与新闻类别的一致性
4. **历史相关性** (15%权重) - 考虑用户历史阅读偏好

### LLM质量评估

使用推理模型进行5维度评估：
- 准确性、吸引力、清晰度、合理性、创新性

### 综合评分

- ROUGE评分：25%
- 个性化效果：35%
- 标题质量：20%
- LLM评估：20%

## 📈 性能表现

### 评估指标改进

| 评估指标 | 原版本 | 增强版 | 改进说明 |
|---------|--------|--------|----------|
| 个性化匹配度 | ~0.01 | ~0.35+ | 评估算法科学化 |
| 类别相关性 | ~0.00 | ~0.42+ | 新增语义匹配 |
| 标题质量评估 | 基于规则 | LLM评估 | 更客观准确 |
| 综合评分 | 0.31 | 0.45+ | 多维度评估 |

### 系统优势

- **🔒 安全性**：API密钥安全存储，不会意外泄露
- **⚡ 效率**：智能批量处理，成本可控
- **🎯 准确性**：改进的评估方法更科学客观
- **🔧 灵活性**：支持多模型，便于横向比较
- **📊 全面性**：多维度评估，综合反映系统性能

## 💡 使用建议

1. **首次使用**：运行完整的Jupyter Notebook流程
2. **日常开发**：使用main.py进行批量处理
3. **模型对比**：在配置文件中添加新模型进行比较
4. **效果优化**：根据评估结果调整提示词设计

## 🔍 技术细节

### API配置安全性

- 敏感信息存储在独立文件中
- 自动添加到.gitignore，防止意外提交
- 支持环境变量配置

### 多模型支持

```python
# 动态切换模型
from llm_client import LLMClient
client = LLMClient(model_name="deepseek-r1-0528")
client.switch_model("deepseek-chat-v3-0324")
```

### 增强版评估

```python
# 使用评估器
from evaluator import Evaluator
evaluator = Evaluator(use_llm_evaluation=True)
results = evaluator.comprehensive_evaluation(data)
```

## 🚧 未来规划

- [ ] 支持更多大模型API
- [ ] 实现在线学习和反馈优化
- [ ] 增加多语言支持
- [ ] 开发Web界面
- [ ] 集成更多评估指标

## 📝 更新日志

### v2.0.0 (当前版本)
- ✅ 重构API配置管理，提升安全性
- ✅ 添加多模型支持和动态切换
- ✅ 改进个性化评估算法（4维度评估）
- ✅ 引入LLM质量评估功能
- ✅ 升级综合评分体系
- ✅ 优化Jupyter Notebook流程

### v1.0.0 (原始版本)
- 基础个性化标题生成功能
- ROUGE评估支持
- 简单的个性化效果评估

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进本项目！

## 📄 许可证

本项目采用MIT许可证。 