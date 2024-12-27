# Desktop Live2D AI Assistant
(这是乱写的别看)
这是一个基于 Live2D 的桌面 AI 助手应用，集成了以下功能：
- Live2D 模型展示
- 大语言模型对话功能
- 阿里funasr语音识别服务

## 环境要求
- Python 3.8+
- 阿里funasr（用于语音识别服务）
- OpenAI API Key（用于大语言模型服务）

## 安装
1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 配置环境变量：
创建 `.env` 文件并添加以下配置：
```
OPENAI_API_KEY=你的OpenAI API密钥
```

## 运行
```bash
python main.py
``` 
