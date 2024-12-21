from openai import OpenAI
import os
from dotenv import load_dotenv

def test_chatgpt():
    # 加载环境变量
    load_dotenv(override=True)
    
    # 初始化 OpenAI 客户端
    client = OpenAI()
    
    try:
        # 发送测试请求
        completion = client.chat.completions.create(
            model="gpt-4o",  # 使用稳定的模型
            messages=[
                {"role": "system", "content": "你是一个友好的AI助手。"},
                {"role": "user", "content": "你好，这是一个测试消息。"}
            ],
            max_tokens=50  # 限制回复长度
        )
        
        # 打印响应信息
        print("连接成功！")
        print(f"模型响应: {completion.choices[0].message.content}")
        print(f"完整响应: {completion}")
        
    except Exception as e:
        print(f"连接失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_chatgpt()